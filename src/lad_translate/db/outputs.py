"""
Hardware output storage.

Device profiles and their channel maps, in one tenant's schema. Same rule as
db/sessions.py: every statement filters on tenant_id, including the ones where
the primary key alone would find the row. A device_id identifies a device, so a
bug that passes the wrong tenant_id would return another venue's patch; with
the filter it returns nothing.

Writes go through save_device(), which replaces a device's whole channel map in
one transaction rather than accepting per-channel edits. Two reasons. The map
has cross-row invariants -- no two languages on a wire, no two on an IR channel
-- and checking those against a half-applied map means deciding whether the
row being moved counts as occupied. And an operator dragging channels around in
the portal is describing an end state, not a sequence of moves: sending the end
state means a dropped request leaves the previous patch intact instead of a
half-repatched rig.
"""

from __future__ import annotations

import uuid

from ..config import OutputChannel, OutputDevice, TenantContext
from ..obs.log import get_logger
from .tenancy import validate_schema

log = get_logger(__name__)

NAME_CONSTRAINT = "audio_output_devices_tenant_name_unique"


class DuplicateDeviceName(ValueError):
    """
    Two devices for one tenant claiming the same name.

    Its own type because reusing a name is an ordinary operator slip with an
    obvious fix, and it should reach them as that rather than as a driver-level
    integrity error. Raised here so callers never have to know which database
    is underneath or what its constraints are called.
    """


class OutputStore:
    """Reads and writes audio output profiles in one tenant's schema."""

    def __init__(self, pool, tenant: TenantContext) -> None:
        self._pool = pool
        self._tenant = tenant
        self._schema = validate_schema(tenant.schema)

    @property
    def tenant_id(self) -> str:
        return self._tenant.tenant_id

    async def list_devices(self) -> list[OutputDevice]:
        """Every device for this tenant, channel maps included."""
        async with self._pool.acquire() as conn:
            devices = await conn.fetch(
                f"""SELECT device_id::text, name, kind, device_name, channel_count,
                           sample_rate, enabled
                    FROM {self._schema}.audio_output_devices
                    WHERE tenant_id = $1::uuid
                    ORDER BY name""",
                self.tenant_id,
            )
            if not devices:
                return []

            # One query for every channel rather than one per device: a venue
            # with eight profiles is eight extra round trips for no gain.
            channels = await conn.fetch(
                f"""SELECT device_id::text, language, channel, ir_channel, label,
                           gain_db, enabled
                    FROM {self._schema}.audio_output_channels
                    WHERE tenant_id = $1::uuid
                    ORDER BY channel""",
                self.tenant_id,
            )

        by_device: dict[str, list[OutputChannel]] = {}
        for row in channels:
            by_device.setdefault(row["device_id"], []).append(
                OutputChannel(
                    language=row["language"],
                    channel=row["channel"],
                    ir_channel=row["ir_channel"],
                    label=row["label"],
                    gain_db=row["gain_db"],
                    enabled=row["enabled"],
                )
            )

        return [self._device(row, tuple(by_device.get(row["device_id"], ()))) for row in devices]

    async def get_device(self, device_id: str) -> OutputDevice | None:
        """One device with its channel map, or None if this tenant has no such device."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT device_id::text, name, kind, device_name, channel_count,
                           sample_rate, enabled
                    FROM {self._schema}.audio_output_devices
                    WHERE device_id = $1::uuid AND tenant_id = $2::uuid""",
                device_id,
                self.tenant_id,
            )
            if row is None:
                return None
            channels = await conn.fetch(
                f"""SELECT language, channel, ir_channel, label, gain_db, enabled
                    FROM {self._schema}.audio_output_channels
                    WHERE device_id = $1::uuid AND tenant_id = $2::uuid
                    ORDER BY channel""",
                device_id,
                self.tenant_id,
            )

        return self._device(
            row,
            tuple(
                OutputChannel(
                    language=c["language"],
                    channel=c["channel"],
                    ir_channel=c["ir_channel"],
                    label=c["label"],
                    gain_db=c["gain_db"],
                    enabled=c["enabled"],
                )
                for c in channels
            ),
        )

    async def save_device(self, device: OutputDevice) -> OutputDevice:
        """
        Create or replace a device and its entire channel map, atomically.

        `device` has already validated its own invariants in __post_init__, so
        what reaches SQL is a map that is internally consistent. The unique
        constraints are still there as the last word, because this is not the
        only thing that can write these tables.
        """
        device_id = device.device_id or str(uuid.uuid4())

        async with self._pool.acquire() as conn, conn.transaction():
            try:
                await self._upsert_device(conn, device_id, device)
            except Exception as exc:
                # asyncpg names the constraint on the exception, so the clash
                # is identified without importing its exception types or
                # matching on message text.
                if getattr(exc, "constraint_name", "") == NAME_CONSTRAINT:
                    raise DuplicateDeviceName(
                        f"another device is already called {device.name!r}"
                    ) from exc
                raise

            # ON CONFLICT ... WHERE tenant_id = ours means a device_id owned by
            # another tenant updates no row and reports nothing. Catch it here
            # rather than go on to write channels against a device we do not own.
            owned = await conn.fetchval(
                f"""SELECT 1 FROM {self._schema}.audio_output_devices
                    WHERE device_id = $1::uuid AND tenant_id = $2::uuid""",
                device_id,
                self.tenant_id,
            )
            if not owned:
                raise LookupError(f"no device {device_id} for this tenant")

            # Replace the map wholesale. Deleting first means a channel freed in
            # this edit is free for another language in the same edit, which a
            # row-by-row upsert would reject on the unique constraint.
            await conn.execute(
                f"""DELETE FROM {self._schema}.audio_output_channels
                    WHERE device_id = $1::uuid AND tenant_id = $2::uuid""",
                device_id,
                self.tenant_id,
            )
            for channel in device.channels:
                await conn.execute(
                    f"""INSERT INTO {self._schema}.audio_output_channels
                            (channel_id, device_id, tenant_id, language, channel,
                             ir_channel, label, gain_db, enabled)
                        VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9)""",
                    str(uuid.uuid4()),
                    device_id,
                    self.tenant_id,
                    channel.language,
                    channel.channel,
                    channel.ir_channel,
                    channel.label,
                    channel.gain_db,
                    channel.enabled,
                )

        log.info(
            "output device saved",
            extra={
                "device_id": device_id,
                "device": device.name,
                "kind": device.kind,
                "channels": len(device.channels),
                "languages": device.languages,
            },
        )
        saved = await self.get_device(device_id)
        if saved is None:  # pragma: no cover - the transaction above committed it
            raise LookupError(f"device {device_id} vanished immediately after save")
        return saved

    async def delete_device(self, device_id: str) -> bool:
        """Remove a device and its channels. False when this tenant has no such device."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""DELETE FROM {self._schema}.audio_output_devices
                    WHERE device_id = $1::uuid AND tenant_id = $2::uuid""",
                device_id,
                self.tenant_id,
            )
        deleted = result.endswith(" 1")
        if deleted:
            log.info("output device deleted", extra={"device_id": device_id})
        return deleted

    async def _upsert_device(self, conn, device_id: str, device: OutputDevice) -> None:
        """
        Write the device row itself.

        Split out so the caller can name the one failure worth translating --
        a duplicate name -- without wrapping the whole transaction in a try and
        having to decide what every other error means.
        """
        await conn.execute(
            f"""INSERT INTO {self._schema}.audio_output_devices
                    (device_id, tenant_id, name, kind, device_name,
                     channel_count, sample_rate, enabled)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (device_id) DO UPDATE SET
                    name          = EXCLUDED.name,
                    kind          = EXCLUDED.kind,
                    device_name   = EXCLUDED.device_name,
                    channel_count = EXCLUDED.channel_count,
                    sample_rate   = EXCLUDED.sample_rate,
                    enabled       = EXCLUDED.enabled,
                    updated_at    = now()
                WHERE {self._schema}.audio_output_devices.tenant_id = $2::uuid""",
            device_id,
            self.tenant_id,
            device.name,
            device.kind,
            device.device_name,
            device.channel_count,
            device.sample_rate,
            device.enabled,
        )

    def _device(self, row, channels: tuple[OutputChannel, ...]) -> OutputDevice:
        return OutputDevice(
            device_id=row["device_id"],
            name=row["name"],
            kind=row["kind"],
            device_name=row["device_name"],
            channel_count=row["channel_count"],
            sample_rate=row["sample_rate"],
            enabled=row["enabled"],
            channels=channels,
        )
