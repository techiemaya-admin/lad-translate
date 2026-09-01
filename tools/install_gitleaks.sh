#!/usr/bin/env bash
# Fetch the pinned gitleaks binary into .local/bin and enable the pre-push hook.
#
# Version and checksums are pinned to match .github/workflows/secret-scan.yml.
# If you bump one, bump the other, or CI and the hook stop agreeing about what
# is safe to push.
#
# The download is verified before it runs. CI already did this and this script
# did not, which is the wrong way round: the hook is the gate that matters,
# because CI only catches a secret AFTER it has reached GitHub. A security gate
# that executes an unverified binary is not a security gate.
#
# Hashes come from the release's own gitleaks_${VERSION}_checksums.txt. The
# linux_x64 entry there matches the value secret-scan.yml already pinned, which
# is what establishes the file is the right one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="8.30.1"
BIN_NAME="gitleaks"

case "$(uname -s)-$(uname -m)" in
  Darwin-x86_64)
    ASSET="gitleaks_${VERSION}_darwin_x64.tar.gz"
    SHA256="dfe101a4db2255fc85120ac7f3d25e4342c3c20cf749f2c20a18081af1952709" ;;
  Darwin-arm64)
    ASSET="gitleaks_${VERSION}_darwin_arm64.tar.gz"
    SHA256="b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5" ;;
  Linux-x86_64)
    ASSET="gitleaks_${VERSION}_linux_x64.tar.gz"
    SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" ;;
  Linux-aarch64)
    ASSET="gitleaks_${VERSION}_linux_arm64.tar.gz"
    SHA256="e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080" ;;
  # Git for Windows. uname reports MINGW64_NT-10.0-<build>, and the release
  # ships a zip rather than a tarball. The hook itself needs no change: MSYS
  # resolves gitleaks to gitleaks.exe on its own, including for `test -x`.
  MINGW*-x86_64|MSYS*-x86_64)
    ASSET="gitleaks_${VERSION}_windows_x64.zip"
    SHA256="d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e"
    BIN_NAME="gitleaks.exe" ;;
  *)
    echo "unsupported platform: $(uname -s)-$(uname -m)" >&2
    echo "add its asset and checksum from gitleaks_${VERSION}_checksums.txt" >&2
    exit 1 ;;
esac

mkdir -p "$ROOT/.local/bin"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching gitleaks $VERSION ($ASSET)"
curl -sSfL -o "$TMP/$ASSET" \
  "https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/${ASSET}"

# Verify BEFORE unpacking, so a tampered archive is never even expanded.
if command -v sha256sum >/dev/null 2>&1; then
  echo "${SHA256}  ${TMP}/${ASSET}" | sha256sum -c - >/dev/null
elif command -v shasum >/dev/null 2>&1; then
  echo "${SHA256}  ${TMP}/${ASSET}" | shasum -a 256 -c - >/dev/null
else
  echo "no sha256sum or shasum available; refusing to run an unverified binary" >&2
  exit 1
fi
echo "checksum ok"

case "$ASSET" in
  *.zip)    unzip -q -o "$TMP/$ASSET" "$BIN_NAME" -d "$TMP" ;;
  *.tar.gz) tar xzf "$TMP/$ASSET" -C "$TMP" "$BIN_NAME" ;;
esac
mv "$TMP/$BIN_NAME" "$ROOT/.local/bin/$BIN_NAME"
chmod +x "$ROOT/.local/bin/$BIN_NAME"
"$ROOT/.local/bin/$BIN_NAME" version

git -C "$ROOT" config core.hooksPath .githooks
echo "pre-push hook enabled. Bypass a single push with --no-verify."
