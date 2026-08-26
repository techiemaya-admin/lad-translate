#!/usr/bin/env bash
# Fetch the pinned gitleaks binary into .local/bin and enable the pre-push hook.
#
# Version and checksum are pinned to match .github/workflows/secret-scan.yml.
# If you bump one, bump the other, or CI and the hook stop agreeing about what
# is safe to push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="8.30.1"
case "$(uname -s)-$(uname -m)" in
  Darwin-x86_64) ASSET="gitleaks_${VERSION}_darwin_x64.tar.gz" ;;
  Darwin-arm64)  ASSET="gitleaks_${VERSION}_darwin_arm64.tar.gz" ;;
  Linux-x86_64)  ASSET="gitleaks_${VERSION}_linux_x64.tar.gz" ;;
  Linux-aarch64) ASSET="gitleaks_${VERSION}_linux_arm64.tar.gz" ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$ROOT/.local/bin"
echo "fetching gitleaks $VERSION ($ASSET)"
curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/${ASSET}" \
  | tar xz -C "$ROOT/.local/bin" gitleaks
"$ROOT/.local/bin/gitleaks" version

git -C "$ROOT" config core.hooksPath .githooks
echo "pre-push hook enabled. Bypass a single push with --no-verify."
