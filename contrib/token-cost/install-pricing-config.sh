#!/usr/bin/env bash
# Move the token cost report's pricing, exchange rate and timezone out of the
# code and into an editable file, and warn on the page when they go stale.
#
# Changes nothing about how the page looks or where it lives. Same tab, same
# numbers today. The difference is that tomorrow you can correct them without
# editing Python, and the page tells you when they are old.
#
#   bash install-pricing-config.sh

set -Eeuo pipefail

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

REPO="$HOME/open-chat-reviewer"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET="$REPO/src/chatreview/token_panel.py"
CONFIG="$REPO/.chatreview/token-pricing.json"

[[ -d "$REPO" ]]                     || die "Repository not found at $REPO"
[[ -f "$HERE/token_panel_v2.py" ]]   || die "token_panel_v2.py must sit beside this script"
[[ -f "$HERE/token-pricing.json" ]]  || die "token-pricing.json must sit beside this script"
[[ -f "$TARGET" ]]                   || die "$TARGET is missing; run install-token-tab.sh first"

say "Backing up the current module"
cp -f "$TARGET" "$TARGET.bak.$(date +%Y%m%d%H%M%S)"

say "Installing the updated module"
cp -f "$HERE/token_panel_v2.py" "$TARGET"
# keep the copy used by install-token-tab.sh in step, so re-running it cannot regress this
[[ -f "$HOME/token-tab/token_panel.py" ]] && cp -f "$HERE/token_panel_v2.py" "$HOME/token-tab/token_panel.py"

say "Writing the pricing file"
mkdir -p "$REPO/.chatreview"
if [[ -f "$CONFIG" ]]; then
    printf 'already present, leaving your edits alone: %s\n' "$CONFIG"
else
    cp "$HERE/token-pricing.json" "$CONFIG"
    chmod 600 "$CONFIG"
    printf 'created %s\n' "$CONFIG"
fi

say "Checking the timezone setting"
# shellcheck disable=SC1090
set -a; [[ -f "$REPO/.chatreview/archive.env" ]] && source "$REPO/.chatreview/archive.env"; set +a
printf 'CHATREVIEW_TIMEZONE=%s\n' "${CHATREVIEW_TIMEZONE:-unset, the report will use UTC}"

say "Restarting the web service"
if systemctl --user is-active open-chat-reviewer-web.service >/dev/null 2>&1; then
    systemctl --user restart open-chat-reviewer-web.service
    ok=0
    for _ in $(seq 1 45); do
        curl -fsS --max-time 5 "http://127.0.0.1:8765/token-report" -o /tmp/token-report-check.html 2>/dev/null && { ok=1; break; }
        sleep 2
    done
    if [[ "$ok" == "1" ]]; then
        printf '\n\033[1;32mDone.\033[0m Page still serving.\n'
        grep -o "Last confirmed [^,]*" /tmp/token-report-check.html | head -1 || true
        grep -o "Timezone [A-Za-z/_]*" /tmp/token-report-check.html | head -1 || true
        rm -f /tmp/token-report-check.html
    else
        printf '\n\033[1;33mNot answering.\033[0m journalctl --user -u open-chat-reviewer-web -n 40\n'
    fi
else
    printf 'web service is not running; start it and the change takes effect\n'
fi

cat <<EOF

  Edit prices   $CONFIG
  Then just reload the page. No restart needed, it is read on every request.
  Update as_at when you confirm the numbers; after 90 days the page warns you.
  A bad or missing file falls back to the built in defaults and says so.

  Revert        cp "$TARGET".bak.* "$TARGET" && systemctl --user restart open-chat-reviewer-web

EOF
