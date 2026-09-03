#!/usr/bin/env bash
# Install AutoBayes as Claude Code skills.
#
#   ./install.sh              # symlink into ~/.claude/skills (default)
#   ./install.sh --copy       # copy instead, if you would rather not symlink
#   ./install.sh --uninstall
#
# Symlinking is the default so that `git pull` in this repo updates the installed
# skills. Use --copy if you want them pinned.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
MODE="link"

for arg in "$@"; do
  case "$arg" in
    --copy)      MODE="copy" ;;
    --uninstall) MODE="uninstall" ;;
    -h|--help)   sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

# The repo root is itself the `autobayes` skill (SKILL.md + reference/ + scaffold/).
# Companion skills live one per directory under skills/.
install_one() {
  local src="$1" name="$2" dest="$SKILLS_DIR/$2"

  if [ -e "$dest" ] || [ -L "$dest" ]; then
    if [ -L "$dest" ]; then
      rm "$dest"
    else
      echo "  ! $name exists and is not a symlink — leaving it alone" >&2
      return 0
    fi
  fi

  case "$MODE" in
    link) ln -s "$src" "$dest"; echo "  + $name -> $src" ;;
    copy) cp -R "$src" "$dest"; echo "  + $name (copied)" ;;
  esac
}

if [ "$MODE" = "uninstall" ]; then
  echo "Removing AutoBayes skills from $SKILLS_DIR"
  for name in autobayes $(ls "$REPO/skills" 2>/dev/null || true); do
    dest="$SKILLS_DIR/$name"
    if [ -L "$dest" ]; then rm "$dest"; echo "  - $name"
    elif [ -d "$dest" ]; then echo "  ! $name is a real directory, not removing" >&2
    fi
  done
  exit 0
fi

mkdir -p "$SKILLS_DIR"
echo "Installing AutoBayes skills into $SKILLS_DIR"

install_one "$REPO" "autobayes"
for skill_dir in "$REPO"/skills/*/; do
  [ -d "$skill_dir" ] || continue
  install_one "${skill_dir%/}" "$(basename "$skill_dir")"
done

cat <<'EOF'

Installed. Two ways to use it:

  Start a project     ask your agent to "set up a Bayesian modelling project",
                      or run:  bin/autobayes-init /path/to/project

  Mid-project help    the skills are symptom-triggered. When a gradient comes back
                      nan or a likelihood will not differentiate, autobayes-custom-op
                      loads on its own.

Read reference/README.md first — it is the part that saves you the most time.
EOF
