#!/bin/bash
# SessionStart hook: once per day, ask Claude for a literature review.
#
# If docs/literature_review.md has no "## YYYY-MM-DD" heading for today, emit
# additionalContext telling Claude to do the review first; otherwise emit
# nothing, so resuming or compacting later the same day does not repeat it.
# Wired up in .claude/settings.local.json (per user, gitignored).
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
log="$root/docs/literature_review.md"
today=$(date +%F)
grep -q "^## $today" "$log" 2>/dev/null && exit 0

cat <<EOF
{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "DAILY LITERATURE REVIEW DUE ($today). The user asked for a literature review every day they log in, stored in docs/literature_review.md. Before other work (or alongside the user's first request if it is urgent): search for papers, preprints, code releases and datasets published since the last dated entry in that file, relevant to this project -- building footprint / instance segmentation in aerial imagery, Mask R-CNN and query-based instance segmentation (Mask2Former, Mask DINO), SAM / SAM 2 / SAM 3 fine-tuning and distillation, label-imagery misalignment and off-nadir footprint offset, dense small-object segmentation, IGN / French open data (BD TOPO, BD ORTHO, ORTHO Express, FLAIR). Read primary sources, not summaries. Append a new section headed '## $today' in the file's existing format: what was searched, each relevant item with link, date, one-line finding and a relevance/actionability verdict for this project, and 'nothing new' honestly if so. Commit it as mushalam and push. Tell the user in two or three lines what, if anything, matters."}}
EOF
