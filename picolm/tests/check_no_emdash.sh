#!/bin/sh
# check_no_emdash.sh -- assert a GBNF grammar keeps the em dash out of the output.
#
# WHY THIS EXISTS
# ---------------
# Chat-tuned models love the em dash (U+2014). Ask for one and you get three;
# ask for a plain paragraph and you still get them for asides. ~/noshit.gbnf
# describes the allowed character set explicitly -- letters (ASCII +
# Hungarian), digits, punctuation, linebreaks -- and U+2014 is simply not a
# member of any class, so a correct grammar engine must reject it.
#
# This is a good end-to-end test of the GBNF engine because:
#   * it exercises multi-byte UTF-8 (the model emits "—" as 3 bytes, often as
#     part of a longer token, so partial-UTF8 handling must be right);
#   * it exercises a recursive rule (en-char+ with a repeated group), i.e. the
#     pushdown stacks, not just a flat alternation;
#   * it has an unmistakable pass/fail signal that does not depend on judging
#     the quality of the prose.
#
# USAGE
#   tests/check_no_emdash.sh <model.gguf> [picolm] [grammar] [runs]
# Exit: 0 = no em dash in any run, 1 = an em dash got through, 2 = setup error.

set -u

MODEL="${1:?usage: check_no_emdash.sh <model.gguf> [picolm] [grammar] [runs]}"
BIN="${2:-./picolm}"
GRAMMAR="${3:-$HOME/noshit.gbnf}"
RUNS="${4:-5}"

[ -x "$BIN" ] || { echo "check_no_emdash: $BIN is not executable" >&2; exit 2; }
[ -r "$MODEL" ] || { echo "check_no_emdash: cannot read $MODEL" >&2; exit 2; }
[ -r "$GRAMMAR" ] || { echo "check_no_emdash: cannot read $GRAMMAR" >&2; exit 2; }

# The prompt that reliably draws em dashes out of a model. Written to provoke
# them explicitly, so a pass cannot be blamed on the model not trying.
PROMPT='Write one sentence about the sea, using em dashes (—) for every aside, at least three of them.'

echo "model:   $MODEL"
echo "grammar: $GRAMMAR"
echo "runs:    $RUNS"
echo

bad=0
n=0
while [ "$n" -lt "$RUNS" ]; do
    n=$((n + 1))
    OUT="$("$BIN" "$MODEL" -c 512 -n 120 -t 0.9 -pf "$PROMPT" --grammar "$GRAMMAR" 2>&1 | grep '^OUTPUT:' || true)"
    if [ -z "$OUT" ]; then
        # An empty run would "pass" the em-dash check for the wrong reason.
        echo "RUN $n: FAIL -- produced no output at all"
        bad=$((bad + 1))
        continue
    fi
    if printf '%s' "$OUT" | grep -q '—'; then
        echo "RUN $n: FAIL -- em dash emitted"
        printf '%s\n\n' "$OUT"
        bad=$((bad + 1))
    else
        # Show the text so a "pass" that produced nothing is visible.
        printf 'RUN %d: ok -- %s\n\n' "$n" "$(printf '%s' "$OUT" | cut -c1-160)"
    fi
done

if [ "$bad" -ne 0 ]; then
    echo "FAILED: $bad of $RUNS runs contained an em dash"
    exit 1
fi
echo "PASSED: no em dash in $RUNS runs"
exit 0
