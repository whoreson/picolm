#!/usr/bin/env python3
"""Compare two PICOLM_DUMP_LOGITS files and report the worst divergence.

Why this exists
---------------
tools/oracle_check.sh used to compare *generated text* by md5. That is far
stricter than float arithmetic allows: two kernels that compute the same thing
in a different summation order agree to ~1e-6, but a greedy decode can flip a
token on a ~1e-2 logit shift and turn an agreement into a byte mismatch. The
check therefore passed only as long as the model's numerics happened to sit on
the "stable" side of a token boundary, and it started failing the moment the SSM
recurrence's default chunk size changed.

Comparing logits with a tolerance asserts what the check is actually for: a
kernel-selection toggle, or a SIMD build versus the scalar reference, must not
change the model's *beliefs* materially. It still catches every real bug the
hash caught (a kernel that reads the wrong row, wrong stride, wrong block size
or wrong quantization all move logits by orders of magnitude more than 1e-3).

Exit codes: 0 = agree within tolerance, 1 = disagree, 2 = bad input.
"""

import sys

# Absolute tolerance on the logit scale. Reference: logits here are O(1)-O(20)
# on a tiny random-weight model and O(1)-O(20) on real ones. 1e-3 is ~1000x the
# float32 reassociation noise we expect (~1e-6) and still ~10x below the
# smallest real defect we have seen (a wrong row or stride moves logits by
# >=1e-2). Override with --atol for a stricter or looser run.
DEFAULT_ATOL = 1e-3


def read_dump(path):
    """Parse a PICOLM_DUMP_LOGITS file into {pos: {token: logit}}."""
    out = {}
    pos = None
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("pos="):
                # pos=<n> vocab=<v>
                fields = dict(kv.split("=", 1) for kv in line.split())
                pos = int(fields["pos"])
                out[pos] = {}
                continue
            if pos is None:
                continue
            tok, val = line.split()
            out[pos][int(tok)] = float(val)
    return out


def main(argv):
    atol = DEFAULT_ATOL
    args = []
    i = 1
    while i < len(argv):
        if argv[i] == "--atol":
            i += 1
            atol = float(argv[i])
        else:
            args.append(argv[i])
        i += 1

    if len(args) != 2:
        sys.stderr.write("usage: logits_diff.py [--atol X] A.log B.log\n")
        return 2

    a = read_dump(args[0])
    b = read_dump(args[1])

    if not a or not b:
        sys.stderr.write("error: one of the dumps is empty\n")
        return 2
    if set(a) != set(b):
        sys.stdout.write(
            "FAIL positions differ: only-A=%s only-B=%s\n"
            % (sorted(set(a) - set(b))[:8], sorted(set(b) - set(a))[:8])
        )
        return 1

    worst = 0.0
    worst_pos = None
    worst_tok = None
    for pos in sorted(a):
        if set(a[pos]) != set(b[pos]):
            sys.stdout.write("FAIL pos=%d vocab differs\n" % pos)
            return 1
        for tok, va in a[pos].items():
            d = abs(va - b[pos][tok])
            if d > worst:
                worst, worst_pos, worst_tok = d, pos, tok

    # Top-1 agreement is worth reporting separately: it is what actually decides
    # a greedy decode, and a large top-1 gap is a much stronger smell than a
    # single large element elsewhere in the distribution.
    top1_mismatch = []
    for pos in sorted(a):
        ta = max(a[pos].items(), key=lambda kv: kv[1])[0]
        tb = max(b[pos].items(), key=lambda kv: kv[1])[0]
        if ta != tb:
            top1_mismatch.append((pos, ta, tb))

    if worst <= atol:
        sys.stdout.write(
            "ok   max|dlogit|=%.3e (atol %.0e)%s\n"
            % (worst, atol, "" if not top1_mismatch else
               "; NOTE top-1 differs at pos %s" % [p for p, _, _ in top1_mismatch])
        )
        return 0

    sys.stdout.write(
        "FAIL max|dlogit|=%.3e > atol %.0e at pos=%d token=%d\n"
        % (worst, atol, worst_pos, worst_tok)
    )
    for pos, ta, tb in top1_mismatch[:5]:
        sys.stdout.write("     top-1 differs at pos=%d: %d vs %d\n" % (pos, ta, tb))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
