#!/usr/bin/env python3
"""cmp_engine.py -- compare PicoLM logits against the ik_llama.cpp oracle.

Every past harness mistake is checked, not assumed:

  * token ids  -- both engines are fed the SAME ids (PicoLM's own tokenizer
                  produces them, unless --tokens is given).
  * positions  -- ik's QORACLE_LOGITS records the pos it dumped; PicoLM's dump
                  carries pos= too. We pair by pos, never by order, and we
                  FAIL if the requested pos is missing on either side.
  * alignment  -- the prompt's last-token logits are at pos = n_prompt-1. If
                  the two sides disagree on which pos that is, we say so.
  * numpy reentry -- stray modules in cwd shadow stdlib; sys.path[0] is ''

usage:
  cmp_engine.py --model M --prompt "text" [--ntok N] [--pos P] [--gen K]
                [--picolm-args "..."] [--picolm-env "K=V K2=V2"]
                [--tokens 1,2,3] [--quiet]

Exit 0 only if every compared position passed --atol (default 1e-3).
"""
import argparse, os, struct, subprocess, sys, tempfile

sys.path.insert(0, '')            # never let cwd shadow stdlib (see /tmp/bisect.py)
import numpy as np

PICOLM = os.environ.get('PICOLM_BIN', './picolm')
ORACLE = os.environ.get('ORACLE_BIN',
                        os.path.expanduser('~/ik_llama.cpp/build/bin/llama-glmoracle'))


def read_ik(path):
    """QORACLE_LOGITS: repeated [int32 pos][int32 nv][float32 x nv]."""
    out = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            pos, nv = struct.unpack('ii', h)
            body = f.read(4 * nv)
            if len(body) != 4 * nv:
                raise RuntimeError(f'oracle dump truncated at pos {pos}')
            out[pos] = np.frombuffer(body, dtype=np.float32).copy()
    return out


def read_pic(path):
    """PICOLM_DUMP_LOGITS: 'pos=.. vocab=..' then 'idx logit' per line."""
    out, pos, vals = {}, None, []
    with open(path, errors='replace') as f:
        for line in f:
            if line.startswith('pos='):
                if pos is not None:
                    out[pos] = np.array(vals, dtype=np.float32)
                pos = int(line.split()[0].split('=')[1])
                vals = []
            else:
                q = line.split()
                if len(q) == 2:
                    vals.append(float(q[1]))
    if pos is not None:
        out[pos] = np.array(vals, dtype=np.float32)
    return out


def tok_ids_of_prompt(model, prompt, extra_env):
    """Token ids from PicoLM itself, so the oracle cannot see a different prompt."""
    env = dict(os.environ)
    env['PICOLM_EXIT_AFTER_TOKENIZE'] = '1'
    env.update(extra_env)
    r = subprocess.run([PICOLM, model, '-c', '4096', '-p', prompt],
                       capture_output=True, text=True, errors='replace', env=env)
    for line in (r.stdout + r.stderr).splitlines():
        if 'Prompt tokens' in line:
            inside = line[line.index('(') + 1:line.index('):')]
            return int(inside), line.split(':', 1)[1].split()
    raise RuntimeError('could not read token ids from PicoLM:\n' + r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--tokens', default=None, help='explicit comma-separated ids (cross-check)')
    ap.add_argument('--bytes', default=None,
                    help='feed these raw byte values (comma separated) as the prompt; '
                         'used for synthetic models whose vocab is raw bytes')
    ap.add_argument('--gen', type=int, default=1, help='tokens to generate')
    ap.add_argument('--pos', default=None, help='only compare this pos')
    ap.add_argument('--atol', type=float, default=1e-3)
    ap.add_argument('--oracle-batch', type=int, default=0,
                    help='default: round ntok up to the next power of two (>=64)')
    ap.add_argument('--picolm-args', default='')
    ap.add_argument('--picolm-env', default='')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = os.path.expanduser('~/gcc-15/lib64')
    env.update({'QORACLE_STATS': 'none'})
    penv = dict(os.environ)
    for kv in a.picolm_env.split():
        k, _, v = kv.partition('=')
        penv[k] = v

    if a.bytes is not None:
        prompt_bytes = bytes(int(x) for x in a.bytes.split(','))
        ids = [str(x) for x in a.bytes.split(',')]
        if a.tokens and a.tokens.split(',') != ids:
            print(f'TOKEN MISMATCH: --tokens says {a.tokens} vs --bytes {a.bytes}',
                  file=sys.stderr)
            return 2
    else:
        if a.prompt is None:
            print('need --prompt (or --bytes for a synthetic raw-byte prompt)',
                  file=sys.stderr)
            return 2
        prompt_bytes = a.prompt.encode()
        n, ids = tok_ids_of_prompt(a.model, a.prompt, penv)
        if a.tokens and a.tokens.split(',') != ids:
            print(f'TOKEN MISMATCH: --tokens says {a.tokens} but the tokenizer says '
                  f'{",".join(ids)}', file=sys.stderr)
            return 2
    ntok = len(ids)
    if not a.quiet:
        print(f'[{ntok} tokens] {" ".join(ids)}', file=sys.stderr)

    d = tempfile.mkdtemp(prefix='cmpeng_', dir=os.environ.get('CMP_TMP', '/kv'))
    ik_bin, pic_txt = os.path.join(d, 'ik.bin'), os.path.join(d, 'pic.txt')

    # --- oracle: prompt + generation, so pos pairs up with PicoLM's dumps ---
    oenv = dict(env)
    oenv.update({'QORACLE_TOKENS': ','.join(ids),
                 'QORACLE_LOGITS': ik_bin,
                 'QORACLE_N': str(a.gen)})
    if a.pos is not None:
        oenv['QORACLE_POS'] = a.pos
    # ik aborts with GGML_ASSERT(n_tokens_all <= cparams.n_batch) if the prompt
    # does not fit one batch; --oracle-batch=0 sizes it from ntok.
    nb = a.oracle_batch
    if nb <= 0:
        nb = 64
        while nb < ntok + a.gen:
            nb *= 2
    b = str(nb)
    r = subprocess.run([ORACLE, '-m', a.model, '-c', '4096', '-n', str(a.gen),
                        '-t', '0', '-b', b, '-ub', b],
                       capture_output=True, text=True, errors='replace', env=oenv)
    if not os.path.exists(ik_bin):
        print('ORACLE FAILED:\n' + r.stdout[-2000:] + r.stderr[-2000:], file=sys.stderr)
        return 2
    ik = read_ik(ik_bin)
    ik_gen = [int(l.split('argmax=')[1]) for l in r.stdout.splitlines()
              if l.startswith('gen[')]

    # --- picolm: same prompt text via stdin (byte-exact, no argv escaping) ---
    penv['PICOLM_DUMP_LOGITS'] = pic_txt
    args = [PICOLM, a.model, '-c', '4096', '-n', str(a.gen), '-t', '0']
    args += a.picolm_args.split()
    r = subprocess.run(args, input=prompt_bytes, capture_output=True, env=penv)
    r = subprocess.CompletedProcess(r.args, r.returncode,
                                    r.stdout.decode('utf-8', 'replace'),
                                    r.stderr.decode('utf-8', 'replace'))
    if not os.path.exists(pic_txt):
        print('PICOLM FAILED:\n' + r.stdout[-2000:] + r.stderr[-2000:], file=sys.stderr)
        return 2
    pic = read_pic(pic_txt)

    # --- harness self-check: the sets of positions must intersect as expected ---
    only_ik = sorted(set(ik) - set(pic))
    only_pic = sorted(set(pic) - set(ik))
    common = sorted(set(ik) & set(pic))
    if not common:
        print(f'NO OVERLAPPING POSITIONS: ik={sorted(ik)} pic={sorted(pic)}', file=sys.stderr)
        return 2
    if not a.quiet and (only_ik or only_pic):
        print(f'note: pos only in ik {only_ik}, only in picolm {only_pic}', file=sys.stderr)

    worst_cos, worst_ma, bad, rows = 1.0, 0.0, 0, []
    for pos in common:
        x, y = ik[pos], pic[pos]
        if x.shape != y.shape:
            print(f'pos {pos}: SHAPE MISMATCH ik={x.shape} pic={y.shape}', file=sys.stderr)
            return 2
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        ma = float(np.abs(x - y).max())
        ti, tp = int(x.argmax()), int(y.argmax())
        worst_cos, worst_ma = min(worst_cos, cos), max(worst_ma, ma)
        if ma > a.atol:
            bad += 1
        rows.append((pos, cos, ma, ti, tp))

    for pos, cos, ma, ti, tp in rows:
        flag = '' if ma <= a.atol else '  <-- FAIL'
        print(f'pos {pos:5d}: cos={cos:.9f} maxabs={ma:.3e} top ik={ti} pic={tp}'
              + (' MISMATCH' if ti != tp else '') + flag)
    if ik_gen:
        pgen = [int(t) for t in
                (l.split(']')[1].split('=')[0].strip() for l in r.stderr.splitlines()
                 if 'TOPDEC' in l)]
        if pgen:
            ok = pgen[:len(ik_gen)] == ik_gen[:len(pgen)]
            print(f'greedy tokens: {"MATCH" if ok else "DIFF"}  ik={ik_gen}  pic={pgen}')

    if not a.keep:
        for f in (ik_bin, pic_txt):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass
    print(f'\nworst cos={worst_cos:.9f} worst maxabs={worst_ma:.3e} '
          f'({bad}/{len(rows)} positions over atol={a.atol})')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
