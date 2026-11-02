#!/usr/bin/env python3
"""JAX-defect rows of custom_ad and text_custom_ad, each with its proof; rows are listed, never dropped."""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "test", "golden", "custom_ad_defects.tsv")
SIGNATURE = re.compile(r"axis (-?\d+) of (argument \w+|args\[\d+\])")
NAME_LINE = re.compile(r"^(\s*name=)(\S+)$")

PROBE = r'''
import jax, jax.numpy as jnp
x = jnp.ones((2, 3))
def first(*a): return a[0] + a[1]
def second(*b): return b[0] + b[1]
def pair(p, q): return p + q
for g in (first, second, pair):
    try:
        jax.vmap(g, in_axes=(-1, 0))(x, x)
    except ValueError as e:
        print("vmap", g.__name__, repr(str(e)))
def foo(v): return jnp.sin(v)
def bar(v): return jnp.sin(v)
for inner in (foo, bar):
    h = jax.custom_vjp(inner)
    h.defvjp(lambda v: (inner(v), v), lambda r, c: (c,))
    h.__name__ = "g"
    w = jax.custom_jvp(h)
    w.defjvp(lambda p, t: (h(*p), t[0]))
    text = str(jax.make_jaxpr(w)(1.0))
    print("custom_jvp_of", inner.__name__, [l.strip() for l in text.splitlines() if l.strip().startswith("name=")])
grad = jax.custom_gradient(lambda v: (jnp.sin(v), lambda c: (c,)))
grad.__name__ = "g"
w = jax.custom_jvp(grad)
w.defjvp(lambda p, t: (grad(*p), t[0]))
text = str(jax.make_jaxpr(w)(1.0))
print("custom_jvp_of custom_gradient", [l.strip() for l in text.splitlines() if l.strip().startswith("name=")])
'''


def jax_line(relpath, needle):
    import jax

    path = os.path.join(os.path.dirname(jax.__file__), "_src", relpath)
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if needle in line:
                return f"jax/_src/{relpath}:{number}: {line.strip()}"
    return f"jax/_src/{relpath}: {needle}"


def unescape(text):
    return text.encode("utf-8").decode("unicode_escape")


def read_log(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6 and f[2] != "0":
                rows.append(f)
    return rows


def read_texts(reference, dump):
    theirs, ours = {}, {}
    with open(reference, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for field in ("jaxpr", "text"):
                if isinstance(r.get(field), str):
                    theirs[(r["set"].split("/")[-1], r["case"], field)] = r[field]
    with open(dump, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            suite, case = d["case"].split("/", 1)
            ours[(suite.split(":")[1], case, d["field"])] = d["v"]
    return theirs, ours


def name_only(theirs, ours):
    a, b = theirs.splitlines(), ours.splitlines()
    if len(a) != len(b):
        return None
    names = []
    for x, y in zip(a, b):
        if x == y:
            continue
        mx, my = NAME_LINE.match(x), NAME_LINE.match(y)
        if not (mx and my and mx.group(1) == my.group(1)):
            return None
        names.append((mx.group(2), my.group(2)))
    return names or None


def classify(log, reference, dump):
    signature = jax_line("api.py", "ba = inspect.signature(fn).bind(*args, **kwargs)")
    sourceinfo = jax_line("api_util.py", 'return f"{fun.__name__} at {filename}:{lineno}"')
    unwrap = jax_line("api_util.py", "fun = inspect.unwrap(fun)")
    wrapped = jax_line("custom_derivatives.py", "def wrapped_fun(*args, **kwargs):")
    theirs, ours = read_texts(reference, dump)
    out = []
    for f in read_log(log):
        suite, row = f[4], f[0]
        if suite.startswith("text_custom_ad:") and len(f) >= 8 and f[5] == "error_text":
            jax_text, ojax_text = unescape(f[6]), unescape(f[7])
            if "of argument " in jax_text and SIGNATURE.sub(r"axis \1 of ARG", jax_text) == SIGNATURE.sub(r"axis \1 of ARG", ojax_text):
                said = re.search(r"axis -?\d+ of argument \w+", jax_text).group(0)
                out.append((suite, row, "c", f"identical once the Python function's parameter names are removed: JAX names the vmapped function's parameter ({said}) from inspect.signature of the generator's def g(*a), at {signature}; renaming the Python parameter changes the text (see prove), and an OCaml function has no parameter names"))
        elif suite.startswith("custom_ad:") and f[5] == "text_diff":
            case, field = row.rsplit(":", 1)
            key = (suite.split(":")[1], case, field)
            if key in theirs and key in ours:
                names = name_only(theirs[key], ours[key])
                if names and all(j == "wrapped_fun" for j, _ in names):
                    out.append((suite, row, "c", f"identical except name=, where JAX prints the __name__ of inspect.unwrap(fun) ({unwrap}; {sourceinfo}): the wrapped function is a custom_gradient result, whose __wrapped__ is JAX's own Python def at {wrapped}, so the name is wrapped_fun whatever the generator called it (ojax prints g); a traced Python function's name (see prove)"))
    return out


def write(rows, path):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("suite\trow\tkind\tproof\n")
        for r in sorted(rows):
            fh.write("\t".join(r) + "\n")
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify", help="write test/golden/custom_ad_defects.tsv from a run_goldens log and an OJAX_COMPOSE_DUMP of custom_ad")
    c.add_argument("--log", required=True)
    c.add_argument("--dump", required=True)
    c.add_argument("--reference", default=os.path.join(ROOT, "goldens", "custom_ad", "text", "reference.jsonl"))
    c.add_argument("--out", default=OUT)
    p = sub.add_parser("prove", help="rerun JAX: the texts follow the Python names only")
    p.add_argument("--python", required=True)
    args = parser.parse_args()
    if args.cmd == "classify":
        rows = classify(args.log, args.reference, args.dump)
        write(rows, args.out)
        counts = {}
        for r in rows:
            counts[r[0]] = counts.get(r[0], 0) + 1
        sys.stdout.write(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    else:
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        sys.stdout.write(subprocess.run([args.python, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout)


if __name__ == "__main__":
    main()
