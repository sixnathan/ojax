import argparse
import os
import re
import subprocess
import sys

ORDER = re.compile(r"(complexfloating, floating|floating, complexfloating)")
ADDRESS = re.compile(r"<function ufunc\.at at 0x[0-9a-f]+>")

PROBE = r'''
import jax.numpy as jnp
x = jnp.zeros((4,), bool)
i = jnp.array([0, 1])
for op in ("add", "multiply", "subtract"):
    try:
        getattr(x.at[i], op)(True)
    except TypeError as e:
        print(op, str(e))
try:
    x.at[i].apply(jnp.negative)
except TypeError as e:
    print("apply", str(e))
try:
    jnp.add.at(jnp.zeros(4), (slice(None, None, 2),), 1.0, inplace=False)
except TypeError as e:
    print("ufunc_at", str(e).split(" as an abstract")[0])
'''


def unescape(text):
    return text.encode("utf-8").decode("unicode_escape")


def classify(jax_text, ojax_text):
    if ORDER.sub("ORDER", jax_text) == ORDER.sub("ORDER", ojax_text) and ORDER.search(jax_text):
        return "a", "Accepted dtypes order differs between JAX processes, under PYTHONHASHSEED 0..5 and between two runs of the generator with one seed (see prove)"
    if ADDRESS.search(jax_text) and ADDRESS.sub("<function ufunc.at>", jax_text) == ojax_text:
        return "c", "JAX's text holds the address of the Python function object ufunc.at, different in every process (see prove)"
    return "", "unexplained"


def run_classify(log, out):
    rows = []
    with open(log, errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or not parts[4].startswith("text_index_fuzz:") or parts[2] == "0":
                continue
            kind, reason = classify(unescape(parts[6]), unescape(parts[7]))
            rows.append((parts[4], parts[0], kind or "none", reason))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("suite\trow\tdefect_class\tproof\n")
        for row in sorted(rows):
            fh.write("\t".join(row) + "\n")
    counts = {}
    for _, _, kind, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    sys.stdout.write(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")


def run_prove(python):
    seen = {}
    for seed in range(6):
        env = dict(os.environ, PYTHONHASHSEED=str(seed), JAX_PLATFORMS="cpu")
        text = subprocess.run([python, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout
        for line in text.splitlines():
            op, _, message = line.partition(" ")
            seen.setdefault(op, set()).add(message)
    for op, messages in sorted(seen.items()):
        sys.stdout.write(f"{op}: {len(messages)} distinct texts over PYTHONHASHSEED 0..5\n")
        for message in sorted(messages):
            sys.stdout.write("  " + message + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--log", required=True)
    c.add_argument("--out", required=True)
    p = sub.add_parser("prove")
    p.add_argument("--python", required=True)
    args = parser.parse_args()
    if args.cmd == "classify":
        run_classify(args.log, args.out)
    else:
        run_prove(args.python)


if __name__ == "__main__":
    main()
