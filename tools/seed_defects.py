import argparse
import os
import subprocess
import sys

SEED_DEFECTS = os.path.join("test", "golden", "seed_defects.tsv")
KINDS = ("a", "b", "c", "d")

HISTORY_PROBE = r"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if sys.argv[1] == "after_slice":
    jax.lax.slice(jnp.zeros((8, 2)), (2, 0), (5, 2))
try:
    jax.lax.dynamic_slice_in_dim(jnp.zeros((int(sys.argv[2]), 2)), 2, 3, 0)
except Exception as error:
    print(type(error).__name__ + ": " + str(error))
"""

PROOFS = {
    ("40618829", "lax_dynamic_slice_in_dim_6x2_902a1355_float64_shape_small"): (HISTORY_PROBE, ("alone", "after_slice"), ("0",)),
    ("52790361", "lax_dynamic_slice_in_dim_6x2_902a1355_float64_shape_small"): (HISTORY_PROBE, ("alone", "after_slice"), ("0",)),
    ("52790361", "lax_dynamic_slice_in_dim_6x2_902a1355_float64_shape_other"): (HISTORY_PROBE, ("alone", "after_slice"), ("1",)),
}


def read_seed_defects(root, seed=None):
    rows = {}
    path = os.path.join(root, SEED_DEFECTS)
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 6 or f[4] not in KINDS or not f[5]:
                sys.exit(f"seed_defects: bad line in {path}: {line.rstrip()}")
            if seed is not None and f[0] != str(seed):
                continue
            name, mode = f[1].split("/")
            rows[("text_" + name + ":" + mode, f[2] + ":" + f[3])] = f[4]
    return rows


def prove(python):
    failures = 0
    for (seed, case), (probe, modes, probe_args) in PROOFS.items():
        texts = {}
        for mode in modes:
            out = subprocess.run([python, "-I", "-c", probe, mode, *probe_args], capture_output=True, text=True, cwd="/", check=False)
            texts[mode] = out.stdout.strip()
            print(seed + "\t" + case + "\t" + mode + "\t" + texts[mode])
        if len(set(texts.values())) < 2:
            failures += 1
            print(seed + "\t" + case + "\tnot reproduced")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prove", "list"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()
    if args.command == "list":
        for (suite, row), kind in sorted(read_seed_defects(args.root, args.seed).items()):
            print(suite + "\t" + row + "\t" + kind)
        return
    sys.exit(1 if prove(os.path.abspath(args.python)) else 0)


if __name__ == "__main__":
    main()
