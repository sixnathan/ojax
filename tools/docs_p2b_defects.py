#!/usr/bin/env python3
"""Prove which docs_p2b rows carry a JAX text that changes with PYTHONHASHSEED and write them as kind a defect rows."""
import argparse
import json
import os
import subprocess
import tempfile
import sys

PROBE = r"""
import json, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", sys.argv[1] == "1")
import jax.numpy as jnp
out = {}
for row in json.loads(sys.argv[2]):
    if row["group"] == "one_hot":
        try:
            jax.nn.one_hot(jnp.asarray(np.array([0, 2, 1, 1], dtype=np.int32)).astype(row["dtype"]), 3)
            out[row["id"]] = ""
        except Exception as e:
            out[row["id"]] = type(e).__name__ + ": " + str(e)
        continue
    x = jnp.asarray(np.array([1, 2, 3])).astype(row["d1"])
    y = jnp.asarray(np.array([3, 0, 1])).astype(row["d2"])
    try:
        with jax.numpy_dtype_promotion("strict"):
            if row["case"] == "add":
                jnp.add(x, y)
            elif row["case"] == "multiply":
                jax.jit(jnp.multiply)(x, y)
            elif row["case"] == "where":
                jnp.where(x > y, x, y)
            else:
                jnp.concatenate([x, y])
        out[row["id"]] = ""
    except Exception as e:
        out[row["id"]] = type(e).__name__ + ": " + str(e)
print(json.dumps(out))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    args = ap.parse_args()
    seeds = args.seeds.split(",")
    lines = ["suite\trow\tkind\tproof"]
    for tag, flag in (("x64_off", "0"), ("x64_on", "1")):
        with open(os.path.join(args.goldens, "docs_p2b", tag + ".json")) as fh:
            rows = [r for r in json.load(fh)["rows"] if r["group"] == "promo" or (r["group"] == "one_hot" and r["kind"] == "raise")]
        spec = json.dumps([{k: r[k] for k in ("id", "group", "case", "d1", "d2", "dtype") if k in r} for r in rows])
        texts = {}
        for seed in seeds:
            env = dict(os.environ, PYTHONHASHSEED=seed)
            res = subprocess.run([args.python, "-c", PROBE, flag, spec], env=env, capture_output=True, text=True, check=True, cwd=tempfile.gettempdir())
            for k, v in json.loads(res.stdout.strip().splitlines()[-1]).items():
                texts.setdefault(k, {}).setdefault(v, []).append(seed)
        for r in rows:
            variants = texts[r["id"]]
            if len(variants) < 2:
                continue
            if r["group"] == "one_hot":
                heads = "; ".join(f"PYTHONHASHSEED {','.join(s)}: {v}" for v, s in sorted(variants.items()))
                proof = ("JAX lists the accepted dtypes of iota's check from a Python set, whose order follows PYTHONHASHSEED; rerun of this row: " + heads)
            else:
                heads = "; ".join(f"PYTHONHASHSEED {','.join(s)}: {v.split(' have ')[0]}" for v, s in sorted(variants.items()))
                proof = ("JAX names the dtypes of a failed promotion from a Python set (jax/_src/dtypes.py:845 N = set(nodes)), whose order follows the dtype hashes and so PYTHONHASHSEED; rerun of this row: " + heads)
            lines.append(f"docs_p2b:{tag}\t{r['id']}\ta\t{proof}")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
