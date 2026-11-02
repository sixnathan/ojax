#!/usr/bin/env python3
"""Write test/golden/poison_values.tsv from real JAX. Run from outside the repo with reference JAX.

One case per operation of tools/parity_registry.py, with NaN, negative NaN, infinity and negative zero
written into its floating-point inputs, eager and under jit, keeping JAX's result bytes.
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_parity_goldens as gp
import emit_common as ec
import jax
import numpy as np
import parity_registry as registry

LIMIT = 4096
SKIP_PREFIX = ("sw_",)
SKIP_OPS = {"swg_matmul", "swv_matmul", "swj_matmul", "swg_trace", "swv_trace", "swj_trace"}


def poison(array):
    if array.dtype.kind not in "fc" or array.size == 0:
        return array
    out = array.copy()
    flat = out.reshape(-1)
    values = [np.nan, -np.nan, np.inf, -np.inf, -0.0]
    for k, value in enumerate(values):
        if k * 2 < flat.size:
            flat[k * 2] = value
    return out


def outputs(c, arrays, jitted):
    names = sorted(arrays)
    fn = registry.OPS[c["op"]]
    if jitted:
        result = jax.jit(lambda *v: fn(**dict(zip(names, v)), p=c["params"]))(*[arrays[n] for n in names])
    else:
        result = fn(**arrays, p=c["params"])
    outs = list(result) if isinstance(result, (tuple, list)) else [result]
    outs = [np.asarray(o, order="C") for o in outs]
    return outs


def shape_text(array):
    return "x".join(map(str, array.shape)) or "-"


def dtype_text(array):
    return "bfloat16" if array.dtype.kind == "V" else array.dtype.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x64", choices=["x64_off", "x64_on"])
    args = parser.parse_args()
    x64 = args.x64 == "x64_on"
    jax.config.update("jax_enable_x64", x64)
    wanted = "float64" if x64 else "float32"
    seen = set()
    for c in gp.expand():
        op = c["op"]
        if c["dtype"] != wanted or op in seen or op.startswith(SKIP_PREFIX) or op in SKIP_OPS:
            continue
        if x64 and op in registry.SINGLE_PRECISION_ONLY:
            continue
        rng = np.random.default_rng(zlib.adler32(c["case_id"].encode("utf-8")))
        drawn = {k: gp.draw(rng, c["inputs"][k], c["dtype"]) for k in sorted(c["inputs"])}
        arrays = {k: poison(v) for k, v in drawn.items()}
        if not any(v.dtype.kind in "fc" for v in arrays.values()):
            continue
        written = False
        for mode in ("eager", "jit"):
            try:
                outs = outputs(c, arrays, mode == "jit")
            except Exception:
                continue
            if sum(o.nbytes for o in outs) + sum(a.nbytes for a in arrays.values()) > LIMIT:
                continue
            if any(o.dtype.kind == "V" and o.dtype.itemsize != 2 for o in outs):
                continue
            written = True
            fields = [args.x64, mode, op, c["case_id"], json.dumps(c["params"], sort_keys=True), str(len(arrays))]
            for name, a in arrays.items():
                fields += [name, dtype_text(a), shape_text(a), a.tobytes().hex()]
            for o in outs:
                fields += [dtype_text(o), shape_text(o), o.tobytes().hex()]
            print("\t".join(fields))
        if written:
            seen.add(op)


if __name__ == "__main__":
    main()
