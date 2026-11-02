#!/usr/bin/env python3
"""Write test/golden/special_values.tsv from real JAX. Run from outside the repo with reference JAX.

The goldens hold no NaN, infinity or negative zero in any input. Each row here runs one sw_ operation
of tools/parity_registry.py over those values, eager and under jit, and keeps JAX's result bytes.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np
import parity_registry as registry

SKIPPED = {"sw_matmul", "sw_trace"}
WORDS = {
    "float32": ("<u4", np.float32, (0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FA12345, 0x80000000, 0, 0x7F800000, 0xFF800000, 0x3FC00000),
                (0x80000000, 0, 0x80000000, 0, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0xFFC00000)),
    "float64": ("<u8", np.float64, (0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001, 0x7FF8000000000001, 0x8000000000000000, 0,
                                    0x7FF0000000000000, 0xFFF0000000000000, 0x3FF8000000000000),
                (0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000, 0x3FF0000000000000, 0xBFF0000000000000,
                 0x7FF8000000000000, 0xFFF8000000000000)),
    "float16": ("<u2", np.float16, (0x7E00, 0xFE00, 0x7C01, 0x7E01, 0x8000, 0, 0x7C00, 0xFC00, 0x3E00),
                (0x8000, 0, 0x8000, 0, 0x8000, 0x3C00, 0xBC00, 0x7E00, 0xFE00)),
}
SETS = {False: ["float32", "float16", "complex64"], True: ["float64", "complex128"]}


def inputs(dtype):
    if dtype in WORDS:
        unsigned, np_dtype, first, second = WORDS[dtype]
        return [np.frombuffer(np.array(w, unsigned).tobytes(), np_dtype).reshape(3, 3) for w in (first, second)]
    real = "float32" if dtype == "complex64" else "float64"
    one, two = inputs(real)
    return [
        (np.frombuffer(np.stack([one, two], -1).tobytes(), dtype).reshape(3, 3)),
        (np.frombuffer(np.stack([two, one], -1).tobytes(), dtype).reshape(3, 3)),
    ]


def run(op, a, jitted):
    fn = registry.OPS[op]
    result = jax.jit(lambda x: fn(a=x, p={}))(a) if jitted else fn(a=a, p={})
    outs = list(result) if isinstance(result, (tuple, list)) else [result]
    outs = [np.asarray(o, order="C") for o in outs]
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x64", choices=["x64_off", "x64_on"])
    args = parser.parse_args()
    x64 = args.x64 == "x64_on"
    jax.config.update("jax_enable_x64", x64)
    ops = sorted(op for op in registry.OPS if op.startswith("sw_") and op not in SKIPPED)
    for dtype in SETS[x64]:
        for index, a in enumerate(inputs(dtype)):
            for op in ops:
                if op.startswith("sw_") and op[3:].startswith(("st_", "nn_")) and False:
                    continue
                for mode in ("eager", "jit"):
                    try:
                        outs = run(op, a, mode == "jit")
                    except Exception:
                        continue
                    fields = [args.x64, mode, op, dtype, str(index), a.tobytes().hex(), str(len(outs))]
                    for o in outs:
                        fields += [o.dtype.name if o.dtype.name != "void16" else "bfloat16", "x".join(map(str, o.shape)) or "-", o.tobytes().hex()]
                    print("\t".join(fields))


if __name__ == "__main__":
    main()
