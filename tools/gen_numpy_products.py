#!/usr/bin/env python3
"""Write test/golden/numpy_products.ml. Run from outside the repo with reference JAX.

tools/parity_registry.py computes a @ a.T on a raw NumPy input for sw_matmul and sw_trace, so
that product is NumPy's, not JAX's. Each row here holds the generator's product and real JAX's
product of the same input, and checks the golden is NumPy's product byte for byte.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = [
    ("sw_matmul", "x64_off", "6f500ed1", "float32"),
    ("sw_matmul", "x64_on", "6f500ed1", "float32"),
    ("sw_matmul", "x64_on", "6f500ed1", "float64"),
    ("sw_trace", "x64_on", None, "float64"),
]
TRACE_FIRST_WORD = 0xBFEA42F4B8893D90


def trace_case(goldens):
    inputs = os.path.join(goldens, "parity/x64_on/inputs")
    for name in sorted(os.listdir(inputs)):
        if name.startswith("sw_trace_") and name.endswith("_float64.npz"):
            a = np.load(os.path.join(inputs, name))["a"]
            if a.dtype == np.complex128 and np.frombuffer(a.tobytes(), "<u8")[0] == TRACE_FIRST_WORD:
                return name[: -len(".npz")]
    sys.exit("no sw_trace row with the recorded input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x64", choices=["x64_off", "x64_on"])
    parser.add_argument("--goldens", default=os.path.join(ROOT, "goldens"))
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", args.x64 == "x64_on")
    for op, x64, seed, dtype in ROWS:
        if x64 != args.x64:
            continue
        case = trace_case(args.goldens) if seed is None else f"{op}_3x3_{seed}_{dtype}"
        directory = os.path.join(args.goldens, "parity", x64)
        a = np.load(os.path.join(directory, "inputs", case + ".npz"))["a"]
        golden = np.load(os.path.join(directory, "outputs", case + ".npz"))["out0"]
        numpy_product = a @ a.T
        expected = numpy_product if op == "sw_matmul" else np.trace(numpy_product)
        if op == "sw_matmul" and golden.tobytes() != expected.tobytes():
            sys.exit(f"{case}: the golden is not NumPy's product")
        wrapped = jnp.asarray(a)
        jax_product = np.asarray(wrapped @ wrapped.T)
        jitted = np.asarray(jax.jit(lambda x: x @ x.T)(wrapped))
        if jitted.tobytes() != jax_product.tobytes():
            sys.exit(f"{case}: JAX's eager and jit products differ")
        print(
            f'  ("{case}", "{x64}", "{a.dtype.name}", "{a.tobytes().hex()}",\n'
            f'   "{numpy_product.tobytes().hex()}",\n'
            f'   "{jax_product.tobytes().hex()}");'
        )


if __name__ == "__main__":
    main()
