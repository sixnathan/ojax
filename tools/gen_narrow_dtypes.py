#!/usr/bin/env python3
"""Write test/golden/narrow_dtypes.ml from real JAX. Run from outside the repo with reference JAX.

The goldens hold no int2, int4, uint2, uint4 or small-float arrays. Each row here is one operation
over every bit pattern of such a dtype, with JAX's result bytes.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)
BITS = {
    "int2": 2,
    "int4": 4,
    "uint2": 2,
    "uint4": 4,
    "float4_e2m1fn": 4,
    "float8_e3m4": 8,
    "float8_e4m3": 8,
    "float8_e4m3b11fnuz": 8,
    "float8_e4m3fn": 8,
    "float8_e4m3fnuz": 8,
    "float8_e5m2": 8,
    "float8_e5m2fnuz": 8,
    "float8_e8m0fnu": 8,
}
GRID = np.array(
    [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, 3.0, 7.0, 8.0, 100.0, -100.0, 1e-3, -1e-3, 1e30, -1e30,
     1e-30, np.inf, -np.inf, np.nan, -np.nan],
    dtype=np.float32,
)
OPS = {
    "put": lambda x: jax.device_put(x),
    "neg": lambda x: -jnp.asarray(x),
    "add": lambda x: jnp.asarray(x) + jnp.asarray(x),
    "to_f32": lambda x: jnp.asarray(x).astype(jnp.float32),
}


def pattern(name):
    dtype = np.dtype(getattr(jnp, name))
    bits = BITS[name]
    if name.startswith("int"):
        return np.array([i - (1 << bits) if i >= 1 << (bits - 1) else i for i in range(1 << bits)], dtype=dtype)
    if name.startswith("uint"):
        return np.array(range(1 << bits), dtype=dtype)
    return np.frombuffer(np.arange(1 << bits, dtype="u1").tobytes(), dtype)


def jax_name(array):
    return "float32" if array.dtype == np.float32 else array.dtype.name


def row(name, op, source, source_name, result, result_name):
    print(f'  ("{name}", "{op}", "{source_name}", "{source.tobytes().hex()}", "{result_name}", "{result.tobytes().hex()}");')


def main():
    print("let rows =\n  [")
    for name in BITS:
        x = pattern(name)
        for op, fn in OPS.items():
            r = np.asarray(fn(x))
            row(name, op, x, name, r, "float32" if op == "to_f32" else name)
        r = np.asarray(jnp.asarray(GRID).astype(getattr(jnp, name)))
        row(name, "from_f32", GRID, "float32", r, name)
    print("  ]")


if __name__ == "__main__":
    main()
