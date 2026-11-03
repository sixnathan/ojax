import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import argparse
import json
import platform
import warnings

import numpy as np

import jax
import jax.numpy as jnp

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inputs():
    a9 = np.array([3.0, 1.0, 2.0, 3.0, 0.0, -1.0, 1.0, 5.0, 0.0], np.float32)
    n9 = np.array([3.0, np.nan, 2.0, 3.0, -0.0, np.nan, 1.0, 0.0, 2.0], np.float32)
    b6 = np.array([1.0, 7.0, 2.0, 9.0, 0.0, 3.0], np.float32)
    u5 = np.array([4.0, -2.0, 7.0, 0.0, 1.0], np.float32)
    u4 = np.array([7.0, 3.0, -2.0, 8.0], np.float32)
    m23 = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], np.float32)
    mask23 = np.array([[True, False, True], [False, True, True]])
    m33 = np.zeros((3, 3), np.float32)
    return {"a9": a9, "n9": n9, "b6": b6, "u5": u5, "u4": u4, "m23": m23, "mask23": mask23, "m33": m33,
            "s0": np.array(2.0, np.float32), "z0": np.array(0.0, np.float32),
            "v2": np.array([10.0, 20.0], np.float32), "v7": np.arange(7, dtype=np.float32) + 1.0,
            "i9": np.array([3, 1, 2, 3, 0, -1, 1, 5, 0], np.int32), "i6": np.array([1, 7, 2, 9, 0, 3], np.int32)}


CASES = [
    ("argwhere_size", lambda x: jnp.argwhere(x, size=4), ["m23"]),
    ("argwhere_size_fill", lambda x: jnp.argwhere(x, size=5, fill_value=-1), ["m23"]),
    ("argwhere_scalar", lambda x: jnp.argwhere(x, size=2), ["s0"]),
    ("argwhere_scalar_zero", lambda x: jnp.argwhere(x, size=2), ["z0"]),
    ("unique_size", lambda x: jnp.unique(x, size=4), ["a9"]),
    ("unique_size_nan", lambda x: jnp.unique(x, size=6), ["n9"]),
    ("unique_size_fill", lambda x: jnp.unique(x, size=8, fill_value=-9.0), ["a9"]),
    ("unique_size_nan_noequal", lambda x: jnp.unique(x, size=7, equal_nan=False), ["n9"]),
    ("intersect1d_size", lambda x, y: jnp.intersect1d(x, y, size=3), ["a9", "b6"]),
    ("intersect1d_size_fill", lambda x, y: jnp.intersect1d(x, y, size=6, fill_value=-1.0), ["a9", "b6"]),
    ("intersect1d_size_unique", lambda x, y: jnp.intersect1d(x, y, assume_unique=True, size=3), ["u5", "u4"]),
    ("intersect1d_size_nan", lambda x, y: jnp.intersect1d(x, y, size=4), ["n9", "b6"]),
    ("intersect1d_size_int", lambda x, y: jnp.intersect1d(x, y, size=5, fill_value=-1), ["i9", "i6"]),
    ("setdiff1d_size", lambda x, y: jnp.setdiff1d(x, y, size=3), ["a9", "b6"]),
    ("setdiff1d_size_fill", lambda x, y: jnp.setdiff1d(x, y, size=6, fill_value=-1.0), ["a9", "b6"]),
    ("setdiff1d_size_unique", lambda x, y: jnp.setdiff1d(x, y, assume_unique=True, size=4, fill_value=9.0), ["u5", "u4"]),
    ("setdiff1d_size_int", lambda x, y: jnp.setdiff1d(x, y, size=5), ["i9", "i6"]),
    ("setxor1d_size", lambda x, y: jnp.setxor1d(x, y, size=4), ["a9", "b6"]),
    ("setxor1d_size_fill", lambda x, y: jnp.setxor1d(x, y, size=9, fill_value=-1.0), ["a9", "b6"]),
    ("setxor1d_size_unique", lambda x, y: jnp.setxor1d(x, y, assume_unique=True, size=6), ["u5", "u4"]),
    ("setxor1d_size_int", lambda x, y: jnp.setxor1d(x, y, size=7, fill_value=-1), ["i9", "i6"]),
    ("isin", lambda x, y: jnp.isin(x, y), ["m23", "b6"]),
    ("isin_invert", lambda x, y: jnp.isin(x, y, invert=True), ["a9", "b6"]),
    ("place_tile", lambda a, m, v: jnp.place(a, m, v, inplace=False), ["m23", "mask23", "v2"]),
    ("place_slice", lambda a, m, v: jnp.place(a, m, v, inplace=False), ["m23", "mask23", "v7"]),
    ("fill_diagonal_tile", lambda a, v: jnp.fill_diagonal(a, v, inplace=False), ["m33", "v2"]),
    ("fill_diagonal_slice", lambda a, v: jnp.fill_diagonal(a, v, inplace=False), ["m33", "v7"]),
    ("insert_scalar", lambda a, v: jnp.insert(a, 2, v), ["a9", "s0"]),
    ("insert_vector", lambda a, v: jnp.insert(a, -3, v), ["a9", "v2"]),
    ("insert_axis1", lambda a, v: jnp.insert(a, 1, v, axis=1), ["m23", "v2"]),
    ("insert_flat", lambda a, v: jnp.insert(a, 4, v), ["m23", "v7"]),
]


def out_row(v):
    h = np.asarray(v)
    return {"dtype": str(h.dtype), "shape": list(h.shape), "hex": h.tobytes().hex()}


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    jax.clear_caches()
    table = inputs()
    rows = []
    for name, fn, arg_names in CASES:
        args = [table[k] for k in arg_names]

        def f(*a):
            return fn(*a)

        row = {"id": name, "x64": x64, "args": [{"name": k, "dtype": str(table[k].dtype), "shape": list(table[k].shape), "hex": table[k].tobytes().hex()} for k in arg_names]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mode, call in (("eager", lambda: f(*[jnp.asarray(a) for a in args])), ("jit", lambda: jax.jit(f)(*args))):
                try:
                    row[mode] = out_row(call())
                except Exception as e:
                    row[mode + "_error"] = type(e).__name__ + ": " + str(e)
            try:
                row["jaxpr"] = str(jax.make_jaxpr(f)(*args))
                row["hlo"] = jax.jit(f).lower(*args).as_text()
            except Exception as e:
                row["text_error"] = type(e).__name__ + ": " + str(e)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "size_ops"))
    args = parser.parse_args()
    if platform.python_version() != PY_VERSION or jax.__version__ != JAX_VERSION:
        raise SystemExit("needs python " + PY_VERSION + " and jax " + JAX_VERSION)
    rows = run(False) + run(True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cases.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
