import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_NUM_GENERATED_CASES"] = "10"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import hashlib
import json
import platform
import shutil
import zlib

import numpy as np

import jax
from jax import lax

import emit_common as ec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"


def load_tolerances():
    with open(os.path.join(ROOT, "tools", "tolerances.json"), encoding="utf-8") as fh:
        return json.load(fh)


TOLERANCES = load_tolerances()


def resolve_tol(dtype_name):
    if ec.is_exact_dtype(dtype_name):
        return "exact", {"atol": 0, "rtol": 0}
    tol = TOLERANCES["default"][dtype_name]
    return "allclose", {"atol": tol, "rtol": tol}


def preflight():
    if platform.python_version() != PY_VERSION:
        raise SystemExit("python " + platform.python_version() + " != " + PY_VERSION)
    if jax.__version__ != JAX_VERSION:
        raise SystemExit("jax " + jax.__version__ + " != " + JAX_VERSION)


def draw(kind, seed, shape, dtype):
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    if kind == "default":
        return rng.uniform(-1.0, 1.0, shape).astype(dtype)
    if kind == "positive":
        return rng.uniform(0.25, 1.75, shape).astype(dtype)
    if kind == "small":
        return rng.uniform(-4.0, 4.0, shape).astype(dtype)
    if kind == "int":
        return rng.randint(-5, 6, shape).astype(dtype)
    raise SystemExit("unknown rng kind " + kind)


def dot(a, b):
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())))


CASES = [
    ("cubic_f32_4", lambda x: lax.sub(lax.mul(x, lax.mul(x, x)), x),
     [((4,), "float32", "default")]),
    ("i32_poly_4", lambda x: lax.sub(lax.mul(x, x), x),
     [((4,), "int32", "int")]),
    ("i32_two_4", lambda a, b: lax.sub(lax.mul(a, b), a),
     [((4,), "int32", "int"), ((4,), "int32", "int")]),
    ("select_min_f32_4", lambda x, y: lax.select_n(lax.lt(x, y), y, x),
     [((4,), "float32", "default"), ((4,), "float32", "default")]),
    ("abs_add_sign_f32_4", lambda x: lax.add(lax.abs(x), lax.sign(x)),
     [((4,), "float32", "default")]),
    ("broadcast_mul_f32_2x3",
     lambda x, y: lax.mul(lax.broadcast_in_dim(x, (2, 3), (1,)), y),
     [((3,), "float32", "default"), ((2, 3), "float32", "default")]),
    ("convert_trunc_add_i32_4",
     lambda x: (lambda t: lax.add(t, t))(lax.convert_element_type(x, np.int32)),
     [((4,), "float32", "small")]),
    ("reshape_reduce_f32_2x3",
     lambda x: lax.reduce_sum(lax.reshape(x, (6,)), (0,)),
     [((2, 3), "float32", "default")]),
    ("minmax_chain_f32_4",
     lambda x, y: lax.sub(lax.max(x, y), lax.min(x, y)),
     [((4,), "float32", "default"), ((4,), "float32", "default")]),
    ("exp_neg_f32_4", lambda x: lax.exp(lax.neg(x)),
     [((4,), "float32", "default")]),
    ("sin_mul_f32_4", lambda x, y: lax.mul(lax.sin(x), y),
     [((4,), "float32", "default"), ((4,), "float32", "default")]),
    ("tanh_exp_sin_f32_4", lambda x: lax.tanh(lax.exp(lax.sin(x))),
     [((4,), "float32", "default")]),
    ("sum_sin_f32_5", lambda x: lax.reduce_sum(lax.sin(x), (0,)),
     [((5,), "float32", "default")]),
    ("cos_log_f32_4", lambda x: lax.cos(lax.log(x)),
     [((4,), "float32", "positive")]),
    ("pow_f32_4", lambda x, y: lax.pow(x, y),
     [((4,), "float32", "positive"), ((4,), "float32", "default")]),
    ("matmul_f32_2x3x4", dot,
     [((2, 3), "float32", "default"), ((3, 4), "float32", "default")]),
    ("matmul_add_f32_2x4",
     lambda a, b, c: lax.add(dot(a, b), lax.broadcast_in_dim(c, (2, 4), (1,))),
     [((2, 3), "float32", "default"), ((3, 4), "float32", "default"),
      ((4,), "float32", "default")]),
    ("reduce_matmul_f32", lambda a, b: lax.reduce_sum(dot(a, b), (0, 1)),
     [((2, 3), "float32", "default"), ((3, 4), "float32", "default")]),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(outdir):
    entries = []
    for base, dirs, files in os.walk(outdir):
        dirs.sort()
        for f in sorted(files):
            if f == "SHA256SUMS":
                continue
            path = os.path.join(base, f)
            rel = os.path.relpath(path, outdir)
            entries.append((rel, sha256_file(path)))
    entries.sort()
    lines = "".join(digest + "  " + rel + "\n" for rel, digest in entries)
    with open(os.path.join(outdir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write(lines)


def gen(outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for case_id, fn, specs in CASES:
        seed = zlib.adler32(case_id.encode("utf-8"))
        inputs = [
            draw(kind, seed + i, shape, np.dtype(dt))
            for i, (shape, dt, kind) in enumerate(specs)
        ]
        out = np.asarray(jax.jit(fn)(*inputs))
        in_arrays = {"in" + str(i): v for i, v in enumerate(inputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), out0=out)
        args_meta = [
            {
                "name": "in" + str(i),
                "shape": [int(d) for d in v.shape],
                "dtype": v.dtype.name,
            }
            for i, v in enumerate(inputs)
        ]
        compare, tol = resolve_tol(out.dtype.name)
        manifest_cases.append(
            {
                "case_id": case_id,
                "compare": compare,
                "tol": tol,
                "args": args_meta,
                "output": {
                    "name": "out0",
                    "shape": [int(d) for d in out.shape],
                    "dtype": out.dtype.name,
                },
            }
        )
    manifest_cases.sort(key=lambda c: c["case_id"])
    manifest = {
        "schema_version": 1,
        "module": "e2e",
        "jax_version": JAX_VERSION,
        "x64": False,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def main():
    preflight()
    outdir = os.path.join(ROOT, "goldens", "e2e")
    n = gen(outdir)
    print("e2e goldens:", n, "cases ->", outdir)


if __name__ == "__main__":
    main()
