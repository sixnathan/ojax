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

import functools
import hashlib
import json
import platform
import shutil
import sys
import zlib

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import dtypes as jax_dtypes

import emit_common as ec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"


def preflight():
    if platform.python_version() != PY_VERSION:
        raise SystemExit("python " + platform.python_version() + " != " + PY_VERSION)
    if jax.__version__ != JAX_VERSION:
        raise SystemExit("jax " + jax.__version__ + " != " + JAX_VERSION)


def load_tolerances():
    with open(os.path.join(ROOT, "tools", "tolerances.json"), encoding="utf-8") as fh:
        return json.load(fh)


TOLERANCES = load_tolerances()


def resolve_tol(dtype_name):
    if ec.is_exact_dtype(dtype_name):
        return "exact", {"atol": 0, "rtol": 0}
    tol = TOLERANCES["default"][dtype_name]
    return "allclose", {"atol": tol, "rtol": tol}


def np_dtype(dtype_name):
    return np.dtype(dtype_name)


def _rand_dtype(rand, shape, dtype, scale=1.0, post=None):
    shape = tuple(shape)
    vals = np.asarray(scale * rand(*shape)).astype(dtype)
    if post is not None:
        vals = np.asarray(post(vals), dtype).astype(dtype)
    return vals.reshape(shape)


def rand_default(rng, shape, dtype):
    return _rand_dtype(rng.randn, shape, dtype, scale=3.0)


def rand_small(rng, shape, dtype):
    return _rand_dtype(rng.randn, shape, dtype, scale=1e-3)


def rand_positive(rng, shape, dtype):
    return _rand_dtype(rng.rand, shape, dtype, scale=2.0, post=lambda x: x + 1)


def rand_nonzero(rng, shape, dtype):
    post = lambda x: np.where(x == 0, np.array(1, dtype=x.dtype), x)
    return _rand_dtype(rng.randn, shape, dtype, scale=3.0, post=post)


def rand_int(rng, shape, dtype):
    high = np.iinfo(dtype).max
    return rng.randint(0, high, size=tuple(shape), dtype=dtype)


def rand_bool(rng, shape, dtype):
    return np.asarray(rng.rand(*tuple(shape)) < 0.5, dtype=dtype)


RNG_FACTORIES = {
    "rand_default": rand_default,
    "rand_small": rand_small,
    "rand_positive": rand_positive,
    "rand_nonzero": rand_nonzero,
    "rand_int": rand_int,
    "rand_bool": rand_bool,
}


def draw(rng_name, seed, shape, dtype_name):
    rng = np.random.RandomState(seed)
    return RNG_FACTORIES[rng_name](rng, shape, np_dtype(dtype_name))


def weak_scalar(dtype_name, seed):
    rng = np.random.RandomState(seed)
    if dtype_name.startswith("complex"):
        return complex(rng.randn(), rng.randn())
    if dtype_name.startswith("float") or dtype_name.startswith("bfloat"):
        return float(rng.randn())
    if dtype_name == "bool":
        return bool(rng.rand() < 0.5)
    return int(rng.randint(0, 10))


def lax_add(params):
    return lambda *xs: jax.lax.add(*xs)


def lax_reduce_sum(params):
    axes = tuple(params["axes"])
    return lambda x: jax.lax.reduce_sum(x, axes=axes)


LAX_BUILDERS = {
    "add": lax_add,
    "reduce_sum": lax_reduce_sum,
}


def _add_all(*xs):
    return functools.reduce(jnp.add, xs)


def run_case(c, seed):
    op = c["op"]
    if op in LAX_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(LAX_BUILDERS[op](c["params"]))(*inputs)
        return inputs, [np.asarray(out)], [None]
    if op == "result_type":
        operands = []
        stored = []
        for i, a in enumerate(c["args"]):
            if a["weak"]:
                v = weak_scalar(a["dtype"], seed + i)
                operands.append(v)
                canon = jax_dtypes.canonicalize_dtype(np.asarray(v).dtype)
                stored.append(np.asarray(v).astype(canon))
            else:
                arr = draw(a["rng"], seed + i, a["shape"], a["dtype"])
                operands.append(arr)
                stored.append(arr)
        aval = jax.eval_shape(_add_all, *operands)
        out = jax.jit(_add_all)(*operands)
        return stored, [np.asarray(out)], [bool(aval.weak_type)]
    raise SystemExit("unknown op " + op)


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


def gen_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(
            (not a["weak"]) and ec.is_wide64(a["dtype"]) for a in c["args"]
        ):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        stored, outputs, out_weak = run_case(c, seed)
        in_arrays = {a["name"]: np.asarray(v) for a, v in zip(c["args"], stored)}
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        out0 = np.asarray(outputs[0])
        compare, tol = resolve_tol(out0.dtype.name)
        args_meta = [
            {
                "name": a["name"],
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
                "weak": a["weak"],
            }
            for a, v in zip(c["args"], stored)
        ]
        outs_meta = []
        for i, o in enumerate(outputs):
            oa = np.asarray(o)
            entry = {
                "name": "out" + str(i),
                "shape": [int(d) for d in oa.shape],
                "dtype": oa.dtype.name,
            }
            if out_weak[i] is not None:
                entry["weak"] = out_weak[i]
            outs_meta.append(entry)
        manifest_cases.append(
            {
                "case_id": case_id,
                "op": c["op"],
                "primitive": c["primitive"],
                "nargs": c["nargs"],
                "args": args_meta,
                "params": c["params"],
                "outputs": outs_meta,
                "compare": compare,
                "tol": tol,
                "grads": None,
            }
        )
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def load_spec(module):
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    cases = list(doc["cases"])
    cases.sort(key=ec.sort_key)
    return cases


def generate(module):
    preflight()
    cases = load_spec(module)
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_goldens.py <module>")
    n_off, n_on = generate(sys.argv[1])
    sys.stdout.write(sys.argv[1] + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
