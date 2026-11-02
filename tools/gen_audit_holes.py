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

BWD = {
    "ok": lambda r, g: (g * r[1], g * r[0]),
    "shape": lambda r, g: (jnp.ones(2, jnp.float32), r[0]),
    "scalar": lambda r, g: (jnp.float32(1), r[0]),
    "f64": lambda r, g: (jnp.ones(3, jnp.float64), r[0]),
    "i32": lambda r, g: (jnp.ones(3, jnp.int32), r[0]),
    "c64": lambda r, g: (jnp.ones(3, jnp.complex64), r[0]),
    "both": lambda r, g: (jnp.ones(2, jnp.int32), r[0]),
    "second": lambda r, g: (g, jnp.ones((3, 1), jnp.float32)),
    "count": lambda r, g: (g,),
}


def vjp_case(kind, symbolic):
    bwd = BWD[kind]
    f = jax.custom_vjp(lambda x, y: x * y)
    if symbolic:
        def fwd(x, y):
            return x.value * y.value, (x.value, y.value)
    else:
        def fwd(x, y):
            return x * y, (x, y)
    f.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    x = jnp.arange(3, dtype=jnp.float32)
    y = jnp.linspace(1, 2, 3, dtype=jnp.float32)
    return lambda: jax.grad(lambda x, y: f(x, y).sum(), argnums=(0, 1))(x, y)


def defjvps_case(which):
    f = jax.custom_jvp(lambda x, y: jnp.sin(x) * y)
    f.defjvps(lambda g, ans, x, y: g * jnp.cos(x) * y, lambda g, ans, x, y: g * jnp.sin(x))
    x = jnp.linspace(0.1, 1.3, 4, dtype=jnp.float32)
    y = jnp.linspace(2.0, -1.0, 4, dtype=jnp.float32)
    tx = jnp.linspace(0.5, 1.5, 4, dtype=jnp.float32)
    ty = jnp.linspace(-1.0, 3.0, 4, dtype=jnp.float32)
    if which == "jvp":
        return lambda: jax.jvp(f, (x, y), (tx, ty))
    if which == "grad":
        return lambda: jax.grad(lambda x, y: f(x, y).sum(), argnums=(0, 1))(x, y)
    return lambda: jax.jit(lambda x, y, tx, ty: jax.jvp(f, (x, y), (tx, ty)))(x, y, tx, ty)


def value_cases():
    out = {}
    for sym in (False, True):
        for kind in BWD:
            out["vjp_%s_%s" % ("sym" if sym else "plain", kind)] = vjp_case(kind, sym)
    for which in ("jvp", "grad", "jit_jvp"):
        out["defjvps_" + which] = defjvps_case(which)
    return out


def put_fn(mode):
    if mode is None:
        return lambda a, i, v: jnp.put(a, i, v, inplace=False)
    return lambda a, i, v: jnp.put(a, i, v, mode=mode, inplace=False)


def traced_cases():
    a = jnp.zeros((2, 3), jnp.float32)
    i = jnp.zeros((4,), jnp.int32)
    v = jnp.zeros((4,), jnp.float32)
    a1 = jnp.zeros((5,), jnp.int32)
    v2 = jnp.zeros((2,), jnp.float32)
    out = {}
    for mode in (None, "clip", "wrap"):
        tag = mode or "none"
        out["put_%s" % tag] = (put_fn(mode), [a, i, v])
        out["put_%s_1d_int" % tag] = (put_fn(mode), [a1, i, a1[:4]])
        out["put_%s_tiled" % tag] = (put_fn(mode), [a, i, v2])
    b9 = jnp.zeros((9,), jnp.bool_)
    f9 = jnp.zeros((9,), jnp.float32)
    out["compress_size"] = (lambda c, x: jnp.compress(c, x, size=4, fill_value=-5.0), [b9, f9])
    out["compress_size_long_cond"] = (lambda c, x: jnp.compress(c, x, size=3), [jnp.zeros((11,), jnp.bool_), f9])
    out["compress_size_short_cond"] = (lambda c, x: jnp.compress(c, x, size=3), [jnp.zeros((5,), jnp.bool_), f9])
    out["compress_size_axis1"] = (lambda c, x: jnp.compress(c, x, axis=1, size=2, fill_value=7.0), [jnp.zeros((3,), jnp.bool_), a])
    out["compress_size_int"] = (lambda c, x: jnp.compress(c, x, size=2, fill_value=-1.0), [b9, jnp.zeros((9,), jnp.int32)])
    out["extract_size"] = (lambda c, x: jnp.extract(c, x, size=5), [jnp.zeros((2, 3), jnp.bool_), a])
    return out


def flat(x):
    if isinstance(x, (tuple, list)):
        return [z for y in x for z in flat(y)]
    return [x]


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    rows = []
    for name, f in value_cases().items():
        row = {"id": name, "x64": x64}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = f()
            outs = []
            for a in flat(res):
                h = np.asarray(a)
                outs.append({"dtype": str(h.dtype), "shape": list(h.shape), "hex": h.tobytes().hex()})
            row["outs"] = outs
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    for name, (fn, args) in traced_cases().items():
        def f(*a):
            return fn(*a)

        row = {"id": name, "x64": x64, "args": [[str(a.dtype), list(a.shape)] for a in args]}
        try:
            row["jaxpr"] = str(jax.make_jaxpr(f)(*args))
            row["hlo"] = jax.jit(f).lower(*args).as_text()
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "audit_holes"))
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
