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

SECOND = {"i32": jnp.int32, "bool": jnp.bool_, "u8": jnp.uint8}
BWD = {
    "f32": lambda r, g: (g * 2.0, jnp.ones(3, jnp.float32)),
    "f32_shape": lambda r, g: (g * 2.0, jnp.ones(7, jnp.float32)),
    "same": lambda r, g: (g * 2.0, r[1]),
    "scalar": lambda r, g: (g * 2.0, jnp.float32(4.0)),
}


def primal(x, n):
    return x * 2.0 + n.astype(jnp.float32).sum()


def make(kind, symbolic):
    bwd = BWD[kind]
    f = jax.custom_vjp(primal)
    if symbolic:
        def fwd(x, n):
            return primal(x.value, n.value), (x.value, n.value)
    else:
        def fwd(x, n):
            return primal(x, n), (x, n)
    f.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    return f


def inputs(second):
    x = jnp.linspace(1.0, 2.0, 3, dtype=jnp.float32)
    n = jnp.asarray([1, 0, 3], dtype=SECOND[second])
    return x, n


def pulled(f):
    def run(x, n):
        _, pull = jax.vjp(f, x, n)
        return pull(jnp.linspace(0.5, 1.5, 3, dtype=jnp.float32))

    return run


def value_cases():
    out = {}
    for second in SECOND:
        for kind in BWD:
            for sym in (False, True):
                tag = "%s_%s_%s" % (second, kind, "sym" if sym else "plain")
                f = make(kind, sym)
                x, n = inputs(second)
                out["vjp_" + tag] = (lambda f=f, x=x, n=n: pulled(f)(x, n))
                out["jit_vjp_" + tag] = (lambda f=f, x=x, n=n: jax.jit(pulled(f))(x, n))
        for kind in ("f32", "same"):
            f = make(kind, False)
            x, n = inputs(second)
            out["grad_int_%s_%s" % (second, kind)] = (
                lambda f=f, x=x, n=n: jax.grad(lambda x, n: f(x, n).sum(), argnums=(0, 1), allow_int=True)(x, n)
            )
    for kind in BWD:
        for sym in (False, True):
            f = make(kind, sym)
            x, _ = inputs("i32")
            out["grad_derived_%s_%s" % (kind, "sym" if sym else "plain")] = (
                lambda f=f, x=x: jax.grad(lambda x: f(x, (x * 3.0).astype(jnp.int32)).sum())(x)
            )
    return out


def derived(f):
    return lambda x: jax.grad(lambda x: f(x, (x * 3.0).astype(jnp.int32)).sum())(x)


def through_jit(f):
    inner = jax.jit(lambda y: (y * 3.0).astype(jnp.int32) + 1)
    return lambda x: jax.grad(lambda x: f(x, inner(x)).sum())(x)


def traced_cases():
    out = {}
    for second in ("i32", "bool"):
        for kind in ("f32", "f32_shape"):
            for sym in (False, True):
                tag = "%s_%s_%s" % (second, kind, "sym" if sym else "plain")
                out["trace_vjp_" + tag] = (pulled(make(kind, sym)), list(inputs(second)))
    for kind in ("f32", "f32_shape"):
        for sym in (False, True):
            tag = "%s_%s" % (kind, "sym" if sym else "plain")
            out["trace_derived_" + tag] = (derived(make(kind, sym)), [inputs("i32")[0]])
            out["trace_jit_" + tag] = (through_jit(make(kind, sym)), [inputs("i32")[0]])
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
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "vjp_int"))
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
