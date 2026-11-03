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
from jax.experimental.ode import odeint

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def decay(y, t):
    return -y


def scaled(y, t, k):
    return -k * y


def forced(y, t, k):
    return jnp.sin(t) - k * y * y


def osc(y, t):
    x, v = y
    return (v, -x)


def inputs(dt):
    y0 = jnp.array([1.0, 2.0], dt)
    ts = jnp.linspace(0.0, 1.0, 5, dtype=dt)
    k = jnp.array(0.5, dt)
    return y0, ts, k


def value_cases(dt):
    y0, ts, k = inputs(dt)
    y1 = (jnp.array([1.0, 0.0], dt), jnp.array([0.0, 1.0], dt))
    tl = jnp.linspace(0.0, 3.0, 4, dtype=dt)
    return {
        "decay": lambda: odeint(decay, y0, ts),
        "scaled": lambda: odeint(scaled, y0, ts, k),
        "forced": lambda: odeint(forced, y0, tl, k),
        "osc": lambda: odeint(osc, y1, tl),
        "tol": lambda: odeint(forced, y0, tl, k, rtol=1e-3, atol=1e-4),
        "mxstep": lambda: odeint(forced, y0, tl, k, mxstep=3),
        "hmax": lambda: odeint(scaled, y0, ts, k, hmax=0.05),
        "jit_scaled": lambda: jax.jit(lambda y0, ts, k: odeint(scaled, y0, ts, k))(y0, ts, k),
        "grad_decay": lambda: jax.grad(lambda y0, ts: odeint(decay, y0, ts).sum(), argnums=(0, 1))(y0, ts),
        "grad_scaled": lambda: jax.grad(lambda y0, ts, k: odeint(scaled, y0, ts, k).sum(), argnums=(0, 1, 2))(y0, ts, k),
        "grad_forced": lambda: jax.grad(lambda y0, ts, k: (odeint(forced, y0, tl, k) ** 2).sum(), argnums=(0, 1, 2))(y0, tl, k),
        "grad_osc": lambda: jax.grad(lambda y1: odeint(osc, y1, tl)[0].sum())(y1),
        "int_t": lambda: odeint(decay, y0, jnp.arange(3)),
    }


def traced_cases(dt):
    y0, ts, k = inputs(dt)
    return {
        "jaxpr_decay": (lambda y0, ts: odeint(decay, y0, ts), [y0, ts]),
        "jaxpr_scaled": (lambda y0, ts, k: odeint(scaled, y0, ts, k), [y0, ts, k]),
        "jaxpr_grad_scaled": (
            lambda y0, ts, k: jax.grad(lambda y0, ts, k: odeint(scaled, y0, ts, k).sum(), argnums=(0, 2))(y0, ts, k),
            [y0, ts, k],
        ),
    }


def flat(x):
    if isinstance(x, (tuple, list)):
        return [z for y in x for z in flat(y)]
    return [x]


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    rows = []
    for dname, dt in (("f32", jnp.float32), ("f64", jnp.float64)):
        if dt == jnp.float64 and not x64:
            continue
        for name, f in value_cases(dt).items():
            row = {"id": name + "_" + dname, "x64": x64}
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
        for name, (fn, args) in traced_cases(dt).items():
            row = {"id": name + "_" + dname, "x64": x64, "args": [[str(a.dtype), list(a.shape)] for a in args]}
            try:
                row["jaxpr"] = str(jax.make_jaxpr(fn)(*args))
                row["hlo"] = jax.jit(fn).lower(*args).as_text()
            except Exception as e:
                row["error_class"] = type(e).__name__
                row["message"] = str(e)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "odeint"))
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
