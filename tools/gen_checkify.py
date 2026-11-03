import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import argparse
import itertools
import json
import platform
import warnings

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import checkify as checkify_src
from jax.experimental import checkify

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh_codes():
    checkify_src.next_code = itertools.count(1).__next__


def pos(x):
    checkify.check(x[0] > 0, "x[0] must be positive, got {}", x[0])
    return jnp.sin(x)


def two(x):
    checkify.check(x[0] > 0, "first {}", x[0])
    y = x * 2
    checkify.check(x[1] < 3, "second {} and {}", x[1], x[2])
    checkify.check(x[2] < 3, "third {}", x[2])
    return y + 1


def kw(x):
    checkify.check(jnp.all(x > 0), "{x} needs to be positive, total {t}", x=x, t=x.sum())
    return x * x


def plain(x):
    checkify.check(x[0] > 0, "no args")
    return jnp.cos(x)


def nonscalar(x):
    checkify.check(x > 0, "bad")
    return x


def inputs(dt):
    return {
        "ok": jnp.array([1.0, 2.0, 3.0], dt),
        "neg": jnp.array([-1.0, 2.0, 3.0], dt),
        "big": jnp.array([1.0, 4.0, 5.0], dt),
        "neg_big": jnp.array([-2.0, 4.0, 2.0], dt),
    }


FUNS = {"pos": pos, "two": two, "kw": kw, "plain": plain}


def value_cases(dt):
    xs = inputs(dt)
    out = {}
    for fname, f in FUNS.items():
        for xname, x in xs.items():
            out["%s_%s" % (fname, xname)] = (lambda f=f, x=x: checkify.checkify(f)(x), "get")
            out["jit_%s_%s" % (fname, xname)] = (lambda f=f, x=x: jax.jit(checkify.checkify(f))(x), "get")
            out["throw_%s_%s" % (fname, xname)] = (lambda f=f, x=x: checkify.checkify(f)(x), "throw")
        out["eager_%s_neg" % fname] = (lambda f=f: f(xs["neg"]), "plain")
        out["eager_%s_ok" % fname] = (lambda f=f: f(xs["ok"]), "plain")
        out["nan_errors_%s_neg" % fname] = (lambda f=f: checkify.checkify(f, errors=checkify.nan_checks)(xs["neg"]), "get")
        out["all_errors_%s_neg" % fname] = (lambda f=f: checkify.checkify(f, errors=checkify.all_checks)(xs["neg"]), "get")
    out["nonscalar"] = (lambda: checkify.checkify(nonscalar)(xs["ok"]), "get")
    return out


def traced_cases(dt):
    x = inputs(dt)["ok"]
    out = {}
    for fname, f in FUNS.items():
        out["jaxpr_checkify_%s" % fname] = (lambda x, f=f: checkify.checkify(f)(x), [x], True)
        out["jaxpr_raw_%s" % fname] = (f, [x], False)
    return out


def flat(x):
    if isinstance(x, (tuple, list)):
        return [z for y in x for z in flat(y)]
    return [x]


def enc(a):
    h = np.asarray(a)
    return {"dtype": str(h.dtype), "shape": list(h.shape), "hex": h.tobytes().hex()}


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    rows = []
    for dname, dt in (("f32", jnp.float32), ("f64", jnp.float64)):
        if dt == jnp.float64 and not x64:
            continue
        for name, (f, mode) in value_cases(dt).items():
            row = {"id": name + "_" + dname, "x64": x64, "mode": mode}
            fresh_codes()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = f()
                if mode == "plain":
                    row["outs"] = [enc(a) for a in flat(res)]
                else:
                    err, out = res
                    leaves = jax.tree_util.tree_leaves(err)
                    row["outs"] = [enc(a) for a in leaves + flat(out)]
                    if mode == "get":
                        row["get"] = err.get()
                    else:
                        try:
                            err.throw()
                            row["thrown"] = None
                        except Exception as e:
                            row["thrown"] = type(e).__name__ + ": " + str(e)
            except Exception as e:
                row["error_class"] = type(e).__name__
                row["message"] = str(e)
            rows.append(row)
        for name, (fn, args, lower) in traced_cases(dt).items():
            row = {"id": name + "_" + dname, "x64": x64, "args": [[str(a.dtype), list(a.shape)] for a in args]}
            try:
                fresh_codes()
                row["jaxpr"] = str(jax.make_jaxpr(fn)(*args))
                fresh_codes()
                if lower:
                    row["hlo"] = jax.jit(fn).lower(*args).as_text()
            except Exception as e:
                row["error_class"] = type(e).__name__
                row["message"] = str(e)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "checkify"))
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
