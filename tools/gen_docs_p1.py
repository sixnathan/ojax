import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import argparse
import contextlib
import io
import json
import platform

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import source_info_util
from jax.ad_checkpoint import checkpoint_name, print_saved_residuals
from jax.custom_batching import custom_vmap, sequential_vmap

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SEED = 0
BASE = {}


def set_base(seed):
    BASE.clear()
    if seed != DEFAULT_SEED:
        rng = np.random.default_rng([seed, 7])
        BASE["x"] = [float(v) / 4 for v in rng.integers(-12, 17, 3)]
        BASE["y"] = [float(v) / 4 for v in rng.integers(-12, 17, 3)]
        BASE["m"] = [float(v) / 4 for v in rng.integers(-12, 17, 6)]


def fdt():
    return jnp.arange(3.0).dtype


def vec(key, default):
    if key in BASE:
        return jnp.asarray(BASE[key], dtype=fdt())
    return default


def inputs():
    x = vec("x", jnp.arange(3.0) / 2 + 0.25)
    y = vec("y", jnp.arange(3.0) - 1.0)
    m = vec("m", jnp.arange(6.0) / 4).reshape(2, 3)
    return x, y, m


def residual_cases():
    x, y, m = inputs()
    c = jnp.asarray(np.array([1.5, -0.5, 2.0]), dtype=fdt())

    def inner(*args):
        return jnp.sin(args[0]) * args[0]

    inner_jit = jax.jit(inner)

    def policy_of(name):
        return getattr(jax.checkpoint_policies, name)

    def remat_sin(policy):
        def f(*args):
            def g(*a):
                return jnp.sin(jnp.dot(a[1], a[0])) * a[0][0]
            return jax.checkpoint(g, policy=policy)(*args)
        return f

    cases = {
        "sin_mul": (lambda *args: jnp.sin(args[0]) * args[1], [x, y]),
        "sin_sin": (lambda *args: jnp.sin(jnp.sin(args[0])), [x]),
        "mul_args": (lambda *args: args[0] * args[1], [x, y]),
        "exp_log": (lambda *args: jnp.log(jnp.exp(args[0]) + 2.0), [x]),
        "tanh_sum": (lambda *args: jnp.sum(jnp.tanh(args[0]) * args[1]), [x, y]),
        "dot": (lambda *args: jnp.dot(args[1], args[0]), [x, m]),
        "literal": (lambda *args: jnp.sin(args[0]) * 2.0, [x]),
        "constant": (lambda *args: jnp.sin(args[0]) * c, [x]),
        "named": (lambda *args: checkpoint_name(jnp.sin(args[0]), "h") * args[0], [x]),
        "named_two": (lambda *args: checkpoint_name(jnp.cos(args[0]), "a")
                      * checkpoint_name(jnp.sin(args[1]), "b"), [x, y]),
        "jit_inner": (lambda *args: inner_jit(*args) * args[1], [x, y]),
        "remat_sin_sin": (lambda *args: jax.checkpoint(
            lambda *a: jnp.sin(jnp.sin(a[0])))(*args), [x]),
        "remat_unused": (lambda *args: jnp.sum(args[1]) * 1.0, [x, y]),
        "integer_arg": (lambda *args: jnp.sin(args[0]) * args[1], [x, jnp.arange(3)]),
        "square": (lambda *args: args[0] ** 2, [x]),
        "div": (lambda *args: args[0] / args[1], [x, y + 3.0]),
    }
    for p in ["nothing_saveable", "everything_saveable", "dots_saveable",
              "dots_with_no_batch_dims_saveable"]:
        cases["policy_" + p] = (remat_sin(policy_of(p)), [x, m])
    return cases


def residual_text(fn, args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_saved_residuals(fn, *args)
    return out.getvalue()


def residual_rows(x64):
    rows = []
    for name, (fn, args) in residual_cases().items():
        row = {"id": "residuals_" + name, "x64": x64, "group": "residuals", "case": name}
        row.update(BASE)
        try:
            row["text"] = residual_text(fn, args)
            source_info_util.register_exclusion(os.path.abspath(__file__))
            try:
                row["text_noframe"] = residual_text(fn, args)
            finally:
                source_info_util._exclude_paths.remove(os.path.abspath(__file__))
                source_info_util._exclude_path_regex.cache_clear()
                source_info_util.is_user_filename.cache_clear()
                source_info_util.user_frame.cache_clear()
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    return rows


def vmap_fns():
    def plain(*args):
        return jnp.sin(args[0]) * args[1]

    def rule(axis_size, in_batched, *args):
        a, b = args
        out = jnp.cos(a) + b
        return out, in_batched[0] or in_batched[1]

    f = custom_vmap(plain)
    f.def_vmap(rule)

    def one(*args):
        return args[0] * 3.0

    def rule_one(axis_size, in_batched, *args):
        return jnp.sum(args[0], axis=0, keepdims=True) + args[0] * 0.0, True

    g = custom_vmap(one)
    g.def_vmap(rule_one)

    def rule_unbatched(axis_size, in_batched, *args):
        return args[0][0] * 2.0, False

    h = custom_vmap(one)
    h.def_vmap(rule_unbatched)

    def rule_bad_tree(axis_size, in_batched, *args):
        return (args[0], args[0]), True

    bad = custom_vmap(one)
    bad.def_vmap(rule_bad_tree)

    def rule_bad_spec(axis_size, in_batched, *args):
        return args[0], (True, True)

    bad_spec = custom_vmap(one)
    bad_spec.def_vmap(rule_bad_spec)

    norule = custom_vmap(one)
    seq = sequential_vmap(lambda *args: jnp.sum(jnp.sin(args[0])) * args[1])
    return {"f": f, "g": g, "h": h, "bad": bad, "bad_spec": bad_spec,
            "norule": norule, "seq": seq}


def text_of(out):
    lines = []
    for a in jax.tree_util.tree_leaves(out):
        h = np.asarray(a)
        lines.append(f"{h.dtype} {list(h.shape)} {h.tobytes().hex()}")
    return "\n".join(lines)


def vmap_cases():
    x, y, m = inputs()
    fs = vmap_fns()
    f, g, h = fs["f"], fs["g"], fs["h"]
    return {
        "f_call": (lambda: f(x, y), lambda a, b: f(a, b), [x, y]),
        "f_vmap": (lambda: jax.vmap(f)(m, m + 1.0), lambda a, b: jax.vmap(f)(a, b), [m, m]),
        "f_vmap_in0": (lambda: jax.vmap(f, in_axes=(0, None))(m, y),
                       lambda a, b: jax.vmap(f, in_axes=(0, None))(a, b), [m, y]),
        "f_vmap_in1": (lambda: jax.vmap(f, in_axes=(None, 0))(y, m),
                       lambda a, b: jax.vmap(f, in_axes=(None, 0))(a, b), [y, m]),
        "f_vmap_axis1": (lambda: jax.vmap(f, in_axes=1)(m, m),
                         lambda a, b: jax.vmap(f, in_axes=1)(a, b), [m, m]),
        "f_vmap_out1": (lambda: jax.vmap(f, out_axes=1)(m, m),
                        lambda a, b: jax.vmap(f, out_axes=1)(a, b), [m, m]),
        "f_jit_vmap": (lambda: jax.jit(jax.vmap(f))(m, m),
                       lambda a, b: jax.jit(jax.vmap(f))(a, b), [m, m]),
        "f_vmap_jit": (lambda: jax.vmap(jax.jit(f))(m, m),
                       lambda a, b: jax.vmap(jax.jit(f))(a, b), [m, m]),
        "f_vmap_vmap": (lambda: jax.vmap(jax.vmap(f))(m.reshape(2, 3, 1), m.reshape(2, 3, 1)),
                        lambda a, b: jax.vmap(jax.vmap(f))(a, b),
                        [m.reshape(2, 3, 1), m.reshape(2, 3, 1)]),
        "g_vmap": (lambda: jax.vmap(g)(m), lambda a: jax.vmap(g)(a), [m]),
        "g_call": (lambda: g(x), lambda a: g(a), [x]),
        "h_vmap": (lambda: jax.vmap(h)(m), lambda a: jax.vmap(h)(a), [m]),
        "bad_vmap": (lambda: jax.vmap(fs["bad"])(m), lambda a: jax.vmap(fs["bad"])(a), [m]),
        "bad_spec_vmap": (lambda: jax.vmap(fs["bad_spec"])(m),
                          lambda a: jax.vmap(fs["bad_spec"])(a), [m]),
        "norule_call": (lambda: fs["norule"](x), lambda a: fs["norule"](a), [x]),
        "seq_vmap": (lambda: jax.vmap(fs["seq"])(m, y[:2]),
                     lambda a, b: jax.vmap(fs["seq"])(a, b), [m, y[:2]]),
        "seq_vmap_in0": (lambda: jax.vmap(fs["seq"], in_axes=(0, None))(m, y[0]),
                         lambda a, b: jax.vmap(fs["seq"], in_axes=(0, None))(a, b),
                         [m, y[0]]),
    }


def vmap_rows(x64):
    rows = []
    for name, (thunk, fn, args) in vmap_cases().items():
        row = {"id": "vmap_" + name, "x64": x64, "group": "vmap", "case": name}
        row.update(BASE)
        for mode in ["value", "jaxpr", "lowered"]:
            try:
                if mode == "value":
                    row["value"] = text_of(thunk())
                elif mode == "jaxpr":
                    row["jaxpr"] = str(jax.make_jaxpr(fn)(*args))
                else:
                    row["lowered"] = jax.jit(fn).lower(*args).as_text()
            except Exception as e:
                row[mode] = "raise " + type(e).__name__ + ": " + str(e)
        rows.append(row)
    return rows


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    return residual_rows(x64) + vmap_rows(x64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "docs_p1"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if platform.python_version() != PY_VERSION or jax.__version__ != JAX_VERSION:
        raise SystemExit("needs python " + PY_VERSION + " and jax " + JAX_VERSION)
    set_base(args.seed)
    rows = run(False) + run(True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cases.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
