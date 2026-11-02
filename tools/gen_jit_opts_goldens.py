import argparse
import contextlib
import io
import json
import os
import warnings

import numpy as np

RANK_FNS = [
    "add", "subtract", "multiply", "maximum", "minimum", "arctan2",
    "float_power", "nextafter", "equal", "not_equal", "greater",
    "greater_equal", "less", "less_equal", "bitwise_and", "bitwise_or",
    "bitwise_xor", "left_shift", "right_shift", "logical_and", "logical_or",
    "logical_xor", "true_divide", "power", "floor_divide", "remainder",
    "fmod", "copysign", "hypot", "logaddexp", "logaddexp2", "heaviside",
]
INT_FNS = {"bitwise_and", "bitwise_or", "bitwise_xor", "left_shift",
           "right_shift"}
SHAPES = [((3,), (2, 3)), ((2, 3), (3,)), ((), (2, 3)), ((1, 2), (4, 3, 2)),
          ((4,), (2, 3)), ((2, 3), (2, 3))]

NAN_CASES = [
    "eager_log", "eager_sqrt", "eager_div_zero_zero", "eager_div_one_zero",
    "eager_add_nan_input", "jit_log", "jit_sqrt_sum", "jit_identity_nan",
    "jit_pair_nan", "jit_nested_log", "jit_div_one_zero", "jit_clean",
    "eager_int_div",
]
TEXT_CASES = [
    "don_mul", "keep_unused", "don_pair", "don_sum", "don_reshape",
    "don_astype", "don_weak", "don_swap", "don_dup", "keep_don",
    "keep_unused_none", "don_none", "inline_lowered", "noinline_lowered",
    "inline_jaxpr", "noinline_jaxpr",
]
FLAGS = [("nans", True, False), ("infs", False, True), ("both", True, True)]


def operand(shape, dtype, offset):
    n = int(np.prod(shape))
    vals = (np.arange(n) % 5 + 1 + offset).reshape(shape)
    return np.asarray(vals, dtype=dtype)


def encode(out):
    a = np.asarray(out)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "hex": a.tobytes().hex()}


def run_case(thunk):
    notices = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(notices):
                out = thunk()
            if isinstance(out, (tuple, list)):
                result = {"kind": "ok", "outs": [encode(o) for o in out]}
            else:
                result = {"kind": "ok", "outs": [encode(out)]}
        except Exception as e:
            result = {"kind": "raise", "class": type(e).__name__,
                      "message": str(e)}
    result["warnings"] = [[w.category.__name__, str(w.message)]
                          for w in caught]
    result["notices"] = [l for l in notices.getvalue().splitlines() if l]
    return result


def nan_thunk(jax, jnp, lax, name, x, z, n, one):
    table = {
        "eager_log": lambda: lax.log(x),
        "eager_sqrt": lambda: lax.sqrt(x),
        "eager_div_zero_zero": lambda: lax.div(z, z),
        "eager_div_one_zero": lambda: lax.div(one, z),
        "eager_add_nan_input": lambda: lax.add(n, one),
        "jit_log": lambda: jax.jit(lambda a: lax.mul(lax.log(a), a))(x),
        "jit_sqrt_sum": lambda: jax.jit(
            lambda a: lax.reduce_sum(lax.sqrt(a), (0,)))(x),
        "jit_identity_nan": lambda: jax.jit(lambda a: a)(n),
        "jit_pair_nan": lambda: jax.jit(lambda a: (a, a))(n),
        "jit_nested_log": lambda: jax.jit(
            lambda a: jax.jit(lambda b: lax.log(b))(a))(x),
        "jit_div_one_zero": lambda: jax.jit(lambda a: lax.div(one, a))(z),
        "jit_clean": lambda: jax.jit(lambda a: lax.exp(a))(x),
        "eager_int_div": lambda: lax.div(jnp.asarray(np.array([1], np.int32)),
                                         jnp.asarray(np.array([0], np.int32))),
    }
    return table[name]


def text_case(jax, jnp, name):
    x3 = jnp.ones((3,), jnp.float32)
    y2 = jnp.ones((2,), jnp.float32)
    m23 = jnp.ones((2, 3), jnp.float32)
    jit = jax.jit
    table = {
        "don_mul": (lambda a, b: a * 2.0, (x3, y2), dict(donate_argnums=(0,))),
        "keep_unused": (lambda a, b: a * 2.0, (x3, y2),
                        dict(keep_unused=True)),
        "don_pair": (lambda a, b: (a * 2.0, b + 1.0), (x3, y2),
                     dict(donate_argnums=(0, 1))),
        "don_sum": (lambda a, b: jnp.sum(a), (x3, y2),
                    dict(donate_argnums=(0, 1))),
        "don_reshape": (lambda a: a.reshape(6), (m23,),
                        dict(donate_argnums=(0,))),
        "don_astype": (lambda a: a.astype(jnp.int32), (m23,),
                       dict(donate_argnums=(0,))),
        "don_weak": (lambda a, s: a * s, (y2, 2.0),
                     dict(donate_argnums=(0, 1))),
        "don_swap": (lambda a, b: (b, a), (y2, y2), dict(donate_argnums=(0,))),
        "don_dup": (lambda a, b: (a, a), (y2, y2),
                    dict(donate_argnums=(0, 1))),
        "keep_don": (lambda a, b: a, (y2, y2),
                     dict(donate_argnums=(1,), keep_unused=True)),
        "keep_unused_none": (lambda a, b: a + 1.0, (y2, x3),
                             dict(keep_unused=False)),
        "don_none": (lambda a, b: a + b, (y2, y2), dict(donate_argnums=())),
        "inline_lowered": (lambda a: jit(lambda b: jnp.sin(b), inline=True)(a)
                           + 1.0, (x3,), {}),
        "noinline_lowered": (lambda a: jit(lambda b: jnp.sin(b))(a) + 1.0,
                             (x3,), {}),
    }
    if name == "inline_jaxpr":
        return lambda: str(jax.make_jaxpr(
            lambda a: jit(lambda b: jnp.sin(b), inline=True)(a) + 1.0)(x3))
    if name == "noinline_jaxpr":
        return lambda: str(jax.make_jaxpr(
            lambda a: jit(lambda b: jnp.sin(b))(a) + 1.0)(x3))
    f, a, kw = table[name]
    return lambda: jit(f, **kw).lower(*a).as_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    from jax import lax
    rows = []
    for mode in ["raise", "warn"]:
        for fn in RANK_FNS:
            dtype = np.int32 if fn in INT_FNS else np.float32
            for i, (sa, sb) in enumerate(SHAPES):
                a = operand(sa, dtype, 0)
                b = operand(sb, dtype, 1)
                ja, jb = jnp.asarray(a), jnp.asarray(b)
                f = getattr(jnp, fn)
                with jax.numpy_rank_promotion(mode):
                    r = run_case(lambda: f(ja, jb))
                r.update({"id": f"rank_{mode}_{fn}_{i}", "group": "rank",
                          "mode": mode, "fn": fn,
                          "dtype": np.dtype(dtype).name,
                          "shapes": [list(sa), list(sb)]})
                rows.append(r)
    x = jnp.asarray(np.array([-1.0, 2.0], np.float32))
    z = jnp.asarray(np.array([0.0], np.float32))
    n = jnp.asarray(np.array([np.nan], np.float32))
    one = jnp.asarray(np.array([1.0], np.float32))
    for flag, nans, infs in FLAGS:
        for name in NAN_CASES:
            thunk = nan_thunk(jax, jnp, lax, name, x, z, n, one)
            with jax.debug_nans(nans), jax.debug_infs(infs):
                r = run_case(thunk)
            r.update({"id": f"special_{flag}_{name}", "group": "special",
                      "flag": flag, "case": name})
            rows.append(r)
    for name in TEXT_CASES:
        thunk = text_case(jax, jnp, name)
        holder = {}

        def run():
            holder["text"] = thunk()
            return []
        r = run_case(run)
        r.update({"id": f"text_{name}", "group": "text", "case": name,
                  "text": holder.get("text", "")})
        rows.append(r)
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
