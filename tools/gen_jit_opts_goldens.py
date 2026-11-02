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
    "eager_int_div", "jnp_log", "jnp_sqrt", "jnp_div", "jnp_div1",
    "jnp_arccos", "jnp_mean_jit", "vmap_log", "grad_sqrt", "grad_log",
    "cond_log", "fori_log", "softmax_inf",
]
TEXT_CASES = [
    "don_mul", "keep_unused", "don_pair", "don_sum", "don_reshape",
    "don_astype", "don_weak", "don_swap", "don_dup", "keep_don",
    "keep_unused_none", "don_none", "inline_lowered", "noinline_lowered",
    "inline_jaxpr", "noinline_jaxpr",
]
RANDOM_CASES = [
    ("key_data", 0), ("key_data", 7), ("key_data", -3), ("prng_key", 7),
    ("uniform", 0), ("uniform", 7), ("jit_key_data", 7),
    ("gumbel", "low"), ("gumbel", "high"), ("categorical", "low"),
    ("categorical", "high"), ("gumbel_f64", "high"),
]
DONATE_RUN_CASES = ["run_ok", "run_use", "run_again", "run_unusable",
                    "run_static", "run_keep"]
COPTS_CASES = ["none", "fast_min_max_on", "fast_min_max_off",
               "fast_math_on", "bogus"]
SPECIAL_FNS = ["betaln", "beta", "betainc", "gammainc", "gammaincc", "xlogy",
               "xlog1py", "boxcox", "boxcox1p", "rel_entr", "kl_div",
               "polygamma", "zeta", "poch", "owens_t", "expn"]
SPECIAL_SHAPES = [((3,), (2, 3)), ((2, 3), (3,)), ((1, 2), (4, 3, 2))]
STATS_FNS = {
    "norm": ["logpdf", "pdf", "cdf", "logcdf", "ppf", "sf", "logsf", "isf"],
    "cauchy": ["logpdf", "pdf", "cdf", "logcdf", "sf", "logsf", "isf", "ppf"],
    "logistic": ["logpdf", "pdf", "cdf", "sf", "ppf", "isf"],
    "expon": ["logpdf", "pdf", "cdf", "logsf", "sf", "logcdf", "ppf"],
    "uniform": ["logpdf", "pdf", "cdf", "ppf"],
}
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
    z0 = jnp.asarray(np.float32(0.0))
    inf1 = jnp.asarray(np.array([np.inf, 1.0], np.float32))
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
        "jnp_log": lambda: jnp.log(x),
        "jnp_sqrt": lambda: jnp.sqrt(x),
        "jnp_div": lambda: jnp.divide(z, z),
        "jnp_div1": lambda: jnp.divide(1.0, z),
        "jnp_arccos": lambda: jnp.arccos(x * 2.0),
        "jnp_mean_jit": lambda: jax.jit(lambda a: jnp.mean(jnp.log(a)))(x),
        "vmap_log": lambda: jax.vmap(lax.log)(x),
        "grad_sqrt": lambda: jax.grad(lax.sqrt)(z0),
        "grad_log": lambda: jax.grad(lax.log)(z0),
        "cond_log": lambda: lax.cond(True, lax.log, lambda a: a, x),
        "fori_log": lambda: lax.fori_loop(0, 2, lambda i, a: lax.log(a), x),
        "softmax_inf": lambda: jax.nn.softmax(inf1),
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


def random_case(jax, jnp, name, arg):
    r = jax.random
    offset = arg if isinstance(arg, int) else 0

    def body():
        if name == "key_data":
            return r.key_data(r.key(11))
        if name == "prng_key":
            return r.PRNGKey(11)
        if name == "uniform":
            return r.uniform(r.key(3), (4,))
        if name == "jit_key_data":
            return jax.jit(lambda s: r.key_data(r.key(s)))(
                jnp.asarray(np.int32(11)))
        if name == "gumbel":
            return r.gumbel(r.key(1), (5,))
        if name == "gumbel_f64":
            return r.gumbel(r.key(1), (5,), dtype=jnp.float64)
        if name == "categorical":
            logits = jnp.asarray(np.array([[0.1, -2.0, 3.0, 0.5],
                                           [1.0, 1.0, -30.0, 2.0]],
                                          np.float32))
            return r.categorical(r.key(2), logits, shape=(6, 2))
        raise KeyError(name)

    def run():
        prev = jax.config.jax_random_seed_offset
        jax.config.update("jax_random_seed_offset", offset)
        try:
            return body()
        finally:
            jax.config.update("jax_random_seed_offset", prev)
    return run


def donate_run_case(jax, jnp, name):
    def run():
        x = jnp.ones((3,), jnp.float32)
        f = jax.jit(lambda a: a * 2.0, donate_argnums=0)
        if name == "run_ok":
            y = f(x)
            return [y, np.array(x.is_deleted())]
        if name == "run_use":
            f(x)
            return [jnp.add(x, 1.0)]
        if name == "run_again":
            f(x)
            return [f(x)]
        if name == "run_unusable":
            g = jax.jit(lambda a: jnp.sum(a), donate_argnums=0)
            return [g(x), np.array(x.is_deleted())]
        if name == "run_static":
            h = jax.jit(lambda n, a: a * n, static_argnums=0,
                        donate_argnums=1)
            y = h(3.0, x)
            return [y, np.array(x.is_deleted())]
        if name == "run_keep":
            w = jnp.ones((2,), jnp.float32)
            k = jax.jit(lambda a, b: a * 2.0, donate_argnums=(0, 1),
                        keep_unused=True)
            y = k(x, w)
            return [y, np.array(x.is_deleted()), np.array(w.is_deleted())]
        raise KeyError(name)
    return run


def copts_case(jax, jnp, name):
    opts = {
        "none": None,
        "fast_min_max_on": {"xla_cpu_enable_fast_min_max": True},
        "fast_min_max_off": {"xla_cpu_enable_fast_min_max": False},
        "fast_math_on": {"xla_cpu_enable_fast_math": True},
        "bogus": {"xla_bogus_flag": 1},
    }[name]
    x = jnp.asarray(np.array([np.nan, 1.0, 2.0], np.float32))
    return lambda: jax.jit(lambda a: jnp.max(a), compiler_options=opts)(x)


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
    import jax.scipy.special as jsp
    for mode in ["raise", "warn"]:
        for fn in SPECIAL_FNS:
            for i, (sa, sb) in enumerate(SPECIAL_SHAPES):
                a = jnp.asarray(operand(sa, np.float32, 0))
                b = jnp.asarray(operand(sb, np.float32, 1))
                f = getattr(jsp, fn)
                if fn == "betainc":
                    def call(f=f, a=a, b=b):
                        return f(a, b, a * 0.1)
                else:
                    def call(f=f, a=a, b=b):
                        return f(a, b)
                with jax.numpy_rank_promotion(mode):
                    r = run_case(call)
                r.update({"id": f"rank_special_{mode}_{fn}_{i}",
                          "group": "rank_special", "mode": mode, "fn": fn,
                          "shapes": [list(sa), list(sb)]})
                rows.append(r)
    import jax.scipy.stats as jst
    for mode in ["raise", "warn"]:
        for dist, fns in STATS_FNS.items():
            for fn in fns:
                for i, (sa, sb) in enumerate(SPECIAL_SHAPES[:2]):
                    a = jnp.asarray(operand(sa, np.float32, 0) * 0.1)
                    b = jnp.asarray(operand(sb, np.float32, 1) * 0.05)
                    f = getattr(getattr(jst, dist), fn)
                    with jax.numpy_rank_promotion(mode):
                        r = run_case(lambda f=f, a=a, b=b: f(a, b))
                    r.update({"id": f"rank_stats_{mode}_{dist}_{fn}_{i}",
                              "group": "rank_stats", "mode": mode,
                              "dist": dist, "fn": fn,
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
    for name, arg in RANDOM_CASES:
        prev = jax.config.jax_high_dynamic_range_gumbel
        jax.config.update("jax_high_dynamic_range_gumbel", arg == "high")
        try:
            r = run_case(random_case(jax, jnp, name, arg))
        finally:
            jax.config.update("jax_high_dynamic_range_gumbel", prev)
        r.update({"id": f"random_{name}_{arg}", "group": "random",
                  "case": name, "arg": str(arg)})
        rows.append(r)
    for name in DONATE_RUN_CASES:
        r = run_case(donate_run_case(jax, jnp, name))
        r.update({"id": f"donate_{name}", "group": "donate_run", "case": name})
        rows.append(r)
    for name in COPTS_CASES:
        r = run_case(copts_case(jax, jnp, name))
        r.update({"id": f"copts_{name}", "group": "copts", "case": name})
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
