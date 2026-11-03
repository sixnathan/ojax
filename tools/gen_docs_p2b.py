import argparse
import contextlib
import io
import json
import warnings

import numpy as np

RANK_OPS = ["add", "multiply", "subtract", "power", "arctan2", "maximum",
            "logical_and", "equal", "where", "add_scalar"]
RANK_MODES = ["warn", "raise", "allow"]
CAST_CASES = [("complex64", "float32"), ("complex64", "int32"),
              ("complex64", "bool"), ("complex64", "complex64"),
              ("float32", "int32"), ("complex64", "uint8"),
              ("complex64", "float16")]
CAST_FNS = ["astype", "method", "lax", "lax_jit", "astype_jit"]
ONE_HOT = ["float32", "bool", "int32", "uint8", "float16"]
SOLVES = ["cho_solve", "lu_solve", "solve_triangular"]
SOLVE_SHAPES = ["batched_1d", "vector", "matrix", "jit_batched_1d"]
PROMO_PAIRS = [("int32", "float32"), ("int8", "uint8"), ("float16", "bfloat16"),
               ("int32", "int32"), ("float32", "complex64"),
               ("bool", "int32"), ("uint8", "uint16"), ("float32", "float32")]
PROMO_FNS = ["add", "multiply", "where", "concatenate"]
X64_MODES = ["allow", "warn", "error"]
X64_FNS = [("zeros", "float64"), ("zeros", "int64"), ("full", "float64"),
           ("arange", "int64"), ("astype", "float64"), ("ones", "uint64"),
           ("asarray", "float64"), ("zeros", "float32")]
LEGACY_MODES = ["allow", "warn", "error"]
LEGACY_FNS = ["normal", "split", "typed_normal"]
SIGNAL_CASES = [("welch", "complex", True), ("welch", "real", True),
                ("welch", "complex", False), ("csd", "complex_y", True),
                ("csd", "real", True), ("stft", "complex", True),
                ("stft", "real", True)]
POLY_CASES = ["parse", "parse_bad", "parse_bad2", "ge1", "ge2", "eq",
              "eq_self", "eq_const", "lt", "mul", "pow", "divmod", "floordiv",
              "divmod_sym", "mod", "jmax", "min3", "cons", "cons_bad",
              "neg_ge", "sub_ge", "like", "like_bad", "gt_const", "le_self",
              "mod_sym", "cons_eq"]
OOM_CASES = [("zeros", 40), ("zeros", 43), ("broadcast", 40), ("broadcast", 43)]
DEVICE_CASES = ["cpu", "bogus", "", "CPU", "it's"]
SDG_CASES = ["cudnn", "none"]
REUSE_CASES = ["twice_eager", "twice_jit", "consumed_eager", "consumed_jit",
               "split_once", "split_twice", "split_twice_jit", "bits_fold_in",
               "clone", "off_twice", "while_reuse", "cond_branches",
               "bool_index_jit", "scan_const", "scan_moved", "scan_ok"]


def run_case(jax, thunk):
    notices = io.StringIO()
    jax.clear_caches()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(notices):
                out = thunk()
                jax.effects_barrier()
            result = {"kind": "ok", "text": out if isinstance(out, str)
                      else ""}
        except Exception as e:
            result = {"kind": "raise", "class": type(e).__name__,
                      "message": str(e), "text": ""}
    result["warnings"] = [[w.category.__name__, str(w.message)]
                          for w in caught]
    result["stdout"] = notices.getvalue()
    return result


def text_of(out):
    a = np.asarray(out)
    return f"{a.dtype} {list(a.shape)} {a.tobytes().hex()}"


DEFAULT_SEED = 0
BASE = {}
BASE_SIZES = (("rank_x", 3), ("rank_m", 6), ("b2", 6), ("b1", 3), ("bm", 6),
              ("sdg_a", 6), ("sdg_b", 6))


def set_base(seed):
    BASE.clear()
    if seed != DEFAULT_SEED:
        rng = np.random.default_rng([seed, 6])
        for key, n in BASE_SIZES:
            BASE[key] = [float(v) / 4 for v in rng.integers(-12, 17, n)]


def base(key, default, shape):
    if key in BASE:
        return np.array(BASE[key], dtype=np.float32).reshape(shape)
    return default


def rank_thunk(jax, jnp, op, mode):
    x = jnp.asarray(base("rank_x", np.arange(3, dtype=np.float32) + 1, (3,)))
    m = jnp.asarray(base("rank_m", np.arange(6, dtype=np.float32).reshape(2, 3) / 2, (2, 3)))

    def f(a, b):
        if op == "add":
            return jnp.add(a, b)
        if op == "multiply":
            return jax.jit(jnp.multiply)(a, b)
        if op == "subtract":
            return jnp.subtract(b, a)
        if op == "power":
            return jnp.power(a, b)
        if op == "arctan2":
            return jnp.arctan2(a, b)
        if op == "maximum":
            return jnp.maximum(a, b)
        if op == "logical_and":
            return jnp.logical_and(a > 1, b > 1)
        if op == "equal":
            return jnp.equal(a, b)
        if op == "where":
            return jnp.where(a > 1, a, b)
        if op == "add_scalar":
            return jnp.add(a, 2.0)
        raise KeyError(op)

    def run():
        with jax.numpy_rank_promotion(mode):
            return text_of(f(x, m))
    return run


def cast_thunk(jax, jnp, src, dst, fn):
    vals = np.array([1.5 + 2.0j, -2.25 - 1.0j, 0.0 + 3.0j], dtype=np.complex64)
    if src == "float32":
        vals = np.array([1.5, -2.25, 3.0], dtype=np.float32)
    x = jnp.asarray(vals)
    dt = np.dtype(dst)

    def run():
        if fn == "astype":
            return text_of(jnp.astype(x, dt))
        if fn == "method":
            return text_of(x.astype(dt))
        if fn == "lax":
            return text_of(jax.lax.convert_element_type(x, dt))
        if fn == "lax_jit":
            return text_of(jax.jit(
                lambda a: jax.lax.convert_element_type(a, dt))(x))
        if fn == "astype_jit":
            return text_of(jax.jit(lambda a: jnp.astype(a, dt))(x))
        raise KeyError(fn)
    return run


def one_hot_thunk(jax, jnp, dt):
    x = jnp.asarray(np.array([0, 2, 1, 1], dtype=np.int32)).astype(dt)
    return lambda: text_of(jax.nn.one_hot(x, 3))


def solve_thunk(jax, jnp, fn, kind):
    a = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]],
                 dtype=np.float32)
    a2 = jnp.asarray(np.stack([a, a + np.eye(3, dtype=np.float32)]))
    b2 = jnp.asarray(base("b2", np.arange(6, dtype=np.float32).reshape(2, 3) + 1, (2, 3)))
    b1 = jnp.asarray(base("b1", np.arange(3, dtype=np.float32) + 1, (3,)))
    bm = jnp.asarray(base("bm", np.arange(6, dtype=np.float32).reshape(3, 2) + 1, (3, 2)))
    sl = jax.scipy.linalg

    def call(a, b):
        if fn == "cho_solve":
            return sl.cho_solve(sl.cho_factor(a, lower=True), b)
        if fn == "lu_solve":
            return sl.lu_solve(sl.lu_factor(a), b)
        if fn == "solve_triangular":
            return sl.solve_triangular(a, b, lower=False)
        raise KeyError(fn)

    def run():
        if kind == "batched_1d":
            return text_of(call(a2, b2))
        if kind == "jit_batched_1d":
            return text_of(jax.jit(call)(a2, b2))
        if kind == "vector":
            return text_of(call(jnp.asarray(a), b1))
        if kind == "matrix":
            return text_of(call(jnp.asarray(a), bm))
        raise KeyError(kind)
    return run


def promo_thunk(jax, jnp, d1, d2, fn):
    x = jnp.asarray(np.array([1, 2, 3])).astype(d1)
    y = jnp.asarray(np.array([3, 0, 1])).astype(d2)

    def run():
        with jax.numpy_dtype_promotion("strict"):
            if fn == "add":
                return text_of(jnp.add(x, y))
            if fn == "multiply":
                return text_of(jax.jit(jnp.multiply)(x, y))
            if fn == "where":
                return text_of(jnp.where(x > y, x, y))
            if fn == "concatenate":
                return text_of(jnp.concatenate([x, y]))
            raise KeyError(fn)
    return run


def x64_thunk(jax, jnp, mode, fn, dt):
    enum = {"allow": "ALLOW", "warn": "WARN", "error": "ERROR"}[mode]
    from jax._src import config as jc
    x = jnp.asarray(np.array([1.5, 2.0, -3.0], dtype=np.float32))

    def run():
        with jc.explicit_x64_dtypes(getattr(jc.ExplicitX64Mode, enum)):
            if fn == "zeros":
                return text_of(jnp.zeros(2, dtype=dt))
            if fn == "ones":
                return text_of(jnp.ones(2, dtype=dt))
            if fn == "full":
                return text_of(jnp.full((2,), 7, dtype=dt))
            if fn == "arange":
                return text_of(jnp.arange(4, dtype=dt))
            if fn == "astype":
                return text_of(x.astype(dt))
            if fn == "asarray":
                return text_of(jnp.asarray(x, dtype=dt))
            raise KeyError(fn)
    return run


def legacy_thunk(jax, jnp, mode, fn):
    def run():
        with jax.legacy_prng_key(mode):
            if fn == "normal":
                return text_of(jax.random.normal(jax.random.PRNGKey(3), (2,)))
            if fn == "split":
                return text_of(jax.random.split(jax.random.PRNGKey(5)))
            if fn == "typed_normal":
                return text_of(jax.random.normal(jax.random.key(3), (2,)))
            raise KeyError(fn)
    return run


def reuse_thunk(jax, jnp, name):
    r = jax.random

    def twice(k):
        return r.normal(k), r.uniform(k)

    def split_twice(k):
        return r.split(k), r.split(k)

    def run():
        if name == "bool_index_jit":
            x = jnp.arange(3.0)
            return text_of(jax.jit(lambda a: a[a > 0])(x))
        flag = name != "off_twice"
        with jax.debug_key_reuse(flag):
            if name in ("twice_eager", "off_twice"):
                out = twice(r.key(1))
            elif name == "twice_jit":
                out = jax.jit(twice)(r.key(1))
            elif name == "consumed_eager":
                k = r.key(2)
                r.normal(k)
                out = (r.normal(k),)
            elif name == "consumed_jit":
                k = r.key(3)
                r.normal(k)
                out = (jax.jit(r.normal)(k),)
            elif name == "split_once":
                out = (r.key_data(r.split(r.key(4))),)
            elif name == "split_twice":
                out = split_twice(r.key(5))
            elif name == "split_twice_jit":
                out = jax.jit(split_twice)(r.key(5))
            elif name == "bits_fold_in":
                k = r.key(6)
                out = (r.bits(k), r.key_data(r.fold_in(k, 1)))
            elif name == "clone":
                k = r.key(7)
                out = (r.normal(r.clone(k)), r.normal(k))
            elif name == "while_reuse":
                def body(c):
                    i, k, s = c
                    return i + 1, k, s + r.uniform(k)
                out = jax.lax.while_loop(lambda c: c[0] < 2, body,
                                         (0, r.key(8), 0.0))[2:]
            elif name == "scan_const":
                k = r.key(10)
                out = jax.lax.scan(lambda c, x: (c + r.uniform(k), None), 0.0,
                                   None, length=2)[:1]
            elif name == "scan_moved":
                out = jax.lax.scan(
                    lambda c, x: ((c[0], c[1] + r.uniform(c[0])), None),
                    (r.key(11), 0.0), None, length=2)[0][1:]
            elif name == "scan_ok":
                def body(c, x):
                    k1, k2 = r.split(c[0])
                    return (k1, c[1] + r.uniform(k2)), None
                out = jax.lax.scan(body, (r.key(12), 0.0), None,
                                   length=2)[0][1:]
            elif name == "cond_branches":
                k = r.key(9)
                out = (jax.lax.cond(True, lambda k: r.normal(k),
                                    lambda k: r.uniform(k), k),)
            else:
                raise KeyError(name)
        return "\n".join(text_of(o) for o in out)
    return run


def signal_thunk(jax, jnp, fn, kind, onesided):
    k = np.arange(16)
    re = jnp.asarray(((k * 7 % 11) / 4 - 1).astype(np.float32))
    im = jnp.asarray(((k * 5 % 13) / 8 - 0.5).astype(np.float32))
    x = jax.lax.complex(re, im) if kind == "complex" else re
    y = jax.lax.complex(im, re) if kind == "complex_y" else im
    sg = jax.scipy.signal

    def run():
        if fn == "welch":
            out = sg.welch(x, nperseg=8, return_onesided=onesided)
        elif fn == "csd":
            out = sg.csd(x, y, nperseg=8, return_onesided=onesided)
        else:
            out = sg.stft(x, nperseg=8, return_onesided=onesided)
        return "\n".join(text_of(o) for o in out)
    return run


def poly_thunk(name):
    from jax import export
    S = export.symbolic_shape

    def run():
        a, b = S("a, b")
        out = {
            "parse": lambda: S("a, b+1, 2*a"),
            "parse_bad": lambda: S("a + "),
            "parse_bad2": lambda: S("a, b)"),
            "ge1": lambda: a >= 1,
            "ge2": lambda: a >= 2,
            "eq": lambda: a == b,
            "eq_self": lambda: a == a,
            "eq_const": lambda: a == 3,
            "lt": lambda: a < b,
            "mul": lambda: a * b - a,
            "pow": lambda: (a + 1) ** 2,
            "divmod": lambda: divmod(a * 4 + 2, 2),
            "floordiv": lambda: a // 2,
            "divmod_sym": lambda: divmod(a * b, b),
            "mod": lambda: a % 3,
            "jmax": lambda: S("max(a, b)"),
            "min3": lambda: S("min(a, 3)"),
            "cons": lambda: S("c", constraints=("c >= 4",))[0] >= 3,
            "cons_bad": lambda: S("c", constraints=("c >= ",)),
            "neg_ge": lambda: (-a) >= 0,
            "sub_ge": lambda: (a - b) >= 0,
            "like": lambda: S("a, _", like=(None, 4)),
            "like_bad": lambda: S("a, 3", like=(None, 4)),
            "gt_const": lambda: (a + 3) > 3,
            "le_self": lambda: a <= a * b,
            "mod_sym": lambda: (a * b + 1) % b,
            "cons_eq": lambda: S("c, d", constraints=("c == d + 2",))[0] >= 3,
        }[name]()
        return str(out)
    return run


def oom_thunk(jax, jnp, kind, log2):
    n = 2 ** log2

    def run():
        x = jnp.float32(1)
        if kind == "zeros":
            out = jax.jit(lambda a: jnp.zeros((n,), jnp.float32))(x)
        else:
            out = jax.jit(lambda a: jnp.broadcast_to(a, (n,)) * 2)(x)
        out.block_until_ready()
        return "allocated"
    return run


def device_thunk(jax, jnp, name):
    def run():
        with jax.default_device(name):
            return text_of(jnp.add(jnp.arange(3.0), 1.0))
    return run


def sdg_thunk(jax, jnp, impl):
    a = jnp.asarray(base("sdg_a", (np.arange(6, dtype=np.float32) / 4).reshape(2, 3), (2, 3)))
    b = jnp.asarray(base("sdg_b", (np.arange(6, dtype=np.float32) - 2).reshape(3, 2), (3, 2)))

    def run():
        kw = {"implementation": impl} if impl != "none" else {}
        return text_of(jax.nn.scaled_dot_general(a, b, (((1,), (0,)), ((), ())), **kw))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    set_base(args.seed)
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    rows = []

    def add(r, **kw):
        r.update(kw)
        if kw.get("group") in ("rank", "solve", "sdg"):
            r.update(BASE)
        rows.append(r)

    for op in RANK_OPS:
        for mode in RANK_MODES:
            add(run_case(jax, rank_thunk(jax, jnp, op, mode)),
                id=f"rank_{op}_{mode}", group="rank", case=op, mode=mode)
    for src, dst in CAST_CASES:
        for fn in CAST_FNS:
            add(run_case(jax, cast_thunk(jax, jnp, src, dst, fn)),
                id=f"cast_{fn}_{src}_{dst}", group="cast", case=fn, src=src,
                dst=dst)
    for dt in ONE_HOT:
        add(run_case(jax, one_hot_thunk(jax, jnp, dt)),
            id=f"one_hot_{dt}", group="one_hot", dtype=dt)
    for fn in SOLVES:
        for kind in SOLVE_SHAPES:
            add(run_case(jax, solve_thunk(jax, jnp, fn, kind)),
                id=f"solve_{fn}_{kind}", group="solve", case=fn, shape=kind)
    for d1, d2 in PROMO_PAIRS:
        for fn in PROMO_FNS:
            add(run_case(jax, promo_thunk(jax, jnp, d1, d2, fn)),
                id=f"promo_{fn}_{d1}_{d2}", group="promo", case=fn, d1=d1,
                d2=d2)
    for mode in X64_MODES:
        for fn, dt in X64_FNS:
            add(run_case(jax, x64_thunk(jax, jnp, mode, fn, dt)),
                id=f"x64mode_{mode}_{fn}_{dt}", group="x64mode", mode=mode,
                case=fn, dtype=dt)
    for mode in LEGACY_MODES:
        for fn in LEGACY_FNS:
            add(run_case(jax, legacy_thunk(jax, jnp, mode, fn)),
                id=f"legacy_{mode}_{fn}", group="legacy", mode=mode, case=fn)
    for fn, kind, onesided in SIGNAL_CASES:
        add(run_case(jax, signal_thunk(jax, jnp, fn, kind, onesided)),
            id=f"signal_{fn}_{kind}_{onesided}", group="signal", case=fn,
            input=kind, onesided=onesided)
    for name in POLY_CASES:
        add(run_case(jax, poly_thunk(name)), id=f"poly_{name}", group="poly",
            case=name)
    for kind, log2 in OOM_CASES:
        add(run_case(jax, oom_thunk(jax, jnp, kind, log2)),
            id=f"oom_{kind}_{log2}", group="oom", case=kind, log2=log2)
    for name in DEVICE_CASES:
        add(run_case(jax, device_thunk(jax, jnp, name)),
            id=f"device_{name.replace(chr(39), '_') or 'empty'}",
            group="device", case=name)
    for impl in SDG_CASES:
        add(run_case(jax, sdg_thunk(jax, jnp, impl)), id=f"sdg_{impl}",
            group="sdg", case=impl)
    for name in REUSE_CASES:
        add(run_case(jax, reuse_thunk(jax, jnp, name)),
            id=f"reuse_{name}", group="reuse", case=name)
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
