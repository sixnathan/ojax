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
REUSE_CASES = ["twice_eager", "twice_jit", "consumed_eager", "consumed_jit",
               "split_once", "split_twice", "split_twice_jit", "bits_fold_in",
               "clone", "off_twice", "while_reuse", "cond_branches",
               "bool_index_jit"]


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


def rank_thunk(jax, jnp, op, mode):
    x = jnp.asarray(np.arange(3, dtype=np.float32) + 1)
    m = jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3) / 2)

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
    b2 = jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3) + 1)
    b1 = jnp.asarray(np.arange(3, dtype=np.float32) + 1)
    bm = jnp.asarray(np.arange(6, dtype=np.float32).reshape(3, 2) + 1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    rows = []

    def add(r, **kw):
        r.update(kw)
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
    for name in REUSE_CASES:
        add(run_case(jax, reuse_thunk(jax, jnp, name)),
            id=f"reuse_{name}", group="reuse", case=name)
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
