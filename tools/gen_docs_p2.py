import argparse
import contextlib
import io
import json
import warnings

import numpy as np

DPRINT_CASES = [
    "kw_pos", "spec_ordered", "no_placeholders", "indexed", "int_spec",
    "escape", "bool_arr", "int_arr", "mat", "scalar_default", "exp_spec",
    "align_spec", "plain_text", "two_kw", "pct_spec", "g_spec", "neg_scalar",
    "weak_scalar", "f64_vals", "nan_inf",
]
DPRINT_EXTRA = [
    "f16_scalar", "bf16_arr", "big_arr", "complex_scalar", "u8_arr",
    "bool_scalar_spec", "repr_conv", "str_conv", "i64_arr", "f16_spec",
    "hex_spec", "center_spec", "zero_pad", "int_as_float_spec", "neg_zero",
    "empty_arr", "two_d_int", "inf_scalar", "i8_arr", "f16_arr",
    "u32_hex", "big_scalar", "tiny_scalar", "tenth_scalar", "printopts",
    "suppress_opts",
]
DPRINT_XFORMS = ["vmap", "grad", "cond", "fori", "scan", "nested_jit",
                 "vmap_ordered", "jvp", "while", "remat", "vmap_unbatched",
                 "grad_scan", "jit_grad", "nested_vmap", "cond_ordered"]
DPRINT_MODES = ["eager", "jit", "jaxpr", "lowered"]
DPRINT_ERRORS = ["missing_kw", "missing_pos", "spec_on_array",
                 "mixed_numbering"]
STATIC_INDEX_CASES = [
    ("pos_oob", "eager"), ("neg_oob", "eager"), ("second_axis", "eager"),
    ("in_bounds", "eager"), ("neg_in", "eager"), ("pos_oob", "jit"),
    ("in_bounds", "jit"), ("at_clip", "eager"), ("at_promise", "eager"),
    ("at_fill", "eager"), ("at_nonorm", "eager"), ("slice_mix", "eager"),
    ("at_set", "eager"), ("int_arr_idx", "eager"),
]
UFUNC_AT_BINARY = ["maximum", "minimum", "logaddexp", "logaddexp2",
                   "bitwise_and", "bitwise_or", "bitwise_xor", "logical_and",
                   "logical_or", "logical_xor"]
UFUNC_AT_UNARY = ["negative"]
UFUNC_AT_IDX = ["scalar", "dup", "two_d", "two_d_row"]
INT_UFUNCS = {"bitwise_and", "bitwise_or", "bitwise_xor"}
API_CASES = ["flat", "like_f32", "like_weak", "like_int", "like_bool",
             "like_scalar_int", "like_jit"]
X64WARN_CASES = [
    ("zeros", "int64"), ("ones", "float64"), ("empty", "float64"),
    ("full", "float64"), ("zeros_like", "float64"), ("ones_like", "int64"),
    ("empty_like", "float64"), ("full_like", "complex128"),
    ("linspace", "float64"), ("logspace", "float64"),
    ("geomspace", "float64"), ("zeros", "uint64"), ("ones", "float32"),
    ("full_like", "int32"),
]
DEFAULT_SEED = 0
BASE = {}

AOT_CASES = ["sin_dot", "add", "reduce", "exp_log", "matmul", "cumsum",
             "where", "int_mul", "two_out", "tanh_big"]


def encode(out):
    a = np.asarray(out)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "hex": a.tobytes().hex()}


def run_case(jax, thunk):
    notices = io.StringIO()
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


def dprint_body(jax, jnp, name):
    p = jax.debug.print

    def body(x):
        i = jnp.arange(3, dtype=jnp.int32)
        if name == "kw_pos":
            p("x={x} y={}", x * 2, x=x)
        elif name == "spec_ordered":
            p("v {:.2f} {}", x[0], x, ordered=True)
        elif name == "no_placeholders":
            p("hello", x)
        elif name == "indexed":
            p("{1} then {0}", x, x + 1)
        elif name == "int_spec":
            p("n={:d} m={:5d}", i[1], i[2])
        elif name == "escape":
            p("{{literal}} {}", x)
        elif name == "bool_arr":
            p("b={}", x > 1)
        elif name == "int_arr":
            p("i={}", i * 7)
        elif name == "mat":
            p("m=\n{m}", m=jnp.outer(x, x) / 3)
        elif name == "scalar_default":
            p("s={}", jnp.sum(x) / 7)
        elif name == "exp_spec":
            p("e={:.3e}", x[2] * 1234.5)
        elif name == "align_spec":
            p("[{:>8.3f}] [{:<6}]", x[1] / 3, i[2])
        elif name == "plain_text":
            p("just text")
        elif name == "two_kw":
            p("{b} {a} {b}", a=x, b=i)
        elif name == "pct_spec":
            p("{:.1%}", x[1] / 4)
        elif name == "g_spec":
            p("{:g} {:.3g}", x[2] / 3, x[2] * 1e7)
        elif name == "neg_scalar":
            p("{} {:+.2f}", -x[1], -x[2])
        elif name == "weak_scalar":
            p("{}", x[0] + 0.5)
        elif name == "f64_vals":
            p("{}", x / 3)
        elif name == "nan_inf":
            p("{}", jnp.log(x - 1))
        elif name == "f16_scalar":
            p("{}", (x[1] / 3).astype(jnp.float16))
        elif name == "bf16_arr":
            p("{}", (x / 3).astype(jnp.bfloat16))
        elif name == "big_arr":
            p("{}", jnp.arange(1500, dtype=jnp.float32) + x[0])
        elif name == "complex_scalar":
            p("{}", jax.lax.complex(x[1], x[2]))
        elif name == "u8_arr":
            p("{}", (i * 100).astype(jnp.uint8))
        elif name == "bool_scalar_spec":
            p("[{:>6}] [{}]", x[1] > 0, x[0] > 0)
        elif name == "repr_conv":
            p("{!r}", x)
        elif name == "str_conv":
            p("{!s} {!r}", x[1], x[2])
        elif name == "i64_arr":
            p("{}", jnp.arange(4, dtype=jnp.int64) * 1000000)
        elif name == "f16_spec":
            p("{:.4f}", (x[1] / 3).astype(jnp.float16))
        elif name == "hex_spec":
            p("{:x} {:#X} {:b} {:o}", i[2] * 100, i[2] * 100, i[2], i[2] * 9)
        elif name == "center_spec":
            p("[{:^9.2f}] [{:*^7}]", x[1] / 3, i[2])
        elif name == "zero_pad":
            p("{:08.3f} {:05d}", -x[1] / 3, i[2])
        elif name == "int_as_float_spec":
            p("{:.2f} {:e}", i[2], i[1])
        elif name == "neg_zero":
            p("{} {:.1f}", -x[0], -x[0])
        elif name == "empty_arr":
            p("{}", x[3:])
        elif name == "two_d_int":
            p("{}", jnp.outer(i, i) - 3)
        elif name == "inf_scalar":
            p("{} {:.2f} {:e}", 1 / x[0], -1 / x[0], jnp.log(x[0] - 1))
        elif name == "i8_arr":
            p("{}", (i - 1).astype(jnp.int8))
        elif name == "f16_arr":
            p("{}", (x / 3).astype(jnp.float16))
        elif name == "u32_hex":
            p("{:08x}", (i[2] * 1000).astype(jnp.uint32))
        elif name == "big_scalar":
            p("{} {}", x[1] * 1e16, x[2] * 1e20)
        elif name == "tiny_scalar":
            p("{} {}", x[1] * 1e-5, x[2] * 1e-30)
        elif name == "tenth_scalar":
            p("{} {}", x[1] / 10, x / 10)
        elif name == "printopts":
            with np.printoptions(precision=3):
                p("{} {}", x / 3, x[1] / 3)
        elif name == "suppress_opts":
            with np.printoptions(suppress=True, threshold=5, edgeitems=2):
                p("{}", jnp.arange(8, dtype=jnp.float32) * 1e-9 + x[1])
        elif name == "missing_kw":
            p("{y}", x)
        elif name == "missing_pos":
            p("{} {}", x)
        elif name == "spec_on_array":
            p("{:.2f}", x)
        elif name == "mixed_numbering":
            p("{} {0}", x, x)
        else:
            raise KeyError(name)
        return x + 1
    return body


def base_x(jnp):
    if "x" in BASE:
        return jnp.asarray(BASE["x"], dtype=jnp.arange(3.0).dtype)
    return jnp.arange(3.0)


def base_m(jnp):
    if "m" in BASE:
        return jnp.asarray(BASE["m"], dtype=jnp.arange(3.0).dtype)
    return jnp.arange(6.0).reshape(2, 3)


def set_base(seed):
    BASE.clear()
    if seed != DEFAULT_SEED:
        rng = np.random.default_rng([seed, 2])
        BASE["x"] = [float(v) / 4 for v in rng.integers(-12, 17, 3)]
        BASE["m"] = [[float(v) / 4 for v in r]
                     for r in rng.integers(-12, 17, (2, 3))]


def with_base(row):
    row.update(BASE)
    return row


def dprint_thunk(jax, jnp, name, mode):
    x = base_x(jnp)
    body = dprint_body(jax, jnp, name)
    if mode == "eager":
        return lambda: (body(x), "")[1]
    if mode == "jit":
        return lambda: (jax.jit(body)(x), "")[1]
    if mode == "jaxpr":
        return lambda: str(jax.make_jaxpr(body)(x))
    if mode == "lowered":
        return lambda: jax.jit(body).lower(x).as_text()
    raise KeyError(mode)


def text_of(out):
    a = np.asarray(out)
    return f"{a.dtype} {list(a.shape)} {a.tobytes().hex()}"


def static_index_thunk(jax, jnp, name, mode, flag):
    x = jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3) + 1)
    v = jnp.asarray(np.arange(3, dtype=np.float32) + 1)
    table = {
        "pos_oob": lambda a: v[5],
        "neg_oob": lambda a: v[-4],
        "second_axis": lambda a: x[1, 7],
        "in_bounds": lambda a: x[1, 2],
        "neg_in": lambda a: v[-3],
        "at_clip": lambda a: v.at[5].get(mode="clip"),
        "at_promise": lambda a: v.at[5].get(mode="promise_in_bounds"),
        "at_fill": lambda a: v.at[5].get(mode="fill"),
        "at_nonorm": lambda a: v.at[-1].get(wrap_negative_indices=False),
        "slice_mix": lambda a: x[3, 1:],
        "at_set": lambda a: v.at[5].set(9.0),
        "int_arr_idx": lambda a: v[jnp.asarray(np.array([0, 4], np.int32))],
    }
    f = table[name]

    def run():
        from jax._src import config as jcfg
        with jcfg.check_static_indices(flag):
            if mode == "jit":
                return text_of(jax.jit(f)(v))
            return text_of(f(v))
    return run


def xform_thunk(jax, jnp, name, mode):
    p = jax.debug.print
    lax = jax.lax
    x = base_x(jnp)
    m = base_m(jnp)

    def run():
        if name == "vmap":
            def f(a):
                p("row {}", a)
                return a * 2
            fn, args = jax.vmap(f), (m,)
        elif name == "vmap_ordered":
            def f(a):
                p("row {} {}", a[0], a, ordered=True)
                return a * 2
            fn, args = jax.vmap(f), (m,)
        elif name == "grad":
            def f(a):
                p("a={}", a)
                return jnp.sum(jnp.sin(a))
            fn, args = jax.grad(f), (x,)
        elif name == "jvp":
            def f(a):
                p("a={}", a)
                return jnp.sin(a)
            fn, args = (lambda a: jax.jvp(f, (a,), (jnp.ones_like(a),))), (x,)
        elif name == "cond":
            def f(a):
                return lax.cond(a[0] < 1, lambda b: (p("yes {}", b), b)[1],
                                lambda b: (p("no {}", b), -b)[1], a)
            fn, args = f, (x,)
        elif name == "fori":
            def f(a):
                def body(k, c):
                    p("k={} c={}", k, c)
                    return c + 1
                return lax.fori_loop(0, 3, body, a)
            fn, args = f, (x,)
        elif name == "scan":
            def f(a):
                def body(c, e):
                    p("c={} e={}", c, e)
                    return c + e, c * e
                return lax.scan(body, a[0], a)
            fn, args = f, (x,)
        elif name == "while":
            def f(a):
                def body(c):
                    p("w {}", c)
                    return c + 1
                return lax.while_loop(lambda c: c[0] < 2, body, a)
            fn, args = f, (x,)
        elif name == "remat":
            def f(a):
                def g(b):
                    p("r {}", b)
                    return jnp.sin(b)
                return jnp.sum(jax.checkpoint(g)(a))
            fn, args = jax.grad(f), (x,)
        elif name == "vmap_unbatched":
            def f(a, b):
                p("{} {}", a, b)
                return a + b
            fn, args = jax.vmap(f, in_axes=(0, None)), (m, x)
        elif name == "grad_scan":
            def f(a):
                def body(c, e):
                    p("c={}", c)
                    return c * e, e
                return lax.scan(body, 1.0, a)[0]
            fn, args = jax.grad(f), (x + 1,)
        elif name == "jit_grad":
            def f(a):
                p("a={}", a)
                return jnp.sum(a * a)
            fn, args = jax.jit(jax.grad(f)), (x,)
        elif name == "nested_vmap":
            def f(a):
                p("e {}", a)
                return a * 2
            fn, args = jax.vmap(jax.vmap(f)), (m,)
        elif name == "cond_ordered":
            def f(a):
                return lax.cond(a[0] < 1,
                                lambda b: (p("yes {}", b, ordered=True), b)[1],
                                lambda b: (p("no {}", b, ordered=True), -b)[1],
                                a)
            fn, args = f, (x,)
        elif name == "nested_jit":
            def f(a):
                def g(b):
                    p("inner {}", b)
                    return b + 1
                p("outer {}", a)
                return jax.jit(g)(a) * 2
            fn, args = f, (x,)
        else:
            raise KeyError(name)
        if mode == "eager":
            fn(*args)
            return ""
        if mode == "jit" and name == "jit_grad":
            fn(*args)
            return ""
        if mode == "jit":
            jax.jit(fn)(*args)
            return ""
        return str(jax.make_jaxpr(fn)(*args))
    return run


def ufunc_at_thunk(jax, jnp, fn, idx_kind, mode):
    integer = fn in INT_UFUNCS
    dt = np.int32 if integer else np.float32
    v = jnp.asarray((np.arange(5) + 1).astype(dt))
    m = jnp.asarray((np.arange(6).reshape(2, 3) + 1).astype(dt))
    i = lambda *xs: jnp.asarray(np.array(xs, np.int32))
    u = getattr(jnp, fn)
    unary = fn in UFUNC_AT_UNARY

    def operands():
        if idx_kind == "scalar":
            return v, 2, jnp.asarray(dt(3))
        if idx_kind == "dup":
            return v, i(0, 2, 2), jnp.asarray(np.array([2, 3, 4], dt))
        if idx_kind == "two_d":
            return m, (i(0, 1, 1), i(2, 0, 0)), jnp.asarray(dt(2))
        return m, i(1, 1), jnp.asarray(np.array([2, 3, 1], dt))

    def call():
        a, idx, b = operands()
        if unary:
            return u.at(a, idx, inplace=False)
        return u.at(a, idx, b, inplace=False)

    def run():
        if mode == "jaxpr":
            a, idx, b = operands()
            idx = idx if isinstance(idx, tuple) else (idx,)
            if unary:
                return str(jax.make_jaxpr(
                    lambda a, *ix: u.at(a, ix, inplace=False))(a, *idx))
            return str(jax.make_jaxpr(
                lambda a, b, *ix: u.at(a, ix, b, inplace=False))(a, b, *idx))
        return text_of(call())
    return run


def api_thunk(jax, jnp, name):
    x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
    table = {
        "flat": lambda: x.flat,
        "like_f32": lambda: repr(jax.ShapeDtypeStruct.like(x)),
        "like_weak": lambda: repr(jax.ShapeDtypeStruct.like(x[0, 0] * 2.0)),
        "like_int": lambda: repr(jax.ShapeDtypeStruct.like(
            jnp.arange(4, dtype=jnp.int32))),
        "like_bool": lambda: repr(jax.ShapeDtypeStruct.like(x > 1)),
        "like_scalar_int": lambda: repr(jax.ShapeDtypeStruct.like(
            jnp.asarray(3))),
        "like_jit": lambda: jax.jit(
            lambda a: (a, repr(jax.ShapeDtypeStruct.like(a)))[0])(x) and "",
    }
    return lambda: str(table[name]()) if name != "flat" else table[name]()


def x64warn_thunk(jax, jnp, fn, dt):
    x = jnp.arange(3.0, dtype=jnp.float32)
    d = np.dtype(dt)
    table = {
        "zeros": lambda: jnp.zeros((2,), dtype=d),
        "ones": lambda: jnp.ones((2,), dtype=d),
        "empty": lambda: jnp.empty((2,), dtype=d),
        "full": lambda: jnp.full((2,), 3.0, dtype=d),
        "zeros_like": lambda: jnp.zeros_like(x, dtype=d),
        "ones_like": lambda: jnp.ones_like(x, dtype=d),
        "empty_like": lambda: jnp.empty_like(x, dtype=d),
        "full_like": lambda: jnp.full_like(x, 2.0, dtype=d),
        "linspace": lambda: jnp.linspace(0.0, 1.0, 5, dtype=d),
        "logspace": lambda: jnp.logspace(0.0, 1.0, 4, dtype=d),
        "geomspace": lambda: jnp.geomspace(1.0, 8.0, 4, dtype=d),
    }
    return lambda: text_of(table[fn]())


def aot_fn(jax, jnp, name):
    x34 = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4) / 7)
    y42 = jnp.asarray(np.arange(8, dtype=np.float32).reshape(4, 2) / 5)
    v = jnp.asarray(np.arange(16, dtype=np.float32) / 3)
    i = jnp.asarray(np.arange(6, dtype=np.int32))
    table = {
        "sin_dot": (lambda a, b: jnp.sin(a) @ b + 1.0, (x34, y42)),
        "add": (lambda a, b: a + b, (v, v)),
        "reduce": (lambda a: jnp.sum(a * a), (v,)),
        "exp_log": (lambda a: jnp.log1p(jnp.exp(a)), (x34,)),
        "matmul": (lambda a, b: a @ b, (x34, y42)),
        "cumsum": (lambda a: jnp.cumsum(a), (v,)),
        "where": (lambda a: jnp.where(a > 2.0, a, -a), (v,)),
        "int_mul": (lambda a: a * a + 3, (i,)),
        "two_out": (lambda a, b: (a + 1.0, b * 2.0), (x34, y42)),
        "tanh_big": (lambda a: jnp.tanh(a) * jnp.cos(a), (v,)),
    }
    return table[name]


def aot_rows(jax, jnp):
    rows = []
    for name in AOT_CASES:
        f, args = aot_fn(jax, jnp, name)
        compiled = jax.jit(f).lower(*args).compile()
        cost = compiled.cost_analysis()
        mem = compiled.memory_analysis()
        mem_d = {k: int(getattr(mem, k)) for k in [
            "generated_code_size_in_bytes", "argument_size_in_bytes",
            "output_size_in_bytes", "alias_size_in_bytes",
            "temp_size_in_bytes", "host_generated_code_size_in_bytes",
            "host_argument_size_in_bytes", "host_output_size_in_bytes",
            "host_alias_size_in_bytes", "host_temp_size_in_bytes",
            "peak_memory_in_bytes"]}
        rows.append({"id": f"aot_cost_{name}", "group": "aot_cost",
                     "case": name, "kind": "ok",
                     "cost": sorted([k, float(v)] for k, v in cost.items())})
        rows.append({"id": f"aot_mem_{name}", "group": "aot_mem",
                     "case": name, "kind": "ok",
                     "mem": sorted([k, v] for k, v in mem_d.items())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    set_base(args.seed)
    rows = []
    for name in DPRINT_CASES:
        for mode in DPRINT_MODES:
            r = run_case(jax, dprint_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_{name}_{mode}", "group": "dprint",
                      "case": name, "mode": mode})
            rows.append(with_base(r))
    for name in DPRINT_EXTRA:
        for mode in ["eager", "jit", "jaxpr"]:
            r = run_case(jax, dprint_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_{name}_{mode}", "group": "dprint",
                      "case": name, "mode": mode})
            rows.append(with_base(r))
    for name in DPRINT_XFORMS:
        for mode in ["eager", "jit", "jaxpr"]:
            r = run_case(jax, xform_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_x_{name}_{mode}", "group": "dprint_x",
                      "case": name, "mode": mode})
            rows.append(with_base(r))
    for name in DPRINT_ERRORS:
        for mode in ["eager", "jit"]:
            if name == "spec_on_array" and mode == "jit":
                continue
            r = run_case(jax, dprint_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_{name}_{mode}", "group": "dprint",
                      "case": name, "mode": mode})
            rows.append(with_base(r))
    for name, mode in STATIC_INDEX_CASES:
        for flag in [True, False]:
            r = run_case(jax, static_index_thunk(jax, jnp, name, mode, flag))
            r.update({"id": f"static_index_{name}_{mode}_{flag}",
                      "group": "static_index", "case": name, "mode": mode,
                      "flag": flag})
            rows.append(r)
    for fn in UFUNC_AT_BINARY + UFUNC_AT_UNARY:
        for idx_kind in UFUNC_AT_IDX:
            for mode in ["value", "jaxpr"]:
                r = run_case(jax, ufunc_at_thunk(jax, jnp, fn, idx_kind, mode))
                r.update({"id": f"ufunc_at_{fn}_{idx_kind}_{mode}",
                          "group": "ufunc_at", "case": fn, "idx": idx_kind,
                          "mode": mode})
                rows.append(r)
    for name in API_CASES:
        if name == "like_jit":
            continue
        r = run_case(jax, api_thunk(jax, jnp, name))
        r.update({"id": f"api_{name}", "group": "api", "case": name})
        rows.append(r)
    for fn, dt in X64WARN_CASES:
        r = run_case(jax, x64warn_thunk(jax, jnp, fn, dt))
        r.update({"id": f"x64warn_{fn}_{dt}", "group": "x64warn",
                  "case": fn, "dtype": dt})
        rows.append(r)
    rows.extend(aot_rows(jax, jnp))
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
