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
DPRINT_MODES = ["eager", "jit", "jaxpr", "lowered"]
DPRINT_ERRORS = ["missing_kw", "missing_pos", "spec_on_array",
                 "mixed_numbering"]
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


def dprint_thunk(jax, jnp, name, mode):
    x = jnp.arange(3.0)
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
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp
    rows = []
    for name in DPRINT_CASES:
        for mode in DPRINT_MODES:
            r = run_case(jax, dprint_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_{name}_{mode}", "group": "dprint",
                      "case": name, "mode": mode})
            rows.append(r)
    for name in DPRINT_ERRORS:
        for mode in ["eager", "jit"]:
            if name == "spec_on_array" and mode == "jit":
                continue
            r = run_case(jax, dprint_thunk(jax, jnp, name, mode))
            r.update({"id": f"dprint_{name}_{mode}", "group": "dprint",
                      "case": name, "mode": mode})
            rows.append(r)
    rows.extend(aot_rows(jax, jnp))
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
