import argparse
import contextlib
import io
import json
import warnings

import numpy as np

CASES = [
    "psum_sub", "psum_in1", "psum_out1", "pmax", "pmin_in1", "psum_int",
    "psum_grad", "psum_jvp", "psum_nested", "pmax_nested", "psum_unmapped",
    "pmax_unmapped", "psum_vjp", "psum_grad_in1", "pmax_grad", "pmin_jvp",
    "io_ordered", "io_ordered_two", "io_ordered_multi", "io_unordered",
    "dprint_ordered_two", "io_ordered_noout", "dprint_then_io",
    "io_then_dprint", "io_nested_jit", "io_nested_twice", "dprint_nested_jit",
    "mixed_io_nested", "io_cond", "dprint_cond", "io_cond_after",
]
MODES = ["eager", "jit", "jaxpr", "lowered"]
DEFAULT_SEED = 0
BASE = {}


def set_base(seed):
    BASE.clear()
    if seed != DEFAULT_SEED:
        rng = np.random.default_rng([seed, 3])
        BASE["x"] = [float(v) / 4 for v in rng.integers(-12, 17, 3)]
        BASE["m"] = [[float(v) / 4 for v in r]
                     for r in rng.integers(-12, 17, (2, 3))]
        BASE["xi"] = [int(v) for v in rng.integers(-9, 10, 4)]


def run_case(jax, thunk):
    notices = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(notices):
                out = thunk()
                jax.effects_barrier()
            result = {"kind": "ok", "text": out}
        except Exception as e:
            result = {"kind": "raise", "class": type(e).__name__,
                      "message": str(e), "text": ""}
    result["warnings"] = [[w.category.__name__, str(w.message)]
                          for w in caught]
    result["stdout"] = notices.getvalue()
    return result


def text_of(outs):
    lines = []
    for out in outs:
        a = np.asarray(out)
        lines.append(f"{a.dtype} {list(a.shape)} {a.tobytes().hex()}")
    return "\n".join(lines)


def case_fn(jax, jnp, name):
    from jax import lax
    from jax.experimental import io_callback
    vmap = jax.vmap

    def sub(v):
        return v - lax.psum(v, "i")

    def double(y):
        return np.asarray(y) * 2

    def io(v, ordered):
        return io_callback(double, jax.ShapeDtypeStruct(v.shape, v.dtype), v,
                           ordered=ordered)

    fdt = jnp.arange(3.0).dtype
    x = jnp.asarray(BASE["x"], dtype=fdt) if "x" in BASE else jnp.arange(3.0)
    m = (jnp.asarray(BASE["m"], dtype=fdt) if "m" in BASE
         else jnp.arange(6.0).reshape(2, 3))
    xi = (jnp.asarray(BASE["xi"], dtype=jnp.arange(4).dtype) if "xi" in BASE
          else jnp.arange(4))
    if name == "psum_sub":
        return (lambda a: (vmap(sub, axis_name="i")(a),)), (x,)
    if name == "psum_in1":
        return (lambda a: (vmap(sub, in_axes=1, axis_name="i")(a),)), (m,)
    if name == "psum_out1":
        return (lambda a: (vmap(sub, in_axes=0, out_axes=1, axis_name="i")(a),)), (m,)
    if name == "pmax":
        return (lambda a: (vmap(lambda v: lax.pmax(v, "i"), axis_name="i")(a),)), (x,)
    if name == "pmin_in1":
        return (lambda a: (vmap(lambda v: lax.pmin(v, "i"), in_axes=1, axis_name="i")(a),)), (m,)
    if name == "psum_int":
        return (lambda a: (vmap(lambda v: lax.psum(v, "i"), axis_name="i")(a),)), (xi,)
    if name == "psum_grad":
        return (lambda a: (jax.grad(lambda b: vmap(sub, axis_name="i")(b).sum())(a),)), (x,)
    if name == "psum_jvp":
        return (lambda a: jax.jvp(vmap(sub, axis_name="i"), (a,), (a * 2,))), (x,)
    if name == "psum_nested":
        return (lambda a: (vmap(vmap(sub, axis_name="i"))(a),)), (m,)
    if name == "pmax_nested":
        return (lambda a: (vmap(vmap(lambda v: lax.pmax(v, "i"), axis_name="i"))(a),)), (m,)
    if name == "psum_unmapped":
        return (lambda a, c: (vmap(lambda v, w: v * lax.psum(w, "i"), in_axes=(0, None), axis_name="i")(a, c),)), (x, m[1])
    if name == "pmax_unmapped":
        return (lambda a, c: (vmap(lambda v, w: v + lax.pmax(w, "i"), in_axes=(0, None), axis_name="i")(a, c),)), (x, m[1])
    if name == "psum_vjp":
        return (lambda a, c: (jax.vjp(vmap(sub, axis_name="i"), a)[1](c)[0],)), (x, x + 1)
    if name == "psum_grad_in1":
        return (lambda a: (jax.grad(lambda b: (vmap(sub, in_axes=1, axis_name="i")(b) ** 2).sum())(a),)), (m,)
    if name == "pmax_grad":
        return (lambda a: (jax.grad(lambda b: vmap(lambda v: lax.pmax(v, "i"), axis_name="i")(b).sum())(a),)), (x,)
    if name == "pmin_jvp":
        return (lambda a: jax.jvp(vmap(lambda v: lax.pmin(v, "i"), axis_name="i"), (a,), (a,))), (x,)
    if name == "io_ordered":
        return (lambda a: (io(a, True),)), (m,)
    if name == "io_ordered_two":
        return (lambda a: (io(io(a, True) + 1, True),)), (x,)
    if name == "io_ordered_multi":
        return (lambda a, b: (io(a, True), io(b, True))), (x, m)
    if name == "io_unordered":
        return (lambda a: (io(a, False) * 3,)), (m,)
    if name == "dprint_ordered_two":
        def f(a):
            jax.debug.print("a {}", a, ordered=True)
            jax.debug.print("b {}", a + 1, ordered=True)
            return (a * 2,)
        return f, (x,)
    if name == "io_ordered_noout":
        def g(a):
            io_callback(lambda y: None, None, a, ordered=True)
            return (a + 1,)
        return g, (x,)
    if name == "dprint_then_io":
        def h(a):
            jax.debug.print("a {}", a, ordered=True)
            return (io(a, True),)
        return h, (x,)
    if name == "io_then_dprint":
        def k(a):
            r = io(a, True)
            jax.debug.print("b {}", r, ordered=True)
            return (r,)
        return k, (x,)
    def inner(v):
        return io(v, True)

    def show(v):
        jax.debug.print("in {}", v, ordered=True)
        return v * 2

    def both(v):
        jax.debug.print("both {}", v, ordered=True)
        return io(v, True)

    if name == "io_nested_jit":
        return (lambda a: (jax.jit(inner)(a) + 1,)), (x,)
    if name == "io_nested_twice":
        jitted = jax.jit(inner)
        return (lambda a: (jitted(jitted(a)),)), (x,)
    if name == "dprint_nested_jit":
        return (lambda a: (jax.jit(show)(a) + 1,)), (x,)
    if name == "mixed_io_nested":
        return (lambda a: (jax.jit(both)(a), io(a, True))), (x,)
    if name == "io_cond":
        return (lambda a: (lax.cond(a[0] < 1, inner, lambda v: v, a),)), (x,)
    if name == "dprint_cond":
        def yes(v):
            jax.debug.print("yes {}", v, ordered=True)
            return v
        return (lambda a: (lax.cond(a[0] < 1, yes, lambda v: -v, a),)), (x,)
    if name == "io_cond_after":
        return (lambda a: (io(lax.cond(a[0] < 1, inner, lambda v: v * 3, a), True),)), (x,)
    raise KeyError(name)


def thunk(jax, jnp, name, mode):
    fn, args = case_fn(jax, jnp, name)
    if mode == "eager":
        return lambda: text_of(fn(*args))
    if mode == "jit":
        return lambda: text_of(jax.jit(fn)(*args))
    if mode == "jaxpr":
        return lambda: str(jax.make_jaxpr(fn)(*args))
    if mode == "lowered":
        return lambda: jax.jit(fn).lower(*args).as_text()
    raise KeyError(mode)


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
    for name in CASES:
        for mode in MODES:
            if ("io" in name.split("_")) and mode == "jaxpr":
                continue
            r = run_case(jax, thunk(jax, jnp, name, mode))
            r.update({"id": f"{name}_{mode}", "case": name, "mode": mode})
            r.update(BASE)
            rows.append(r)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=1, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
