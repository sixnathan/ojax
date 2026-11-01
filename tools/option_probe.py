import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def rows():
    import jax.numpy as jnp
    import numpy as np
    from jax._src.lax import linalg as lax_linalg

    import jax

    def _debug_callback(x):
        jax.debug.callback(lambda v: None, x)
        return x + 1.0

    a9 = np.array([3.0, 1.0, 2.0, 3.0, 0.0, -1.0, 1.0, 5.0, 0.0], np.float32)
    b6 = np.array([1.0, 7.0, 2.0, 9.0, 0.0, 3.0], np.float32)
    m23 = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], np.float32)
    ints8 = np.array([1, 3, 3, 0, 7, 2, 1, 1], np.int32)
    cond = a9 > 0
    return [
        ("nonzero_size", lambda x: jnp.nonzero(x, size=4), [m23]),
        ("nonzero_size_fill", lambda x: jnp.nonzero(x, size=5, fill_value=7), [m23]),
        ("flatnonzero_size", lambda x: jnp.flatnonzero(x, size=3), [a9]),
        ("argwhere_size", lambda x: jnp.argwhere(x, size=4), [m23]),
        ("unique_size", lambda x: jnp.unique(x, size=4), [a9]),
        ("unique_size_fill", lambda x: jnp.unique(x, size=8, fill_value=-9.0), [a9]),
        ("unique_counts_size", lambda x: tuple(jnp.unique_counts(x, size=5)), [a9]),
        ("unique_inverse_size", lambda x: tuple(jnp.unique_inverse(x, size=5)), [a9]),
        ("unique_all_size", lambda x: tuple(jnp.unique_all(x, size=5)), [a9]),
        ("union1d_size", lambda x, y: jnp.union1d(x, y, size=8), [a9, b6]),
        ("union1d_size_fill", lambda x, y: jnp.union1d(x, y, size=12, fill_value=-1.0), [a9, b6]),
        ("intersect1d_size", lambda x, y: jnp.intersect1d(x, y, size=3), [a9, b6]),
        ("intersect1d_size_fill", lambda x, y: jnp.intersect1d(x, y, size=6, fill_value=-1.0), [a9, b6]),
        ("setdiff1d_size", lambda x, y: jnp.setdiff1d(x, y, size=3), [a9, b6]),
        ("setdiff1d_size_fill", lambda x, y: jnp.setdiff1d(x, y, size=6, fill_value=-1.0), [a9, b6]),
        ("setxor1d_size", lambda x, y: jnp.setxor1d(x, y, size=4), [a9, b6]),
        ("setxor1d_size_fill", lambda x, y: jnp.setxor1d(x, y, size=9, fill_value=-1.0), [a9, b6]),
        ("compress_size", lambda c, x: jnp.compress(c, x, size=4, fill_value=-5.0), [cond, a9]),
        ("extract_size", lambda c, x: jnp.extract(c, x, size=4), [cond, a9]),
        ("bincount_length", lambda x: jnp.bincount(x, length=5), [ints8]),
        ("bincount_weights", lambda x, w: jnp.bincount(x, weights=w, length=9), [ints8, np.arange(8.0, dtype=np.float32)]),
        ("roots_nostrip", lambda x: jnp.roots(x, strip_zeros=False), [np.array([1.0, -3.0, 2.0], np.float32)]),
        ("sort_complex", lambda x: jnp.sort_complex(x), [np.array([1 + 2j, 1 - 1j, complex(np.nan, 0), -3 + 4j, complex(np.inf, 0)], np.complex64)]),
        ("hessenberg_1", lambda x: tuple(lax_linalg.hessenberg(x)), [np.array([[3.0]], np.float32)]),
        (
            "pure_callback",
            lambda x: jax.pure_callback(lambda v: np.asarray(v) + 1.0, jax.ShapeDtypeStruct(x.shape, x.dtype), x) * 2.0,
            [np.array([1.0, 2.0, 3.0], np.float32)],
        ),
        ("debug_callback", _debug_callback, [np.array([1.0, 2.0, 3.0], np.float32)]),
        ("frexp_edges", lambda x: tuple(jnp.frexp(x)), [np.array([0.0, -0.0, 1.5, -3.0, np.inf, -np.inf, np.nan, 1e-40], np.float32)]),
    ]


def as_bits(v):
    import numpy as np

    arr = np.asarray(v)
    row = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
    if np.iscomplexobj(arr):
        flat = arr.ravel()
        row["bits"] = [str(int(np.float64(p).view(np.int64))) for c in flat for p in (c.real, c.imag)]
    else:
        row["bits"] = [str(int(np.float64(x).view(np.int64))) for x in arr.ravel()]
    return row


def complex_edges(probe, out):
    import jax
    import jax.numpy as jnp
    import numpy as np

    subprocess.run([probe, "complex", out], check=True)
    ours = {}
    for row in map(json.loads, open(out, encoding="utf-8")):
        ours[row["key"]] = row
    specials = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e30])
    unary = dict(
        negative=jnp.negative, absolute=jnp.abs, sign=jnp.sign, sqrt=jnp.sqrt, exp=jnp.exp, log=jnp.log,
        log2=jnp.log2, log10=jnp.log10, log1p=jnp.log1p, expm1=jnp.expm1, sin=jnp.sin, cos=jnp.cos,
        tan=jnp.tan, sinh=jnp.sinh, cosh=jnp.cosh, tanh=jnp.tanh, arcsin=jnp.arcsin, arccos=jnp.arccos,
        arctan=jnp.arctan, arcsinh=jnp.arcsinh, arccosh=jnp.arccosh, arctanh=jnp.arctanh, square=jnp.square,
        conj=jnp.conj, real=jnp.real, imag=jnp.imag, angle=jnp.angle, isnan=jnp.isnan, isinf=jnp.isinf,
        isfinite=jnp.isfinite, reciprocal=jnp.reciprocal,
    )
    binary = dict(add=jnp.add, subtract=jnp.subtract, multiply=jnp.multiply, divide=jnp.true_divide, power=jnp.power)
    total = same = 0
    bad = {}
    for dtype, label, x64 in ((np.complex64, "c64", False), (np.complex128, "c128", True)):
        jax.config.update("jax_enable_x64", x64)
        grid = np.array([complex(a, b) for a in specials for b in specials], dtype)
        n = grid.size
        left = np.repeat(grid, n)
        right = np.tile(grid, n)
        for set_name, table, args in (("u", unary, (grid,)), ("b", binary, (left, right))):
            for name, fn in table.items():
                for mode, run in (("eager", fn), ("jit", jax.jit(fn))):
                    key = "/".join((set_name, name, mode, label))
                    ref = as_bits(run(*args))
                    mine = ours.get(key, {})
                    total += 1
                    if mine.get("row") == ref:
                        same += 1
                    else:
                        count = None
                        if "row" in mine and len(mine["row"]["bits"]) == len(ref["bits"]):
                            count = sum(a != b for a, b in zip(mine["row"]["bits"], ref["bits"]))
                        bad[key] = count if count is not None else mine.get("error", "shape or dtype")[:120] if "row" not in mine else (mine["row"]["dtype"], ref["dtype"])
    jax.config.update("jax_enable_x64", False)
    for key, count in bad.items():
        print(key, count)
    print("complex edges:", same, "of", total, "identical")


def main():
    import jax

    if sys.argv[1] == "complex":
        complex_edges(sys.argv[2], sys.argv[3])
        return
    probe, out = sys.argv[1], sys.argv[2]
    subprocess.run([probe, out], check=True)
    ours = {row["case"]: row for row in map(json.loads, open(out, encoding="utf-8"))}
    totals = {"text": 0, "jit": 0, "eager": 0}
    cases = rows()
    for name, fn, args in cases:
        fn.__name__ = "f"
        mine = ours[name]
        jitted = jax.jit(fn)
        ref_text = jitted.lower(*args).as_text()
        ref_jit = [as_bits(v) for v in jax.tree_util.tree_leaves(jitted(*args))]
        ref_eager = [as_bits(v) for v in jax.tree_util.tree_leaves(fn(*args))]
        t = mine.get("text") == ref_text
        j = mine.get("jit") == ref_jit
        e = mine.get("eager") == ref_eager
        totals["text"] += t
        totals["jit"] += j
        totals["eager"] += e
        print(name, "text" if t else "TEXT_DIFF", "jit" if j else "JIT_DIFF", "eager" if e else "EAGER_DIFF")
        if not t and "text" in mine:
            for a, b in zip(ref_text.split("\n"), mine["text"].split("\n")):
                if a != b:
                    print("   JAX ", a[:170])
                    print("   OJAX", b[:170])
                    break
        for key in ("text_error", "jit_error", "eager_error"):
            if key in mine:
                print("  ", key, mine[key][:240])
        if not j and "jit" in mine:
            print("   jit JAX ", json.dumps(ref_jit)[:300])
            print("   jit OJAX", json.dumps(mine["jit"])[:300])
    print(totals, "of", len(cases))


if __name__ == "__main__":
    main()
