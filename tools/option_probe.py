import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def rows():
    import jax.numpy as jnp
    import numpy as np

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


def main():
    import jax

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
