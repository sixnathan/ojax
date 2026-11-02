import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
SPECIALS = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 1e30]
UNARY = """sin cos tan sinh cosh tanh arcsin arccos arctan arcsinh arccosh arctanh exp expm1 log log1p log2 log10
sqrt square sign abs conj real imag negative positive reciprocal angle exp2 isnan isinf isfinite""".split()
BINARY = "add subtract multiply divide power equal not_equal".split()


def reference():
    import jax
    import jax.numpy as jnp
    import numpy as np

    rows = {}

    def values(r):
        r = np.asarray(r)
        return [{"dtype": str(r.dtype), "hex": r.tobytes().hex()}]

    grid = [complex(re, im) for re in SPECIALS for im in SPECIALS]
    for tag, dt, x64 in (("c64", np.complex64, False), ("c128", np.complex128, True)):
        jax.config.update("jax_enable_x64", x64)
        n = len(grid)
        arr = np.empty(n, dt)
        for i, c in enumerate(grid):
            arr[i] = c
        x = np.array(arr)
        y = np.array([arr[(i * 37) % n] for i in range(n)], dt)
        for name in UNARY:
            f = getattr(jnp, name)
            rows[f"{name}_{tag}_eager"] = values(f(x))
            rows[f"{name}_{tag}_jit"] = values(jax.jit(f)(x))
        for name in BINARY:
            f = getattr(jnp, name)
            rows[f"{name}_{tag}_eager"] = values(f(x, y))
            rows[f"{name}_{tag}_jit"] = values(jax.jit(f)(x, y))
    jax.config.update("jax_enable_x64", True)
    nan = float("nan")
    for name, f, x in [
        ("cholesky_bad", jnp.linalg.cholesky, np.array([[-1.0, 0.0], [0.0, 1.0]])),
        ("inv_singular", jnp.linalg.inv, np.zeros((2, 2))),
        ("nanmean_all", jnp.nanmean, np.array([nan, nan])),
        ("nanmax_all", jnp.nanmax, np.array([nan, nan])),
        ("mean_empty", jnp.mean, np.zeros((0,))),
        ("erfinv_two", jax.scipy.special.erfinv, np.array([2.0])),
        ("arcsin_two", jnp.arcsin, np.array([2.0])),
        ("log_neg", jnp.log, np.array([-2.0])),
        ("ndtri_bad", jax.scipy.special.ndtri, np.array([2.0])),
    ]:
        rows[name + "_eager"] = values(f(x))
        rows[name + "_jit"] = values(jax.jit(f)(x))
    return rows


def main():
    subprocess.run([sys.argv[1], sys.argv[2]], check=True)
    ours = {r["case"]: r["value"] for r in map(json.loads, open(sys.argv[2], encoding="utf-8"))}
    rows = reference()
    same = 0
    for name, ref in rows.items():
        mine = ours.get(name)
        if mine == ref:
            same += 1
            continue
        if isinstance(mine, list) and mine and mine[0]["dtype"] == ref[0]["dtype"]:
            w = 8 if "c64" in name else 16
            a, b = ref[0]["hex"], mine[0]["hex"]
            if ref[0]["dtype"] == "bool":
                w = 2
            bad = [i // w for i in range(0, len(a), w) if a[i : i + w] != b[i : i + w]]
            first = bad[0] * w if bad else 0
            print(name, "elements differing", len(bad), "first", bad[:1], a[first : first + w], b[first : first + w])
        else:
            print(name, "JAX", str(ref)[:60], "OJAX", str(mine)[:200])
    print("identical", same, "of", len(rows))


if __name__ == "__main__":
    main()
