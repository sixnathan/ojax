import json
import os
import subprocess
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def reference():
    import jax
    import jax.numpy as jnp
    import numpy as np

    la = jnp.linalg
    sl = jax.scipy.linalg
    rows = {}

    def values(result):
        return [
            {"dtype": str(np.asarray(l).dtype), "shape": list(np.shape(l)), "hex": np.asarray(l).tobytes().hex()}
            for l in jax.tree_util.tree_leaves(result)
        ]

    def run(name, fn, args):
        f = lambda *xs: fn(*xs)
        f.__name__ = "f"
        row = {}
        try:
            row["text"] = jax.jit(f).lower(*args).as_text()
            row["jit"] = values(jax.jit(f)(*args))
        except Exception as e:
            row["text"] = "ERR " + str(e)
            row["jit"] = "ERR"
        try:
            row["eager"] = values(fn(*args))
        except Exception as e:
            row["eager"] = "ERR " + str(e)
        rows[name] = row

    for suffix, dt in (("_f32", np.float32), ("_c64", np.complex64)):
        for name, fn, shape in [
            ("qr_0x3", la.qr, (0, 3)),
            ("qr_3x0", la.qr, (3, 0)),
            ("qr_0x3_complete", lambda x: la.qr(x, mode="complete"), (0, 3)),
            ("qr_3x0_complete", lambda x: la.qr(x, mode="complete"), (3, 0)),
            ("qr_b2_0x3", la.qr, (2, 0, 3)),
            ("lu_0x0", sl.lu, (0, 0)),
            ("lu_0x3", sl.lu, (0, 3)),
            ("lu_3x0", sl.lu, (3, 0)),
            ("svd_0x3", la.svd, (0, 3)),
            ("svd_3x0", la.svd, (3, 0)),
            ("svd_0x0_reduced", lambda x: la.svd(x, full_matrices=False), (0, 0)),
            ("svd_0x3_reduced", lambda x: la.svd(x, full_matrices=False), (0, 3)),
            ("svd_0x3_novec", lambda x: la.svd(x, compute_uv=False), (0, 3)),
            ("eigh_0x0", la.eigh, (0, 0)),
            ("eig_0x0", la.eig, (0, 0)),
            ("schur_0x0", sl.schur, (0, 0)),
            ("hessenberg_0x0", sl.hessenberg, (0, 0)),
            ("hessenberg_q_0x0", lambda x: sl.hessenberg(x, calc_q=True), (0, 0)),
        ]:
            run(name + suffix, fn, [np.zeros(shape, dt)])
    m = np.array([[2.0, 1.0], [1.0, 3.0]], np.float32)
    for name, fn in [
        ("named_qr", la.qr),
        ("named_svd", la.svd),
        ("named_eigh", la.eigh),
        ("named_eig", la.eig),
        ("named_slogdet", la.slogdet),
    ]:
        run(name, fn, [m])
    m3 = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 7.0]], np.float32)
    m2 = np.array([[1.0, 4.0, 2.0], [5.0, 3.0, 7.0]], np.float32)
    c = np.array([[1 + 1j, 2], [1j, 3 - 1j]], np.complex64)
    run("lu_2x2", sl.lu, [m])
    run("lu_3x2", sl.lu, [m3])
    run("lu_2x3", sl.lu, [m2])
    run("lu_c2x2", sl.lu, [c])
    run("lu_permute_l", lambda x: sl.lu(x, permute_l=True), [m])
    run("slogdet_c2x2", la.slogdet, [c])
    run("slogdet_1x1", la.slogdet, [np.array([[-2.0]], np.float32)])
    return rows


def main():
    probe, out = sys.argv[1], sys.argv[2]
    subprocess.run([probe, out], check=True)
    ours = {r["case"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
    rows = reference()
    verbose = len(sys.argv) > 3
    totals = {"text": 0, "jit": 0, "eager": 0}
    for name, ref in rows.items():
        mine = ours.get(name, {})
        status = []
        for k in totals:
            same = mine.get(k) == ref[k]
            totals[k] += same
            status.append(f"{k}={'ok' if same else 'DIFF'}")
        print(name, *status)
        if mine.get("text") != ref["text"]:
            a, b = ref["text"].split("\n"), str(mine.get("text")).split("\n")
            for x, y in zip(a, b):
                if x != y:
                    print("  JAX ", x[:220])
                    print("  OJAX", y[:220])
                    break
        for k in ("jit", "eager"):
            if mine.get(k) != ref[k] and isinstance(mine.get(k), str):
                print(" ", k, mine[k][:160])
            elif mine.get(k) != ref[k]:
                print(" ", k, "JAX", str(ref[k])[:200], "OJAX", str(mine.get(k))[:200])
    print(totals, "of", len(rows))


if __name__ == "__main__":
    main()
