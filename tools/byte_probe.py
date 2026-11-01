import argparse
import os
import subprocess
import sys

N = 100000
SIZES = [3, 8, 32, 64, 128, 256, 512, 1024]
UNARY = [
    ("sin", -100, 100), ("cos", -100, 100), ("tan", -100, 100), ("exp", -20, 20),
    ("expm1", -5, 5), ("log", 1e-3, 1000), ("log1p", -0.99, 1000), ("sqrt", 0, 1000),
    ("cbrt", -1000, 1000), ("sinh", -20, 20), ("cosh", -20, 20), ("tanh", -20, 20),
    ("arcsin", -1, 1), ("arccos", -1, 1), ("arctan", -100, 100), ("arcsinh", -100, 100),
    ("arccosh", 1, 1000), ("arctanh", -0.99, 0.99), ("erf", -5, 5), ("erfc", -5, 5),
    ("gammaln", 0.01, 100), ("digamma", 0.01, 100),
]
BINARY = [("power", 0.01, 10, -5, 5), ("arctan2", -10, 10, -3, 7)]


def reference(out):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as sp

    def fn(name):
        return getattr(sp if name in ("erf", "erfc", "gammaln", "digamma") else jnp, name)

    def grid(lo, hi):
        return np.array([lo + (hi - lo) * i / (N - 1) for i in range(N)], dtype=np.float64)

    def mat(n, dt):
        cells = [[((i * 7 + j * 13) % 101) / 101.0 - 0.5 for j in range(n)] for i in range(n)]
        return np.array(cells, dtype=np.float64).astype(dt)

    def save(stem, result, *inputs):
        with open(os.path.join(out, stem + ".bin"), "wb") as sink:
            sink.write(np.asarray(result).tobytes())
        with open(os.path.join(out, stem + ".in"), "wb") as sink:
            for x in inputs:
                sink.write(x.tobytes())

    for x64, dt, tag in ((False, np.float32, "f32"), (True, np.float64, "f64")):
        jax.config.update("jax_enable_x64", x64)
        for name, lo, hi in UNARY:
            x = grid(lo, hi).astype(dt)
            save(f"{tag}_{name}", jax.jit(fn(name))(jnp.asarray(x)), x)
        for name, lo, hi, lo2, hi2 in BINARY:
            x = grid(lo, hi).astype(dt)
            y = grid(lo2, hi2)[::-1].astype(dt)
            save(f"{tag}_{name}", jax.jit(fn(name))(jnp.asarray(x), jnp.asarray(y)), x, y)
        for n in SIZES:
            a = mat(n, dt)
            b = a[::-1].copy()
            save(f"{tag}_matmul{n}", jax.jit(jnp.matmul)(jnp.asarray(a), jnp.asarray(b)), a, b)
    print("jax", jax.__version__, jax.devices()[0].client.platform_version)


def program_names():
    names = [n for n, _, _ in UNARY] + [b[0] for b in BINARY] + [f"matmul{n}" for n in SIZES]
    return [(tag, name) for tag in ("f32", "f64") for name in names]


def compare(reference_dir, ojax_dir):
    bad = 0
    for tag, name in program_names():
        stem = f"{tag}_{name}.bin"
        with open(os.path.join(reference_dir, stem), "rb") as fh:
            want = fh.read()
        with open(os.path.join(ojax_dir, stem), "rb") as fh:
            got = fh.read()
        width = 4 if tag == "f32" else 8
        count = len(want) // width
        differing = len(want) != len(got)
        if not differing:
            differing = sum(
                want[i : i + width] != got[i : i + width] for i in range(0, len(want), width)
            )
        verdict = "IDENTICAL" if differing == 0 else "DIFFERENT"
        print(f"{tag} {name:<10} elements={count} differing={int(differing)} {verdict}")
        bad += differing != 0
    print(f"{len(program_names()) - bad} of {len(program_names())} programs byte-identical")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(
        description="run 64 programs through JAX and ojax and compare their output bytes"
    )
    parser.add_argument("step", choices=["all", "reference", "ojax", "compare"])
    parser.add_argument("--dir", required=True, help="directory for program inputs and outputs")
    parser.add_argument("--exe", help="built tools/byte_probe/byte_probe.exe")
    args = parser.parse_args()
    reference_dir = os.path.join(args.dir, "jax")
    ojax_dir = os.path.join(args.dir, "ojax")
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(ojax_dir, exist_ok=True)
    if args.step in ("all", "reference"):
        reference(reference_dir)
    if args.step in ("all", "ojax"):
        subprocess.run([args.exe, reference_dir, ojax_dir], check=True)
    if args.step in ("all", "compare"):
        sys.exit(compare(reference_dir, ojax_dir))


if __name__ == "__main__":
    main()
