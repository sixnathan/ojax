import argparse
import os
import subprocess
import sys

HISTORY_PROBE = r"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if sys.argv[1] == "after_slice":
    jax.lax.slice(jnp.zeros((8, 2)), (2, 0), (5, 2))
try:
    jax.lax.dynamic_slice_in_dim(jnp.zeros((0, 2)), 2, 3, 0)
except Exception as error:
    print(type(error).__name__ + ": " + str(error))
"""

PROOFS = {
    "lax_dynamic_slice_in_dim_6x2_902a1355_float64_shape_small": (HISTORY_PROBE, ("alone", "after_slice")),
}


def prove(python):
    failures = 0
    for case, (probe, modes) in PROOFS.items():
        texts = {}
        for mode in modes:
            out = subprocess.run([python, "-I", "-c", probe, mode], capture_output=True, text=True, cwd="/", check=False)
            texts[mode] = out.stdout.strip()
            print(case + "\t" + mode + "\t" + texts[mode])
        if len(set(texts.values())) < 2:
            failures += 1
            print(case + "\tnot reproduced")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prove"])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    sys.exit(1 if prove(os.path.abspath(args.python)) else 0)


if __name__ == "__main__":
    main()
