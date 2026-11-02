import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import shutil

import emit_common as ec
import gen_goldens as gg
import gen_option_fuzz as of
import jax
import numpy as np

MODULE = "put_modes"

INDICES = {
    "mixed": np.array([-9, -1, 0, 5, 7, 12], dtype=np.int32),
    "high": np.array([6, 11, 100], dtype=np.int32),
    "low": np.array([-7, -12, -100], dtype=np.int32),
    "grid": np.array([[7, -7], [3, 100]], dtype=np.int32),
    "repeat": np.array([8, 8, -8, 2], dtype=np.int32),
}

VALUES = {"full": None, "two": 2, "one": 1}

MODES = [None, "clip", "wrap", "raise", "bogus"]

DEFAULT_SEED = 20261104


def build(set_name, dtype, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    cases = []
    for ind_name, ind in INDICES.items():
        for val_name, count in VALUES.items():
            for mode in MODES:
                if mode in ("raise", "bogus") and (val_name != "full" or ind_name != "mixed"):
                    continue
                n = ind.size if count is None else count
                target_dtype = "int32" if val_name == "one" else dtype
                arrays = {
                    "arg0": (rng.standard_normal((2, 3)) * 4).astype(target_dtype),
                    "arg1": ind,
                    "arg2": (rng.standard_normal((n,)) * 4).astype(dtype),
                }
                options = {"inplace": False}
                if mode is not None:
                    options["mode"] = mode
                case_id = "jnp.put_%s_%s_%s_%s" % (ind_name, val_name, mode or "none", dtype)
                case = {
                    "case_id": case_id,
                    "op": "jnp.put",
                    "dtype": dtype,
                    "variant": "option",
                    "module": "jnp",
                    "name": "put",
                    "positional": [{"$arr": "arg0"}, {"$arr": "arg1"}, {"$arr": "arg2"}],
                    "options": options,
                    "inputs": {},
                }
                cases.append((case, arrays))
    return cases


def write(root, seed=DEFAULT_SEED):
    for set_name, x64 in of.SETS:
        outdir = os.path.join(root, MODULE, set_name)
        dtypes = ["float32"] if not x64 else ["float32", "float64"]
        cases = [c for d in dtypes for c in build(set_name, d, seed)]
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.makedirs(os.path.join(outdir, "ops"))
        jax.config.update("jax_enable_x64", x64)
        jax.clear_caches()
        entries, bundle = [], {}
        for case, arrays in cases:
            entry, saved = of.run_case(case, arrays)
            entries.append(entry)
            bundle.update({case["case_id"] + "/" + k: v for k, v in saved.items()})
        np.savez(os.path.join(outdir, "ops", "jnp.put.npz"), **bundle)
        manifest = {
            "schema_version": 1,
            "module": MODULE,
            "jax_version": gg.JAX_VERSION,
            "x64": x64,
            "unrunnable": [],
            "cases": entries,
        }
        with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
            fh.write(ec.canonical_dumps(manifest))
        gg.write_sha256sums(outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write(args.root, args.seed)


if __name__ == "__main__":
    main()
