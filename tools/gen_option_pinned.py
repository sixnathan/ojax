import argparse
import json
import os
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import emit_common as ec
import gen_goldens as gg
import gen_option_fuzz as of
import jax
import numpy as np

MODULE = "option_pinned"

PINNED = [
    (71204459, "lax.cumlogsumexp_08f92a0a_float64"),
    (71204459, "lax.cumlogsumexp_70cf273a_float64"),
    (71204459, "jnp.corrcoef_be522b5f_float32"),
    (71204459, "jnp.setxor1d_6b174152_float32"),
    (71204459, "jnp.unique_43b835c6_float32"),
    (71204459, "jnp.unique_246731e3_float64"),
    (71204459, "lax_linalg.eigh_aaee3655_float64"),
    (71204459, "lax_linalg.schur_8e3527b8_float64"),
    (71204459, "sp_special.log_softmax_956c2041_float32"),
    (71204459, "sp_special.softmax_956c2041_float32"),
    (71204459, "jnp.tensordot_71574931_float64"),
    (71204459, "jnp.tensordot_e0c74ffd_float64"),
    (71204459, "jnp.vecdot_68a9423f_float64"),
    (71204459, "linalg.vecdot_aeb6496a_float64"),
    (66338877, "sp_special.sph_harm_y_a6f05c88_float32"),
    (66338877, "jnp.tensordot_b5814664_float64"),
    (80917353, "jnp.full_bf38151c_float32"),
    (80917353, "jnp.nanargmax_33b72a61_float64"),
    (80917353, "lax_linalg.eigh_f3542e85_float64"),
    (80917353, "lax.conv_general_dilated_patches_eb394909_float64"),
    (80917353, "lax.conv_general_dilated_patches_60254602_float64"),
    (66338877, "lax.conv_general_dilated_patches_c26d4119_float64"),
]


def pinned_cases():
    by_seed = {}
    for seed, case_id in PINNED:
        by_seed.setdefault(seed, []).append(case_id)
    found = {name: [] for name, _ in of.SETS}
    for seed, ids in by_seed.items():
        expanded = of.expand(seed)
        for set_name, cases in expanded.items():
            for case in cases:
                if case["case_id"] in ids:
                    found[set_name].append((seed, case))
    missing = {c for _, c in PINNED} - {c["case_id"] for v in found.values() for _, c in v}
    if missing:
        raise SystemExit("pinned cases not produced by their seeds: " + ", ".join(sorted(missing)))
    for name in found:
        found[name].sort(key=lambda sc: sc[1]["case_id"])
    return found


def write_set(set_name, x64, cases, outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    jax.config.update("jax_enable_x64", x64)
    jax.clear_caches()
    entries, by_op = [], {}
    for seed, case in cases:
        arrays = of.make_inputs(case, seed)
        entry, saved = of.run_case(case, arrays)
        entry["seed"] = seed
        entries.append(entry)
        by_op.setdefault(case["op"], {}).update({case["case_id"] + "/" + k: v for k, v in saved.items()})
    for op, bundle in by_op.items():
        np.savez(os.path.join(outdir, "ops", op + ".npz"), **bundle)
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
    return len(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    found = pinned_cases()
    for set_name, x64 in of.SETS:
        count = write_set(set_name, x64, found[set_name], os.path.join(base, set_name))
        print(MODULE + " " + set_name + " " + str(count) + " cases")


if __name__ == "__main__":
    main()
