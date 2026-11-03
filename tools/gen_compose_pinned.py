import argparse
import json
import os
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import emit_common as ec
import gen_compose_fuzz as cf
import gen_goldens as gg
import jax
import numpy as np

MODULE = "compose_pinned"

PINNED = [
    (40618829, "compose_00346_float64"),
]


def pinned_cases():
    by_seed = {}
    for seed, case_id in PINNED:
        by_seed.setdefault(seed, []).append(case_id)
    found = {name: [] for name, _ in cf.SETS}
    for seed, ids in by_seed.items():
        for set_name, cases in cf.expand(seed).items():
            for case in cases:
                if case["case_id"] in ids:
                    found[set_name].append((seed, case))
    missing = {c for _, c in PINNED} - {c["case_id"] for v in found.values() for _, c in v}
    if missing:
        raise SystemExit("pinned cases not produced by their seeds: " + ", ".join(sorted(missing)))
    for name in found:
        found[name].sort(key=lambda sc: sc[1]["case_id"])
    return found


def write_set(set_name, x64, cases, outdir, sink):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    jax.config.update("jax_enable_x64", x64)
    jax.clear_caches()
    entries, by_op = [], {}
    for seed, case in cases:
        jax.clear_caches()
        entry, saved, row = cf.run_case(case, cf.make_inputs(case, seed))
        entry["seed"] = seed
        entries.append(entry)
        by_op.setdefault(case["op"], {}).update({case["case_id"] + "/" + k: v for k, v in saved.items()})
        row["set"] = MODULE + "/" + set_name
        sink.write(json.dumps(row, sort_keys=True) + "\n")
    for op, bundle in by_op.items():
        np.savez(os.path.join(outdir, "ops", op + ".npz"), **bundle)
    manifest = {"schema_version": 1, "module": MODULE, "jax_version": gg.JAX_VERSION, "x64": x64, "unrunnable": [], "cases": entries}
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
    os.makedirs(os.path.join(base, "text"), exist_ok=True)
    with open(os.path.join(base, "text", "reference.jsonl"), "w", encoding="utf-8") as sink:
        for set_name, x64 in cf.SETS:
            if not found[set_name]:
                shutil.rmtree(os.path.join(base, set_name), ignore_errors=True)
                continue
            count = write_set(set_name, x64, found[set_name], os.path.join(base, set_name), sink)
            print(MODULE + " " + set_name + " " + str(count) + " cases")


if __name__ == "__main__":
    main()
