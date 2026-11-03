import argparse
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import emit_common as ec
import gen_compose_fuzz as cf
import gen_goldens as gg
import numpy as np

MODULE = "compose_pinned"
COMPOSE_EXPAND = cf.expand
COMPOSE_APPLY = cf.apply_layer

import gen_custom_ad as cad

CUSTOM_APPLY = cad.apply_layer
cf.expand = COMPOSE_EXPAND
cf.apply_layer = COMPOSE_APPLY
cf.MODULE = MODULE

PINNED = [
    ("compose", 93518027, "compose_00052_float32"),
    ("compose", 93518027, "compose_00308_float32"),
    ("compose", 93518027, "compose_00311_float32"),
    ("compose", 93518027, "compose_00515_float32"),
    ("compose", 93518027, "compose_00569_float32"),
    ("compose", 93518027, "compose_00716_float32"),
    ("compose", 93518027, "compose_00161_float64"),
    ("compose", 93518027, "compose_00171_float64"),
    ("compose", 93518027, "compose_00196_float64"),
    ("compose", 93518027, "compose_00387_float64"),
    ("compose", 40618829, "compose_00028_float32"),
    ("compose", 40618829, "compose_00448_float32"),
    ("compose", 40618829, "compose_00739_float32"),
    ("compose", 40618829, "compose_00759_float32"),
    ("compose", 40618829, "compose_00860_float32"),
    ("compose", 40618829, "compose_00980_float32"),
    ("compose", 40618829, "compose_00373_float64"),
    ("compose", 40618829, "compose_00670_float64"),
    ("compose", 40618829, "compose_00915_float64"),
    ("custom", 93518027, "custom_00026_float32"),
    ("custom", 93518027, "custom_00170_float64"),
    ("custom", 93518027, "custom_00232_float64"),
    ("custom", 93518027, "custom_00264_float64"),
    ("custom", 93518027, "custom_00284_float64"),
    ("custom", 40618829, "custom_00076_float64"),
    ("custom", 40618829, "custom_00094_float64"),
    ("custom", 40618829, "custom_00111_float64"),
]


def expand(kind, seed):
    return cad.expand(seed) if kind == "custom" else COMPOSE_EXPAND(seed)


def pinned_cases():
    wanted = {}
    for kind, seed, case_id in PINNED:
        wanted.setdefault((kind, seed), set()).add(case_id)
    found = {name: [] for name, _ in cf.SETS}
    for (kind, seed), ids in sorted(wanted.items()):
        for set_name, cases in expand(kind, seed).items():
            for case in cases:
                if case["case_id"] in ids:
                    pinned = dict(case, case_id="s" + str(seed) + "_" + case["case_id"])
                    found[set_name].append((kind, seed, pinned))
    produced = {(k, s, c["case_id"]) for v in found.values() for k, s, c in v}
    missing = [c for k, s, c in PINNED if (k, s, "s" + str(s) + "_" + c) not in produced]
    if missing:
        raise SystemExit("pinned cases not produced by their seeds: " + ", ".join(missing))
    for name in found:
        found[name].sort(key=lambda ksc: ksc[2]["case_id"])
    return found


def worker(kind, seed, set_name, case_json, out):
    import jax

    jax.config.update("jax_enable_x64", set_name == "x64_on")
    cf.apply_layer = CUSTOM_APPLY if kind == "custom" else COMPOSE_APPLY
    case = json.loads(case_json)
    source = dict(case, case_id=case["case_id"].split("_", 1)[1])
    arrays = cf.make_inputs(source, seed)
    entry, saved, row = cf.run_case(case, arrays)
    entry["seed"] = seed
    entry["generator"] = kind
    np.savez(out + ".npz", **saved)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"entry": entry, "row": row}, fh, sort_keys=True)


def write_set(set_name, x64, cases, outdir, work, sink):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    entries, bundle = [], {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    for kind, seed, case in cases:
        out = os.path.join(work, set_name + "_" + case["case_id"])
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", kind, str(seed), set_name, json.dumps(case), out], env=env, check=True)
        with open(out + ".json", encoding="utf-8") as fh:
            stored = json.load(fh)
        entries.append(stored["entry"])
        row = stored["row"]
        row["set"] = MODULE + "/" + set_name
        sink.write(json.dumps(row, sort_keys=True) + "\n")
        with np.load(out + ".npz") as saved:
            for member in saved.files:
                bundle[case["case_id"] + "/" + member] = saved[member]
    np.savez(os.path.join(outdir, "ops", "compose.npz"), **bundle)
    manifest = {"schema_version": 1, "module": MODULE, "jax_version": gg.JAX_VERSION, "x64": x64, "unrunnable": [], "cases": entries}
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(entries)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    gg.preflight()
    os.makedirs(args.work, exist_ok=True)
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    found = pinned_cases()
    os.makedirs(os.path.join(base, "text"), exist_ok=True)
    with open(os.path.join(base, "text", "reference.jsonl"), "w", encoding="utf-8") as sink:
        for set_name, x64 in cf.SETS:
            count = write_set(set_name, x64, found[set_name], os.path.join(base, set_name), args.work, sink)
            print(MODULE + " " + set_name + " " + str(count) + " cases")


if __name__ == "__main__":
    main()
