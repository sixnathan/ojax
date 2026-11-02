import argparse
import json
import os
import subprocess
import sys

import gen_goldens as gg
import jax
import numpy as np
import apicov_registry
import coverage_registry
import parity_registry

REGISTRIES = {"parity": parity_registry, "coverage": coverage_registry, "apicov": apicov_registry}

MODULE = "aot_cost"
MEM_FIELDS = [
    "generated_code_size_in_bytes", "argument_size_in_bytes",
    "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
    "host_generated_code_size_in_bytes", "host_argument_size_in_bytes",
    "host_output_size_in_bytes", "host_alias_size_in_bytes",
    "host_temp_size_in_bytes", "peak_memory_in_bytes",
]


def selected_cases(x64):
    out = []
    for suite in REGISTRIES:
        path = os.path.join(gg.ROOT, "goldens", suite, "x64_on" if x64 else "x64_off", "manifest.json")
        with open(path, encoding="utf-8") as fh:
            cases = json.load(fh)["cases"]
        first = {}
        for c in sorted(cases, key=lambda c: c["case_id"]):
            first.setdefault(c["op"], c)
        out.extend((suite, first[op]) for op in sorted(first))
    return out


def run_shard(x64, index, count, out):
    jax.config.update("jax_enable_x64", x64)
    rows = []
    for position, (suite, c) in enumerate(selected_cases(x64)):
        if position % count != index:
            continue
        registry = REGISTRIES[suite]
        inputs_dir = os.path.join(gg.ROOT, "goldens", suite, "x64_on" if x64 else "x64_off", "inputs")
        names = [a["name"] for a in c["args"]]
        with np.load(os.path.join(inputs_dir, c["case_id"] + ".npz")) as stored:
            values = [stored[n] for n in names]
        op, params = c["op"], c["params"]

        def call(*vals, op=op, params=params, names=names, registry=registry):
            return registry.OPS[op](**dict(zip(names, vals)), p=params)

        row = {"suite": suite, "case_id": c["case_id"], "op": op, "params": params, "args": c["args"]}
        try:
            compiled = jax.jit(call).lower(*values).compile()
            cost = compiled.cost_analysis()
            mem = compiled.memory_analysis()
        except Exception as error:
            row["error"] = type(error).__name__
            rows.append(row)
            continue
        row["cost"] = sorted([k, float(v)] for k, v in cost.items())
        row["mem"] = sorted([k, int(getattr(mem, k))] for k in MEM_FIELDS)
        rows.append(row)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


def gen_set(x64, outdir, jobs):
    os.makedirs(outdir, exist_ok=True)
    tag = "x64_on" if x64 else "x64_off"
    workers = []
    parts = []
    for index in range(jobs):
        part = os.path.join(outdir, f"{tag}.shard{index}.json")
        parts.append(part)
        command = [sys.executable, os.path.abspath(__file__), "--shard", str(index), str(jobs),
                   "--x64", str(int(x64)), "--out", part]
        workers.append(subprocess.Popen(command))
    if any(w.wait() != 0 for w in workers):
        raise SystemExit("a shard failed for " + tag)
    rows = []
    for part in parts:
        with open(part, encoding="utf-8") as fh:
            rows.extend(json.load(fh))
        os.remove(part)
    rows.sort(key=lambda r: (r["suite"], r["case_id"]))
    with open(os.path.join(outdir, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"jax": jax.__version__, "x64": x64, "rows": rows}, fh, indent=1, sort_keys=True)
    return len(rows), sum(1 for r in rows if "error" in r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--root", default=None)
    parser.add_argument("--shard", type=int, nargs=2, default=None)
    parser.add_argument("--x64", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    gg.preflight()
    if args.shard:
        run_shard(bool(args.x64), args.shard[0], args.shard[1], args.out)
        return
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    n_off, e_off = gen_set(False, base, args.jobs)
    n_on, e_on = gen_set(True, base, args.jobs)
    sys.stdout.write(f"{MODULE} x64_off {n_off} ({e_off} errors) x64_on {n_on} ({e_on} errors)\n")


if __name__ == "__main__":
    main()
