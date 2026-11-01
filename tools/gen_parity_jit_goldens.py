import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib

import emit_common as ec
import gen_goldens as gg
import gen_parity_goldens as gp
import jax
import numpy as np
import parity_registry as registry

MODULE = "parity_jit"


def under_jit(op, names, params):
    def call(*values):
        return registry.OPS[op](**dict(zip(names, values)), p=params)

    return jax.jit(call)


def eager_inputs_match(path, arrays):
    if not os.path.exists(path):
        return True
    with np.load(path) as stored:
        if sorted(stored.files) != sorted(arrays):
            return False
        return all(
            stored[k].dtype == v.dtype and stored[k].shape == v.shape and stored[k].tobytes() == v.tobytes()
            for k, v in arrays.items()
        )


def run_shard(x64, index, count, outdir):
    jax.config.update("jax_enable_x64", x64)
    eager_dir = os.path.join(gg.ROOT, "goldens", gp.MODULE, "x64_on" if x64 else "x64_off", "inputs")
    selected = []
    for c in gp.expand():
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        if x64 and c["op"] in registry.SINGLE_PRECISION_ONLY:
            continue
        selected.append(c)
    entries = []
    for position, c in enumerate(selected):
        if position % count != index:
            continue
        rng = np.random.default_rng(zlib.adler32(c["case_id"].encode("utf-8")))
        names = sorted(c["inputs"])
        arrays = {k: gp.draw(rng, c["inputs"][k], c["dtype"]) for k in names}
        if not eager_inputs_match(os.path.join(eager_dir, c["case_id"] + ".npz"), arrays):
            raise SystemExit("inputs differ from goldens/parity for " + c["case_id"])
        entry = {"case_id": c["case_id"], "op": c["op"]}
        try:
            result = under_jit(c["op"], names, c["params"])(*[arrays[k] for k in names])
            outs = list(result) if isinstance(result, (tuple, list)) else [result]
            outs = [np.asarray(o, order="C") for o in outs]
        except Exception as error:
            entry["jit_error"] = type(error).__name__
            entries.append(entry)
            continue
        if registry.compares_at_single_precision(c["op"], arrays):
            narrow = "float16" if arrays["a"].dtype == np.float16 else "float32"
            outs = [o.astype(narrow) if o.dtype in (np.float64, np.float32) else o for o in outs]
        out_arrays = {"out" + str(i): o for i, o in enumerate(outs)}
        np.savez(os.path.join(outdir, "outputs", c["case_id"] + ".npz"), **out_arrays)
        entry["outputs"] = [
            {"name": k, "dtype": v.dtype.name, "shape": [int(d) for d in v.shape]} for k, v in out_arrays.items()
        ]
        entries.append(entry)
    with open(os.path.join(outdir, "shard" + str(index) + ".json"), "w", encoding="utf-8") as fh:
        json.dump(entries, fh)


def gen_set(x64, outdir, jobs):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "outputs"))
    workers = []
    for index in range(jobs):
        command = [sys.executable, os.path.abspath(__file__), "--shard", str(index), str(jobs), "--x64", str(int(x64)), "--outdir", outdir]
        workers.append(subprocess.Popen(command))
    if any(w.wait() != 0 for w in workers):
        raise SystemExit("a shard failed for " + outdir)
    entries = []
    for index in range(jobs):
        path = os.path.join(outdir, "shard" + str(index) + ".json")
        with open(path, encoding="utf-8") as fh:
            entries.extend(json.load(fh))
        os.remove(path)
    entries.sort(key=lambda e: e["case_id"])
    manifest = {
        "schema_version": 1,
        "module": MODULE,
        "jax_version": gg.JAX_VERSION,
        "x64": x64,
        "cases": entries,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(entries), sum(1 for e in entries if "jit_error" in e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--root", default=None)
    parser.add_argument("--shard", type=int, nargs=2, default=None)
    parser.add_argument("--x64", type=int, default=0)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    gg.preflight()
    if args.shard:
        run_shard(bool(args.x64), args.shard[0], args.shard[1], args.outdir)
        return
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    n_off, e_off = gen_set(False, os.path.join(base, "x64_off"), args.jobs)
    n_on, e_on = gen_set(True, os.path.join(base, "x64_on"), args.jobs)
    sys.stdout.write(f"{MODULE} x64_off {n_off} ({e_off} jit errors) x64_on {n_on} ({e_on} jit errors)\n")


if __name__ == "__main__":
    main()
