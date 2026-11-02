import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import emit_common as ec
import gen_fuzz_goldens as gf
import gen_goldens as gg
import jax
import numpy as np
import option_registry as registry

warnings.filterwarnings("ignore")

MODULE = "option_fuzz"
SETS = (("x64_off", False), ("x64_on", True))
CASE_TIMEOUT = 45
PER_FUNCTION = 16
SEED = 20261102


def op_id(spec):
    return spec["module"] + "." + spec["name"]


def expand(seed):
    cases = {name: [] for name, _ in SETS}
    for spec in registry.all_specs():
        op = op_id(spec)
        seen = set()
        for index in range(PER_FUNCTION):
            rng = np.random.default_rng([seed, registry.digest(op), index])
            sample = registry.sample_case(spec, rng, index)
            key = json.dumps(sample, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            base = op + "_" + format(registry.digest(key), "08x")
            set_name, dtype = SETS[index % 2][0], "float32" if index % 2 == 0 else "float64"
            for chosen, chosen_dtype in [(set_name, dtype)]:
                cases[chosen].append(
                    {
                        "case_id": base + "_" + chosen_dtype,
                        "op": op,
                        "dtype": chosen_dtype,
                        "variant": "option",
                        "module": spec["module"],
                        "name": spec["name"],
                        "positional": sample["positional"],
                        "options": sample["options"],
                        "inputs": sample["inputs"],
                    }
                )
    for name in cases:
        cases[name].sort(key=lambda c: c["case_id"])
    return cases


def make_inputs(case, seed):
    rng = np.random.default_rng([seed, registry.digest(case["case_id"]), 1])
    out = {}
    for name in sorted(case["inputs"]):
        spec = case["inputs"][name]
        shape = tuple(spec["shape"])
        kind = spec["kind"]
        if kind == "real":
            out[name] = rng.standard_normal(shape).astype(case["dtype"])
        elif kind == "positive":
            out[name] = rng.uniform(0.1, 2.0, shape).astype(case["dtype"])
        elif kind == "complex":
            cdt = {"float32": "complex64", "float64": "complex128"}[case["dtype"]]
            out[name] = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(cdt)
        elif kind == "int":
            out[name] = rng.integers(-3, 4, shape).astype("int32")
        else:
            out[name] = rng.integers(0, 2, shape).astype("bool")
    return out


def to_outputs(result):
    return {"out" + str(i): o for i, o in enumerate(registry.flatten_output(result))}


def run_case(case, arrays):
    names = sorted(arrays)
    entry = {
        "case_id": case["case_id"],
        "op": case["op"],
        "dtype": case["dtype"],
        "module": case["module"],
        "name": case["name"],
        "positional": case["positional"],
        "options": case["options"],
        "args": gf.meta({k: arrays[k] for k in names}),
    }
    saved = {"in_" + k: arrays[k] for k in names}

    def message(error):
        return str(error)

    try:
        devices = {k: jax.numpy.asarray(v) for k, v in arrays.items()}
        eager = to_outputs(registry.call(case["module"], case["name"], case["positional"], case["options"], devices))
        entry["eager"] = gf.meta(eager)
        saved.update({"eager_" + k: v for k, v in eager.items()})
    except Exception as error:
        entry["eager_error"] = type(error).__name__
        entry["eager_error_text"] = message(error)

    def traced(*values):
        return registry.call(case["module"], case["name"], case["positional"], case["options"], dict(zip(names, values)))

    try:
        jitted = to_outputs(jax.jit(traced)(*[arrays[k] for k in names]))
        entry["jit"] = gf.meta(jitted)
        saved.update({"jit_" + k: v for k, v in jitted.items()})
    except Exception as error:
        entry["jit_error"] = type(error).__name__
        entry["jit_error_text"] = message(error)
    return entry, saved


def stored_cases(work, seed):
    path = os.path.join(work, "cases.json")
    seed_path = os.path.join(work, "seed")
    if os.path.exists(seed_path):
        with open(seed_path, encoding="utf-8") as fh:
            if int(fh.read()) != seed:
                raise SystemExit("work directory " + work + " was made with another seed")
    else:
        with open(seed_path, "w", encoding="utf-8") as fh:
            fh.write(str(seed))
    if not os.path.exists(path):
        cases = expand(seed)
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(cases, fh)
        os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def worker(set_name, shard, shards, work, seed):
    core, _fd = gf.take_core()
    os.sched_setaffinity(0, {core})
    jax.config.update("jax_enable_x64", set_name == "x64_on")
    cases = stored_cases(work, seed)[set_name]
    directory = os.path.join(work, set_name)
    current = os.path.join(directory, "current." + str(shard))
    count = 0
    for position, case in enumerate(cases):
        if position % shards != shard:
            continue
        stem = os.path.join(directory, case["case_id"].replace("/", "_"))
        if os.path.exists(stem + ".json"):
            continue
        with open(current, "w", encoding="utf-8") as fh:
            fh.write(case["case_id"])
        arrays = make_inputs(case, seed)
        entry, saved = run_case(case, arrays)
        np.savez(stem + ".tmp.npz", **saved)
        os.replace(stem + ".tmp.npz", stem + ".npz")
        with open(stem + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.replace(stem + ".tmp", stem + ".json")
        count += 1
        if count % 50 == 0:
            jax.clear_caches()
    with open(current, "w", encoding="utf-8") as fh:
        fh.write("")


def launch(set_name, shard, shards, work, seed):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.Popen(
        ["nice", "-n", "10", sys.executable, os.path.abspath(__file__), "--worker", set_name, str(shard), str(shards), work, str(seed)],
        env=env,
    )


def supervise(set_name, shards, work, seed, limit):
    directory = os.path.join(work, set_name)
    os.makedirs(directory, exist_ok=True)
    running = {i: launch(set_name, i, shards, work, seed) for i in range(shards)}
    started = time.time()
    while running:
        time.sleep(1)
        if time.time() - started > limit:
            for proc in running.values():
                proc.kill()
                proc.wait()
            return False
        for shard, proc in list(running.items()):
            path = os.path.join(directory, "current." + str(shard))
            status = proc.poll()
            if status == 0:
                del running[shard]
                continue
            stalled = status is None and os.path.exists(path) and time.time() - os.path.getmtime(path) > CASE_TIMEOUT
            if status is None and not stalled:
                continue
            if stalled:
                proc.kill()
                proc.wait()
            name = open(path, encoding="utf-8").read().strip() if os.path.exists(path) else ""
            if name:
                reason = "timeout" if stalled else "crash " + str(status)
                sys.stdout.write("unrunnable " + set_name + " " + name + " " + reason + "\n")
                with open(os.path.join(directory, name.replace("/", "_") + ".json"), "w", encoding="utf-8") as fh:
                    json.dump({"case_id": name, "unrunnable": reason}, fh)
                os.remove(path)
            running[shard] = launch(set_name, shard, shards, work, seed)
    return True


def assemble(set_name, x64, work, seed, outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    cases = stored_cases(work, seed)[set_name]
    directory = os.path.join(work, set_name)
    entries, unrunnable, by_op = [], [], {}
    for case in cases:
        stem = os.path.join(directory, case["case_id"].replace("/", "_"))
        with open(stem + ".json", encoding="utf-8") as fh:
            entry = json.load(fh)
        if "unrunnable" in entry:
            unrunnable.append({"case_id": case["case_id"], "reason": entry["unrunnable"]})
            continue
        entries.append(entry)
        by_op.setdefault(case["op"], []).append(case["case_id"])
    for op, ids in by_op.items():
        bundle = {}
        for case_id in ids:
            with np.load(os.path.join(directory, case_id.replace("/", "_") + ".npz")) as stored:
                for member in stored.files:
                    bundle[case_id + "/" + member] = stored[member]
        np.savez(os.path.join(outdir, "ops", op + ".npz"), **bundle)
    manifest = {
        "schema_version": 1,
        "module": MODULE,
        "seed": seed,
        "jax_version": gg.JAX_VERSION,
        "x64": x64,
        "unrunnable": unrunnable,
        "cases": entries,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(entries), len(unrunnable)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--work", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=560)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    gg.preflight()
    os.makedirs(args.work, exist_ok=True)
    stored = stored_cases(args.work, args.seed)
    if args.count:
        for name, cases in stored.items():
            sys.stdout.write(name + " " + str(len(cases)) + " cases, " + str(len({c["op"] for c in cases})) + " ops\n")
        return
    started = time.time()
    for set_name, _ in SETS:
        left = args.limit - (time.time() - started)
        if not supervise(set_name, args.jobs, args.work, args.seed, left):
            sys.stdout.write("time limit reached, run again to continue\n")
            sys.exit(3)
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    for set_name, x64 in SETS:
        count, unrunnable = assemble(set_name, x64, args.work, args.seed, os.path.join(base, set_name))
        sys.stdout.write(MODULE + " " + set_name + " " + str(count) + " cases, " + str(unrunnable) + " unrunnable\n")


if __name__ == "__main__":
    main()
