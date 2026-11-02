import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import compare_text
import emit_common as ec
import fcntl
import gen_goldens as gg
import jax
import numpy as np
import parity_registry as registry
from jax._src import core as jax_core

MODULE = "fuzz"
SEED = 20261101
SETS = (("x64_off", False), ("x64_on", True))
TRANSFORMED = ("swv_", "swg_", "swj_")
VALUE_VARIANTS = ("edge", "wide")
SHAPE_VARIANTS = ("shape_small", "shape_other")
CASE_TIMEOUT = 45
JAX_DOES_NOT_FINISH = re.compile(r"^(?:sw[gjv]?_)?(?:sx_)?(?:exp1|expi|expn)_.*_(?:edge|wide)$|^(?:sw[gjv]?_)?(?:gcd|lcm)_.*_edge$|^swv_sp_gammainc_.*_edge$")


def digest(text):
    return zlib.adler32(text.encode("utf-8"))


def base_id(op, inputs, params):
    shapes = "_".join("x".join(str(d) for d in v["shape"]) for _, v in sorted(inputs.items()))
    return op + "_" + shapes + "_" + format(digest(json.dumps([inputs, params], sort_keys=True)), "08x")


def uses_float_dtype(inputs):
    return any(s["kind"] in ("real", "positive", "complex") for s in inputs.values())


def fuzz_params(params, seed_rng):
    out = dict(params)
    if "seed" in out:
        out["seed"] = int(seed_rng.integers(0, 2**31 - 1))
    return out


def configs_for(op, inputs, variant):
    if op.startswith(TRANSFORMED):
        pick = digest(str(SEED) + op + variant) % 2
        return [[("x64_off", "float32"), ("x64_on", "float64")][pick]]
    if uses_float_dtype(inputs):
        configs = [("x64_off", "float32"), ("x64_on", "float64")]
        if variant == "edge":
            configs.append(("x64_on", "float32"))
        return configs
    return [("x64_off", "float32"), ("x64_on", "float64")]


def variants_for(op):
    if op.startswith(TRANSFORMED):
        return ("edge", "shape_other")
    return VALUE_VARIANTS + SHAPE_VARIANTS


def expand():
    cases = {name: [] for name, _ in SETS}
    seen = set()
    for op, inputs, params in registry.all_cases():
        base = base_id(op, inputs, params)
        for variant in variants_for(op):
            for set_name, dtype in configs_for(op, inputs, variant):
                x64 = set_name == "x64_on"
                if x64 and op in registry.SINGLE_PRECISION_ONLY:
                    continue
                case_id = base + "_" + dtype + "_" + variant
                key = (set_name, case_id)
                if key in seen:
                    continue
                seen.add(key)
                rng = np.random.default_rng([SEED, digest(case_id)])
                cases[set_name].append(
                    {
                        "case_id": case_id,
                        "op": op,
                        "dtype": dtype,
                        "variant": variant,
                        "inputs": inputs,
                        "params": fuzz_params(params, rng),
                    }
                )
    for name in cases:
        cases[name].sort(key=lambda c: c["case_id"])
    return cases


def excluded(case_id):
    return JAX_DOES_NOT_FINISH.match(case_id) is not None


def stored_cases(work):
    path = os.path.join(work, "cases.json")
    if not os.path.exists(path):
        cases = expand()
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(cases, fh)
        os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mutate_shape(rng, variant, inputs):
    sizes = sorted({d for s in inputs.values() for d in s["shape"]})
    if variant == "shape_small":
        choice = "empty" if sizes and rng.random() < 0.7 else "scalar"
    else:
        choice = "one" if sizes and rng.random() < 0.3 else "grown"
    if choice == "scalar":
        return {k: () for k in inputs}
    if not sizes:
        return {k: tuple(s["shape"]) for k, s in inputs.items()}
    if choice == "empty":
        target = sizes[int(rng.integers(0, len(sizes)))]
        mapping = {target: 0}
    elif choice == "one":
        target = sizes[int(rng.integers(0, len(sizes)))]
        mapping = {target: 1}
    else:
        mapping = {d: d + int(rng.integers(1, 3)) for d in sizes}
    return {k: tuple(mapping.get(d, d) for d in s["shape"]) for k, s in inputs.items()}


def float_pool(name):
    info = np.finfo(name)
    nan = np.array(np.nan, dtype=name)
    return np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            nan,
            np.copysign(nan, -1.0),
            float(info.smallest_subnormal),
            -float(info.smallest_subnormal),
            float(info.tiny),
            -float(info.tiny),
            float(info.max),
            float(info.min),
            float(info.eps),
            1.0,
            -1.0,
            0.5,
        ],
        dtype=name,
    )


def positive(values):
    return np.where(np.signbit(values), -values, values)


def float_values(rng, shape, name, variant):
    if variant == "edge":
        pool = float_pool(name)
        picks = pool[rng.integers(0, len(pool), shape)]
        normal = rng.standard_normal(shape).astype(name)
        return np.where(rng.random(shape) < 0.6, picks, normal)
    if variant == "wide":
        info = np.finfo(name)
        span = float(np.log10(float(info.max))) - 0.5
        magnitude = 10.0 ** rng.uniform(-span, span, shape)
        sign = rng.choice([-1.0, 1.0], shape)
        values = (magnitude * sign).astype(name)
        return np.where(rng.random(shape) < 0.05, np.zeros(shape, dtype=name), values)
    return rng.standard_normal(shape).astype(name)


def int_values(rng, shape, name, variant):
    info = np.iinfo(name)
    if variant == "edge":
        pool = np.array(sorted({info.min, info.max, 0, 1, max(info.min, -1), info.max // 2}), dtype=name)
        picks = pool[rng.integers(0, len(pool), shape)]
        small = rng.integers(max(info.min, -4), 5, shape).astype(name)
        return np.where(rng.random(shape) < 0.6, picks, small)
    if variant == "wide":
        return rng.integers(info.min, info.max, shape, endpoint=True, dtype=name)
    return rng.integers(max(info.min, -4), 5, shape).astype(name)


def draw(rng, spec, shape, dtype, variant):
    kind = spec["kind"]
    if kind == "real":
        return float_values(rng, shape, dtype, variant)
    if kind == "positive":
        if variant in ("edge", "wide"):
            return positive(float_values(rng, shape, dtype, variant))
        return rng.uniform(0.1, 2.0, shape).astype(dtype)
    if kind == "complex":
        cdt = {"float32": "complex64", "float64": "complex128"}[dtype]
        out = np.zeros(shape, dtype=cdt)
        re = float_values(rng, shape, dtype, variant)
        im = float_values(rng, shape, dtype, variant)
        view = out.reshape(-1)
        view.real[...] = re.reshape(-1)
        view.imag[...] = im.reshape(-1)
        return out
    if kind == "int":
        return int_values(rng, shape, "int32", variant)
    if kind == "bool":
        return rng.integers(0, 2, shape).astype("bool")
    if kind == "typed":
        target = spec["dtype"]
        if target == "bool":
            return rng.integers(0, 2, shape).astype("bool")
        if target == "float16":
            return float_values(rng, shape, "float16", variant)
        if target == "bfloat16":
            values = float_values(rng, shape, "float32", variant)
            return np.asarray(jax.numpy.asarray(values, dtype=jax.numpy.bfloat16))
        return int_values(rng, shape, target, variant)
    raise SystemExit("unknown input kind " + kind)


def make_inputs(case):
    rng = np.random.default_rng([SEED, digest(case["case_id"]), 1])
    variant = case["variant"]
    if variant.startswith("shape"):
        shapes = mutate_shape(rng, variant, case["inputs"])
    else:
        shapes = {k: tuple(s["shape"]) for k, s in case["inputs"].items()}
    return {k: np.asarray(draw(rng, case["inputs"][k], shapes[k], case["dtype"], variant)) for k in sorted(case["inputs"])}


def to_outputs(case, arrays, result):
    outs = list(result) if isinstance(result, (tuple, list)) else [result]
    outs = [np.asarray(o, order="C") for o in outs]
    if registry.compares_at_single_precision(case["op"], arrays):
        narrow = "float16" if arrays["a"].dtype == np.float16 else "float32"
        outs = [o.astype(narrow) if o.dtype in (np.float64, np.float32) else o for o in outs]
    return {"out" + str(i): o for i, o in enumerate(outs)}


def meta(arrays):
    return [{"name": k, "dtype": v.dtype.name, "shape": [int(d) for d in v.shape]} for k, v in arrays.items()]


def run_case(case, arrays):
    names = sorted(arrays)
    op = registry.OPS[case["op"]]
    params = case["params"]
    entry = {
        "case_id": case["case_id"],
        "op": case["op"],
        "dtype": case["dtype"],
        "variant": case["variant"],
        "params": params,
        "args": meta({k: arrays[k] for k in names}),
    }
    saved = {"in_" + k: arrays[k] for k in names}
    try:
        devices = {k: jax.numpy.asarray(v) for k, v in arrays.items()}
        eager = to_outputs(case, arrays, op(**devices, p=params))
        entry["eager"] = meta(eager)
        saved.update({"eager_" + k: v for k, v in eager.items()})
    except Exception as error:
        entry["eager_error"] = type(error).__name__

    def call(*values):
        return op(**dict(zip(names, values)), p=params)

    try:
        jitted = to_outputs(case, arrays, jax.jit(call)(*[arrays[k] for k in names]))
        entry["jit"] = meta(jitted)
        saved.update({"jit_" + k: v for k, v in jitted.items()})
    except Exception as error:
        entry["jit_error"] = type(error).__name__
    return entry, saved


def take_core():
    spec = os.environ.get("OJAX_SLOT_CORES", "2-17")
    low, _, high = spec.partition("-")
    directory = os.environ.get("OJAX_SLOT_DIR", "/tmp/ojax_slots")
    os.makedirs(directory, exist_ok=True)
    while True:
        for core in range(int(low), int(high or low) + 1):
            fd = os.open(os.path.join(directory, str(core)), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return core, fd
            except OSError:
                os.close(fd)
        time.sleep(0.2)


def worker(set_name, shard, shards, work):
    core, _fd = take_core()
    os.sched_setaffinity(0, {core})
    x64 = set_name == "x64_on"
    jax.config.update("jax_enable_x64", x64)
    cases = stored_cases(work)[set_name]
    directory = os.path.join(work, set_name)
    current = os.path.join(directory, "current." + str(shard))
    count = 0
    for position, case in enumerate(cases):
        if position % shards != shard or excluded(case["case_id"]):
            continue
        stem = os.path.join(directory, case["case_id"])
        if os.path.exists(stem + ".json"):
            continue
        with open(current, "w", encoding="utf-8") as fh:
            fh.write(case["case_id"])
        arrays = make_inputs(case)
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


def launch(mode, set_name, shard, shards, work):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.Popen(
        ["nice", "-n", "10", sys.executable, os.path.abspath(__file__), mode, set_name, str(shard), str(shards), work],
        env=env,
    )


def supervise(mode, set_name, shards, work, limit, suffix=""):
    directory = os.path.join(work, set_name + suffix)
    os.makedirs(directory, exist_ok=True)
    running = {i: launch(mode, set_name, i, shards, work) for i in range(shards)}
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
                with open(os.path.join(directory, name + ".json"), "w", encoding="utf-8") as fh:
                    json.dump({"case_id": name, "unrunnable": reason}, fh)
                os.remove(path)
            running[shard] = launch(mode, set_name, shard, shards, work)
    return True


def assemble(set_name, x64, work, outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    cases = stored_cases(work)[set_name]
    directory = os.path.join(work, set_name)
    entries = []
    unrunnable = []
    by_op = {}
    for case in cases:
        path = os.path.join(directory, case["case_id"])
        if excluded(case["case_id"]):
            unrunnable.append({"case_id": case["case_id"], "reason": "JAX does not finish in 90 s"})
            continue
        with open(path + ".json", encoding="utf-8") as fh:
            entry = json.load(fh)
        if "unrunnable" in entry:
            unrunnable.append({"case_id": case["case_id"], "reason": entry["unrunnable"]})
            continue
        entries.append(entry)
        by_op.setdefault(case["op"], []).append(case["case_id"])
    for op, ids in by_op.items():
        bundle = {}
        for case_id in ids:
            with np.load(os.path.join(directory, case_id + ".npz")) as stored:
                for member in stored.files:
                    bundle[case_id + "/" + member] = stored[member]
        np.savez(os.path.join(outdir, "ops", op + ".npz"), **bundle)
    manifest = {
        "schema_version": 1,
        "module": MODULE,
        "seed": SEED,
        "jax_version": gg.JAX_VERSION,
        "x64": x64,
        "unrunnable": unrunnable,
        "cases": entries,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(entries), len(unrunnable)


def text_worker(set_name, shard, shards, work):
    core, _fd = take_core()
    os.sched_setaffinity(0, {core})
    jax.config.update("jax_enable_x64", set_name == "x64_on")
    cases = [c for c in stored_cases(work)[set_name] if c["variant"].startswith("shape")]
    directory = os.path.join(work, set_name + "_text")
    current = os.path.join(directory, "current." + str(shard))
    count = 0
    for position, case in enumerate(cases):
        if position % shards != shard:
            continue
        stem = os.path.join(directory, case["case_id"])
        if os.path.exists(stem + ".json") or not os.path.exists(os.path.join(work, set_name, case["case_id"] + ".npz")):
            continue
        with open(current, "w", encoding="utf-8") as fh:
            fh.write(case["case_id"])
        arrays = make_inputs(case)
        names = sorted(arrays)
        op = registry.OPS[case["op"]]

        def fn(*xs, _op=op, _names=names, _params=case["params"]):
            return _op(**dict(zip(_names, xs)), p=_params)

        row = compare_text._reference_case(fn, [arrays[k] for k in names], jax, jax_core)
        row.update({"set": "fuzz/" + set_name, "case": case["case_id"]})
        with open(stem + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(row, fh, sort_keys=True)
        os.replace(stem + ".tmp", stem + ".json")
        count += 1
        if count % 50 == 0:
            jax.clear_caches()
    with open(current, "w", encoding="utf-8") as fh:
        fh.write("")


def assemble_text(work, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "reference.jsonl")
    rows = 0
    with open(path, "w", encoding="utf-8") as sink:
        for set_name, _ in SETS:
            directory = os.path.join(work, set_name + "_text")
            for case in stored_cases(work)[set_name]:
                name = os.path.join(directory, case["case_id"] + ".json")
                if not os.path.exists(name):
                    continue
                with open(name, encoding="utf-8") as fh:
                    row = json.load(fh)
                if "unrunnable" in row:
                    continue
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--worker", "--text-worker"):
        target = worker if sys.argv[1] == "--worker" else text_worker
        args = [sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]]
        target(*args)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--work", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=560)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    gg.preflight()
    os.makedirs(args.work, exist_ok=True)
    if args.count:
        for name, cases in stored_cases(args.work).items():
            sys.stdout.write(name + " " + str(len(cases)) + "\n")
        return
    stored_cases(args.work)
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    started = time.time()
    for set_name, x64 in SETS:
        left = args.limit - (time.time() - started)
        if args.text:
            finished = supervise("--text-worker", set_name, args.jobs, args.work, left, "_text")
        else:
            finished = supervise("--worker", set_name, args.jobs, args.work, left)
        if not finished:
            sys.stdout.write("time limit reached, run again to continue\n")
            sys.exit(3)
    if args.text:
        sys.stdout.write("reference text rows " + str(assemble_text(args.work, os.path.join(base, "text"))) + "\n")
        return
    for set_name, x64 in SETS:
        count, unrunnable = assemble(set_name, x64, args.work, os.path.join(base, set_name))
        sys.stdout.write(MODULE + " " + set_name + " " + str(count) + " cases, " + str(unrunnable) + " unrunnable\n")


if __name__ == "__main__":
    main()
