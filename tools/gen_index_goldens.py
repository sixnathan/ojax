import argparse
import json
import multiprocessing
import os
import shutil
import sys
import warnings
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_text
import emit_common as ec
import gen_goldens as gg
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core

warnings.filterwarnings("ignore")

MODULE = "index_fuzz"
DEFAULT_SEED = 20261103
SETS = (("x64_off", False), ("x64_on", True))
FUZZ_CASES = 500
DTYPES = {
    "x64_off": ("float32", "int32", "uint8", "int8", "bool", "float16", "complex64"),
    "x64_on": ("float32", "float64", "int32", "int64", "uint32", "bool", "complex128"),
}
UPDATES = ("set", "add", "subtract", "multiply", "divide", "power", "min", "max", "apply")
MODES = (None, "clip", "fill", "drop", "promise_in_bounds")


def digest(text):
    return zlib.crc32(text.encode("utf-8"))


def sl(start=None, stop=None, step=None):
    return {"k": "slice", "start": start, "stop": stop, "step": step}


def iv(v):
    return {"k": "int", "v": v}


def arr(name):
    return {"k": "arr", "name": name}


NONE = {"k": "none"}
ELL = {"k": "ellipsis"}


def build_index(spec, arrays):
    out = []
    for item in spec:
        k = item["k"]
        if k == "int":
            out.append(item["v"])
        elif k == "slice":
            out.append(slice(item["start"], item["stop"], item["step"]))
        elif k == "none":
            out.append(None)
        elif k == "ellipsis":
            out.append(Ellipsis)
        elif k == "bool":
            out.append(bool(item["v"]))
        else:
            out.append(arrays[item["name"]])
    return tuple(out)


def program(case):
    params = case["params"]
    spec = params["index"]
    const = params.get("const", False)
    names = sorted(a for a in case["arg_names"])

    def f(*values):
        env = dict(zip(names, values))
        if const:
            env.update({k: np.asarray(v["data"], dtype=v["dtype"]) for k, v in params["consts"].items()})
        x = env["x"]
        idx = build_index(spec, env)
        op = params["op"]
        if op == "getitem":
            return x[idx]
        kw = {}
        for key in ("indices_are_sorted", "unique_indices", "mode", "wrap_negative_indices"):
            if key in params:
                kw[key] = params[key]
        ref = x.at[idx]
        if op == "get":
            if "fill_value" in params:
                kw["fill_value"] = params["fill_value"]
            return ref.get(**kw)
        if op == "apply":
            return ref.apply(jnp.negative, **kw)
        if "y_scalar" in params:
            y = params["y_scalar"]
        else:
            y = env["y"]
        return getattr(ref, op)(y, **kw)

    return f


def values_for(rng, dtype, shape):
    if dtype == "bool":
        return rng.random(shape) < 0.5
    if dtype.startswith("complex"):
        return (rng.integers(-5, 6, shape) + 1j * rng.integers(-5, 6, shape)).astype(dtype)
    if dtype.startswith("uint"):
        return rng.integers(0, 9, shape).astype(dtype)
    if dtype.startswith("int"):
        return rng.integers(-9, 10, shape).astype(dtype)
    return (rng.integers(-20, 21, shape) * 0.25).astype(dtype)


def index_array(rng, shape, size, kind):
    if kind == "bool":
        return rng.random(shape) < 0.5
    low = -size - 2 if kind == "oob" else -size
    high = size + 2 if kind == "oob" else size
    if size == 0:
        low, high = -1, 2
    dt = rng.choice(["int32", "int32", "int8", "uint8", "int64"])
    v = rng.integers(low, max(high, low + 1), shape)
    if dt == "uint8":
        v = np.abs(v)
    return v.astype(dt)


def fixed_specs():
    s3 = [4, 5, 6]
    out = []
    forms = [
        [iv(1)], [iv(-1)], [iv(4)], [iv(-5)], [iv(9)], [sl(1, 3)], [sl(None, None, 2)], [sl(None, None, -1)],
        [sl(3, 0, -2)], [sl(-2)], [sl(10, 20)], [sl(-10, 2)], [NONE], [ELL], [ELL, iv(0)], [iv(0), ELL, NONE],
        [NONE, sl(), NONE], [sl(), sl(None, None, -2), iv(2)], [iv(1), iv(2), iv(3)], [iv(-1), sl(1, 4, 2), NONE, iv(0)],
        [arr("i0")], [sl(), arr("i0")], [arr("i0"), sl(), arr("i1")], [arr("i0"), arr("i1")], [iv(1), sl(), arr("i0")],
        [arr("i0"), iv(1)], [NONE, arr("i0"), NONE, arr("i1")], [ELL, arr("i0")], [arr("s0")], [arr("s0"), sl(None, None, -1)],
        [arr("m0")], [arr("m1")], [sl(), arr("m2")], [{"k": "bool", "v": True}], [{"k": "bool", "v": False}],
        [arr("m0"), arr("i1")], [iv(0), {"k": "bool", "v": True}, sl()], [sl(None, None, 0)], [ELL, ELL],
        [iv(0), iv(0), iv(0), iv(0)], [arr("f0")], [arr("mbad")], [sl(1, 3), sl(4, 1, -1), sl(None, None, 3)],
    ]
    for form in forms:
        out.append({"shape": s3, "index": form})
    for form in ([iv(0)], [iv(-1)], [iv(3)], [sl(2, 2)], [arr("i0")], [arr("m0")], [NONE]):
        out.append({"shape": [3, 0], "index": form})
        out.append({"shape": [0], "index": form})
    return out


def array_specs(rng, shape, spec):
    arrays = {}
    for item in spec:
        if item["k"] != "arr":
            continue
        name = item["name"]
        axis = min(spec.index(item), len(shape) - 1) if shape else 0
        size = shape[axis] if shape else 1
        if name == "i0":
            arrays[name] = index_array(rng, (3,), size, "oob" if rng.random() < 0.3 else "in")
        elif name == "i1":
            arrays[name] = index_array(rng, (2, 1) if rng.random() < 0.5 else (3,), size, "in")
        elif name == "s0":
            arrays[name] = np.asarray(rng.integers(-size, max(size, 1)), dtype="int32")
        elif name == "m0":
            arrays[name] = rng.random((shape[0] if shape else 1,)) < 0.5
        elif name == "m1":
            arrays[name] = rng.random(tuple(shape[:2])) < 0.5
        elif name == "m2":
            arrays[name] = rng.random(tuple(shape[1:])) < 0.5
        elif name == "mbad":
            arrays[name] = rng.random((7,)) < 0.5
        elif name == "f0":
            arrays[name] = np.asarray([0.0, 1.0], dtype="float32")
    return arrays


def random_spec(rng, rank):
    spec = []
    n = int(rng.integers(1, rank + 1)) + (1 if rng.random() < 0.06 else 0)
    used_ell = False
    names = iter(["i0", "i1", "i2"])
    for _ in range(n):
        r = rng.random()
        if r < 0.22:
            spec.append(iv(int(rng.integers(-6, 7))))
        elif r < 0.5:
            pick = lambda: None if rng.random() < 0.35 else int(rng.integers(-7, 8))
            step = None if rng.random() < 0.4 else int(rng.choice([1, 2, 3, -1, -2, -3]))
            spec.append(sl(pick(), pick(), step))
        elif r < 0.6:
            spec.append(NONE)
        elif r < 0.67 and not used_ell:
            used_ell = True
            spec.append(ELL)
        elif r < 0.9:
            spec.append(arr(next(names, "i2")))
        elif r < 0.95:
            spec.append(arr("m0"))
        else:
            spec.append({"k": "bool", "v": bool(rng.random() < 0.5)})
    return spec


def fuzz_arrays(rng, shape, spec):
    arrays = {}
    axis = 0
    shapes = [(3,), (2, 1), (1, 3), (), (2, 3), (0,)]
    for item in spec:
        if item["k"] == "arr":
            size = shape[axis] if axis < len(shape) else 1
            name = item["name"]
            if name == "m0":
                arrays[name] = rng.random((size,)) < 0.5
            else:
                sh = shapes[int(rng.integers(0, len(shapes)))]
                arrays[name] = index_array(rng, sh, size, "oob" if rng.random() < 0.25 else "in")
            axis += 1
        elif item["k"] in ("int", "slice"):
            axis += 1
    return arrays


def options(rng, op):
    p = {"op": op}
    if op == "getitem":
        return p
    r = rng.random()
    if r < 0.5:
        mode = MODES[int(rng.integers(0, len(MODES)))]
        if mode is not None:
            p["mode"] = mode
    if op == "get" and p.get("mode") == "fill" and rng.random() < 0.6:
        p["fill_value"] = float(rng.choice([-1.0, 0.0, 7.0]))
    if rng.random() < 0.15:
        p["indices_are_sorted"] = True
    if rng.random() < 0.15:
        p["unique_indices"] = True
    if rng.random() < 0.12:
        p["wrap_negative_indices"] = False
    return p


def make_case(set_name, case_id, rng, shape, spec, arrays, op, dtype):
    params = options(rng, op)
    params["index"] = spec
    if op in UPDATES and op != "apply" and rng.random() < 0.3:
        params["y_scalar"] = int(rng.integers(-3, 4)) if rng.random() < 0.5 else float(rng.choice([0.5, -2.0, 3.0]))
    if arrays and rng.random() < 0.2 and not any(v.dtype == np.float32 for v in arrays.values()):
        params["const"] = True
        params["consts"] = {k: {"dtype": v.dtype.name, "data": v.tolist()} for k, v in arrays.items()}
        arrays = {}
    return {"case_id": case_id, "op": "index", "dtype": dtype, "shape": shape, "params": params, "index_arrays": {k: [v.dtype.name, list(v.shape)] for k, v in arrays.items()}, "arrays_seed": int(rng.integers(0, 2**31))}, arrays


def expand(seed):
    out = {}
    for set_name, _ in SETS:
        rng = np.random.default_rng([seed, digest(set_name)])
        cases = []
        stash = {}
        dts = DTYPES[set_name]
        n = 0
        for spec in fixed_specs():
            for op in ("getitem", "get", "set", "add"):
                dtype = "float32" if op != "add" else dts[n % len(dts)]
                cid = "fixed_%04d" % n
                n += 1
                arrays = array_specs(rng, spec["shape"], spec["index"])
                case, arrays = make_case(set_name, cid, rng, spec["shape"], spec["index"], arrays, op, dtype)
                cases.append(case)
                stash[cid] = arrays
        for op in UPDATES + ("get",):
            for form in ([arr("i0")], [sl(None, None, 2)], [iv(-1), sl()], [arr("m0")]):
                for dtype in dts:
                    cid = "fixed_%04d" % n
                    n += 1
                    shape = [4, 5]
                    arrays = array_specs(rng, shape, form)
                    case, arrays = make_case(set_name, cid, rng, shape, form, arrays, op, dtype)
                    cases.append(case)
                    stash[cid] = arrays
        for i in range(FUZZ_CASES):
            rank = int(rng.integers(1, 4))
            shape = [int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.25, 0.2, 0.2])) for _ in range(rank)]
            spec = random_spec(rng, rank)
            arrays = fuzz_arrays(rng, shape, spec)
            op = ("getitem", "get") + UPDATES
            op = op[int(rng.integers(0, len(op)))]
            dtype = dts[int(rng.integers(0, len(dts)))]
            cid = "fuzz_%04d" % i
            case, arrays = make_case(set_name, cid, rng, shape, spec, arrays, op, dtype)
            cases.append(case)
            stash[cid] = arrays
        out[set_name] = (cases, stash)
    return out


def to_outputs(result):
    return {"out" + str(i): np.asarray(o, order="C") for i, o in enumerate(jax.tree_util.tree_leaves(result))}


def meta(arrays):
    return [{"name": k, "dtype": v.dtype.name, "shape": [int(d) for d in v.shape]} for k, v in arrays.items()]


def message(error):
    return type(error).__name__, str(error)


def inputs_for(case, index_arrays):
    rng = np.random.default_rng([case["arrays_seed"], 7])
    arrays = {"x": values_for(rng, case["dtype"], tuple(case["shape"]))}
    arrays.update(index_arrays)
    params = case["params"]
    if params["op"] in UPDATES and params["op"] != "apply" and "y_scalar" not in params:
        xj = jnp.asarray(arrays["x"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = dict(arrays)
                if params.get("const"):
                    env.update({k: np.asarray(v["data"], dtype=v["dtype"]) for k, v in params["consts"].items()})
                target = jax.eval_shape(lambda: xj[build_index(params["index"], env)]).shape
        except Exception:
            target = ()
        ydt = case["dtype"] if rng.random() < 0.8 else "float32"
        arrays["y"] = values_for(rng, ydt, target)
    return arrays


def run_case(case, arrays):
    names = sorted(arrays)
    case = dict(case, arg_names=names)
    values = [arrays[k] for k in names]
    entry = {k: case[k] for k in ("case_id", "op", "dtype", "shape", "params")}
    entry["args"] = meta({k: arrays[k] for k in names})
    saved = {"in_" + k: arrays[k] for k in names}
    try:
        eager = to_outputs(program(case)(*[jnp.asarray(v) for v in values]))
        entry["eager"] = meta(eager)
        saved.update({"eager_" + k: v for k, v in eager.items()})
    except Exception as error:
        entry["eager_error"], entry["eager_message"] = message(error)
    try:
        jitted = to_outputs(jax.jit(program(case))(*values))
        entry["jit"] = meta(jitted)
        saved.update({"jit_" + k: v for k, v in jitted.items()})
    except Exception as error:
        entry["jit_error"], entry["jit_message"] = message(error)
    row = compare_text._reference_case(program(case), values, jax, jax_core)
    row.update({"case": case["case_id"]})
    return entry, saved, row


def work(job):
    set_name, x64, case, index_arrays = job
    jax.config.update("jax_enable_x64", x64)
    arrays = inputs_for(case, index_arrays)
    return run_case(case, arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    plan = expand(args.seed)
    os.makedirs(os.path.join(base, "text"), exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    with open(os.path.join(base, "text", "reference.jsonl"), "w", encoding="utf-8") as sink:
        for set_name, x64 in SETS:
            cases, stash = plan[set_name]
            jobs = [(set_name, x64, c, stash[c["case_id"]]) for c in cases]
            with ctx.Pool(args.jobs, maxtasksperchild=100) as pool:
                results = pool.map(work, jobs, chunksize=8)
            outdir = os.path.join(base, set_name)
            if os.path.isdir(outdir):
                shutil.rmtree(outdir)
            os.makedirs(os.path.join(outdir, "ops"))
            bundle = {}
            entries = []
            for entry, saved, row in results:
                entries.append(entry)
                for k, v in saved.items():
                    bundle[entry["case_id"] + "/" + k] = v
                row["set"] = MODULE + "/" + set_name
                sink.write(json.dumps(row, sort_keys=True) + "\n")
            np.savez(os.path.join(outdir, "ops", "index.npz"), **bundle)
            manifest = {"schema_version": 1, "module": MODULE, "seed": args.seed, "jax_version": gg.JAX_VERSION, "x64": x64, "unrunnable": [], "cases": entries}
            with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
                fh.write(ec.canonical_dumps(manifest))
            gg.write_sha256sums(outdir)
            errors = sum(1 for e in entries if "eager_error" in e)
            sys.stdout.write(MODULE + " " + set_name + " " + str(len(entries)) + " cases, " + str(errors) + " eager errors\n")


if __name__ == "__main__":
    main()
