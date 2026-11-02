import argparse
import json
import multiprocessing
import os
import shutil
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_text
import emit_common as ec
import gen_goldens as gg
import gen_index_goldens as gi
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core

warnings.filterwarnings("ignore")

MODULE = "apply_transforms"
SETS = (("x64_off", False), ("x64_on", True))
DTYPES = {"x64_off": ("float32", "int32", "complex64"), "x64_on": ("float64", "int64", "float32")}
FUNCS = {"negative": jnp.negative, "sin": jnp.sin, "square": lambda v: v * v}
FORMS = (
    [gi.iv(1)],
    [gi.sl(None, None, 2)],
    [gi.arr("i0")],
    [gi.arr("i0"), gi.sl(1, 4)],
    [gi.arr("m0")],
    [gi.iv(-1), gi.sl(1, 3)],
)
TRANSFORMS = ("vmap0", "vmap1", "vmap_idx", "vmap_both", "jit_vmap0", "grad", "jvp", "linearize", "vjp")
MODES = (None, "clip", "drop", "promise_in_bounds")
BATCH = 3
SHAPE = [4, 5]


def base_program(case):
    params = case["params"]
    names = sorted(case["arg_names"])
    fn = FUNCS[params["func"]]

    def f(*values):
        env = dict(zip(names, values))
        idx = gi.build_index(params["index"], env)
        kw = {}
        if "mode" in params:
            kw["mode"] = params["mode"]
        return env["x"].at[idx].apply(fn, **kw)

    return f


def program(case):
    params = case["params"]
    names = sorted(case["arg_names"])
    base = base_program(case)
    t = params["transform"]
    xi = names.index("x")

    def with_x(values, x):
        return tuple(values[:xi]) + (x,) + tuple(values[xi + 1:])

    if t in ("vmap0", "vmap1", "vmap_idx", "vmap_both"):
        return jax.vmap(base, in_axes=tuple(params["in_axes"]))
    if t == "jit_vmap0":
        return jax.jit(jax.vmap(base, in_axes=tuple(params["in_axes"])))
    if t == "grad":
        def f(*values):
            return jax.grad(lambda x: jnp.sum(base(*with_x(values, x))))(values[xi])
        return f
    if t == "jvp":
        def f(*values):
            x = values[xi]
            return jax.jvp(lambda x: base(*with_x(values, x)), (x,), (jnp.ones_like(x),))
        return f
    if t == "linearize":
        def f(*values):
            x = values[xi]
            y, lin = jax.linearize(lambda x: base(*with_x(values, x)), x)
            return y, lin(jnp.ones_like(x))
        return f
    if t == "vjp":
        def f(*values):
            x = values[xi]
            y, back = jax.vjp(lambda x: base(*with_x(values, x)), x)
            return (y,) + tuple(back(jnp.ones_like(y)))
        return f
    raise ValueError(t)


def plan():
    out = {}
    for set_name, _ in SETS:
        rng = np.random.default_rng([20261104, gi.digest(set_name)])
        cases, stash = [], {}
        n = 0
        for fi, form in enumerate(FORMS):
            has_i0 = any(item.get("name") == "i0" for item in form)
            for t in TRANSFORMS:
                if t in ("vmap_idx", "vmap_both") and not has_i0:
                    continue
                for di, dtype in enumerate(DTYPES[set_name]):
                    if t in ("grad", "linearize", "vjp") and not dtype.startswith("float"):
                        continue
                    func = ("negative", "sin", "square")[(fi + di + n) % 3]
                    if dtype.startswith("int") and func == "sin":
                        func = "square"
                    mode = MODES[n % len(MODES)]
                    names = ["x"] + [item["name"] for item in form if item["k"] == "arr"]
                    names = sorted(names)
                    batched_x = t in ("vmap0", "vmap_both", "jit_vmap0", "vmap1")
                    x_axis = 1 if t == "vmap1" else 0
                    in_axes = []
                    for nm in names:
                        if nm == "x":
                            in_axes.append(x_axis if batched_x else None)
                        elif nm == "i0":
                            in_axes.append(0 if t in ("vmap_idx", "vmap_both") else None)
                        else:
                            in_axes.append(None)
                    arrays = {}
                    for nm in names:
                        if nm == "i0":
                            oob = n % 5 == 4
                            base = np.asarray([0, 5, -1] if oob else [0, 2, 0], dtype="int32")
                            if t in ("vmap_idx", "vmap_both"):
                                arrays[nm] = np.stack([np.roll(base, k) for k in range(BATCH)])
                            else:
                                arrays[nm] = base
                        elif nm == "m0":
                            arrays[nm] = np.asarray([True, False, True, True])
                    shape = list(SHAPE)
                    if batched_x:
                        shape.insert(x_axis, BATCH)
                    params = {"op": "apply", "transform": t, "func": func, "index": form, "in_axes": in_axes}
                    if mode is not None:
                        params["mode"] = mode
                    cid = "apply_%04d" % n
                    n += 1
                    cases.append({"case_id": cid, "op": "apply_transform", "dtype": dtype, "shape": shape, "params": params, "arrays_seed": int(rng.integers(0, 2**31))})
                    stash[cid] = arrays
        out[set_name] = (cases, stash)
    return out


def run_case(case, arrays):
    rng = np.random.default_rng([case["arrays_seed"], 7])
    arrays = dict(arrays)
    arrays["x"] = gi.values_for(rng, case["dtype"], tuple(case["shape"]))
    names = sorted(arrays)
    case = dict(case, arg_names=names)
    values = [arrays[k] for k in names]
    entry = {k: case[k] for k in ("case_id", "op", "dtype", "shape", "params")}
    entry["args"] = gi.meta({k: arrays[k] for k in names})
    saved = {"in_" + k: arrays[k] for k in names}
    try:
        eager = gi.to_outputs(program(case)(*[jnp.asarray(v) for v in values]))
        entry["eager"] = gi.meta(eager)
        saved.update({"eager_" + k: v for k, v in eager.items()})
    except Exception as error:
        entry["eager_error"], entry["eager_message"] = gi.message(error)
    try:
        jitted = gi.to_outputs(jax.jit(program(case))(*values))
        entry["jit"] = gi.meta(jitted)
        saved.update({"jit_" + k: v for k, v in jitted.items()})
    except Exception as error:
        entry["jit_error"], entry["jit_message"] = gi.message(error)
    row = compare_text._reference_case(program(case), values, jax, jax_core)
    row.update({"case": case["case_id"]})
    return entry, saved, row


def work(job):
    x64, case, arrays = job
    jax.config.update("jax_enable_x64", x64)
    return run_case(case, arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    cases_by_set = plan()
    os.makedirs(os.path.join(base, "text"), exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    with open(os.path.join(base, "text", "reference.jsonl"), "w", encoding="utf-8") as sink:
        for set_name, x64 in SETS:
            cases, stash = cases_by_set[set_name]
            jobs = [(x64, c, stash[c["case_id"]]) for c in cases]
            with ctx.Pool(args.jobs, maxtasksperchild=100) as pool:
                results = pool.map(work, jobs, chunksize=4)
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
            manifest = {"schema_version": 1, "module": MODULE, "jax_version": gg.JAX_VERSION, "x64": x64, "unrunnable": [], "cases": entries}
            with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
                fh.write(ec.canonical_dumps(manifest))
            gg.write_sha256sums(outdir)
            errors = sum(1 for e in entries if "eager_error" in e)
            sys.stdout.write(MODULE + " " + set_name + " " + str(len(entries)) + " cases, " + str(errors) + " eager errors\n")


if __name__ == "__main__":
    main()
