import json
import os
import shutil
import sys
import zlib

import gen_goldens as gg
import emit_common as ec
import numpy as np
import jax
import parity_registry as registry

MODULE = "parity"


def case_id(op, inputs, params, dtype):
    shapes = "_".join("x".join(str(d) for d in v["shape"]) for _, v in sorted(inputs.items()))
    digest = zlib.adler32(json.dumps([inputs, params], sort_keys=True).encode("utf-8"))
    return op + "_" + shapes + "_" + format(digest, "08x") + "_" + dtype


def real_dtype(dtype):
    return {"complex64": "float32", "complex128": "float64"}.get(dtype, dtype)


def draw(rng, spec, dtype):
    shape = tuple(spec["shape"])
    kind = spec["kind"]
    if kind == "real":
        return rng.standard_normal(shape).astype(real_dtype(dtype))
    if kind == "positive":
        return (rng.uniform(0.1, 2.0, shape)).astype(real_dtype(dtype))
    if kind == "complex":
        cdt = {"float32": "complex64", "float64": "complex128"}[real_dtype(dtype)]
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(cdt)
    if kind == "int":
        return rng.integers(-4, 5, shape).astype("int32")
    if kind == "bool":
        return rng.integers(0, 2, shape).astype("bool")
    raise SystemExit("unknown input kind " + kind)


def expand():
    cases = []
    for op, inputs, params in registry.all_cases():
        for dtype in ["float32", "float64"]:
            cases.append(
                {
                    "case_id": case_id(op, inputs, params, dtype),
                    "op": op,
                    "dtype": dtype,
                    "inputs": inputs,
                    "params": params,
                    "primitive": "parity." + op,
                }
            )
    cases.sort(key=lambda c: c["case_id"])
    return cases


def out_meta(name, arr):
    meta = gg._linalg_out_meta(name, arr)
    return meta


def gen_set(cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        if x64 and c["op"] in registry.SINGLE_PRECISION_ONLY:
            continue
        rng = np.random.default_rng(zlib.adler32(c["case_id"].encode("utf-8")))
        arrays = {k: draw(rng, c["inputs"][k], c["dtype"]) for k in sorted(c["inputs"])}
        result = registry.OPS[c["op"]](**arrays, p=c["params"])
        outs = list(result) if isinstance(result, (tuple, list)) else [result]
        outs = [np.asarray(o) for o in outs]
        out_arrays = {"out" + str(i): o for i, o in enumerate(outs)}
        np.savez(os.path.join(outdir, "inputs", c["case_id"] + ".npz"), **arrays)
        np.savez(os.path.join(outdir, "outputs", c["case_id"] + ".npz"), **out_arrays)
        compare, tol, reason = gg._linalg_tol(outs[0].dtype.name)
        entry = {
            "case_id": c["case_id"],
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": [
                {
                    "name": k,
                    "shape": [int(d) for d in v.shape],
                    "dtype": v.dtype.name,
                }
                for k, v in sorted(arrays.items())
            ],
            "outputs": [out_meta("out" + str(i), o) for i, o in enumerate(outs)],
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": MODULE,
        "jax_version": gg.JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(manifest_cases)


def main():
    gg.preflight()
    cases = expand()
    with open(os.path.join(gg.ROOT, "spec", MODULE + ".cases.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps({"cases": cases}))
    base = os.path.join(gg.ROOT, "goldens", MODULE)
    n_off = gen_set(cases, False, os.path.join(base, "x64_off"))
    n_on = gen_set(cases, True, os.path.join(base, "x64_on"))
    sys.stdout.write(MODULE + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
