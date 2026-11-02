#!/usr/bin/env python3
"""Regenerate only the goldens that compares_at_single_precision narrowed. Run from outside the repo with reference JAX."""
import argparse
import json
import os
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import emit_common as ec
import gen_fuzz_goldens as fuzz
import gen_goldens as gg
import gen_parity_jit_goldens as gpj
import jax
import numpy as np
import parity_registry as registry

NATIVE = ("sp_owens_t", "st_norm_ppf", "st_norm_isf")


def affected(op):
    return "_" in op and op.split("_", 1)[1] in NATIVE


def saved_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def bytes_of(path):
    with open(path, "rb") as fh:
        return fh.read()


def write_if_changed(path, members, changed):
    with np.load(path) as stored:
        same = sorted(stored.files) == sorted(members) and all(saved_equal(stored[k], members[k]) for k in members)
    if not same:
        np.savez(path, **members)
        changed.append(path)


def regen_parity(goldens, set_name, x64, changed):
    jax.config.update("jax_enable_x64", x64)
    set_dir = os.path.join(goldens, "parity", set_name)
    jit_dir = os.path.join(goldens, "parity_jit", set_name)
    manifest_path = os.path.join(set_dir, "manifest.json")
    jit_manifest_path = os.path.join(jit_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(jit_manifest_path, encoding="utf-8") as fh:
        jit_manifest = json.load(fh)
    jit_entries = {e["case_id"]: e for e in jit_manifest["cases"]}
    rows = 0
    for case in manifest["cases"]:
        if not affected(case["op"]):
            continue
        rows += 1
        with np.load(os.path.join(set_dir, "inputs", case["case_id"] + ".npz")) as stored:
            arrays = {k: stored[k] for k in stored.files}
        names = sorted(arrays)
        result = registry.OPS[case["op"]](**arrays, p=case["params"])
        eager = {"out" + str(i): np.asarray(o, order="C") for i, o in enumerate(result if isinstance(result, (tuple, list)) else [result])}
        write_if_changed(os.path.join(set_dir, "outputs", case["case_id"] + ".npz"), eager, changed)
        outs = list(eager.values())
        compare, tol, reason = gg._linalg_tol(outs[0].dtype.name)
        import gen_parity_goldens as gp

        case["outputs"] = [gp.out_meta(k, v) for k, v in eager.items()]
        case["compare"] = compare
        case["tol"] = tol
        case.pop("tol_reason", None)
        if reason is not None:
            case["tol_reason"] = reason
        entry = jit_entries[case["case_id"]]
        jitted = gpj.under_jit(case["op"], names, case["params"])(*[arrays[k] for k in names])
        outs = {"out" + str(i): np.asarray(o, order="C") for i, o in enumerate(jitted if isinstance(jitted, (tuple, list)) else [jitted])}
        write_if_changed(os.path.join(jit_dir, "outputs", case["case_id"] + ".npz"), outs, changed)
        entry["outputs"] = [{"name": k, "dtype": v.dtype.name, "shape": [int(d) for d in v.shape]} for k, v in outs.items()]
    for path, doc, directory in ((manifest_path, manifest, set_dir), (jit_manifest_path, jit_manifest, jit_dir)):
        text = ec.canonical_dumps(doc)
        if text.encode("utf-8") != bytes_of(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            changed.append(path)
        gg.write_sha256sums(directory)
    return rows


def regen_fuzz(root, set_name, x64, changed):
    jax.config.update("jax_enable_x64", x64)
    set_dir = os.path.join(root, set_name)
    path = os.path.join(set_dir, "manifest.json")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    by_op = {}
    for index, entry in enumerate(manifest["cases"]):
        if affected(entry["op"]):
            by_op.setdefault(entry["op"], []).append(index)
    rows = 0
    for op, indices in sorted(by_op.items()):
        bundle_path = os.path.join(set_dir, "ops", op + ".npz")
        bundle = {}
        with np.load(bundle_path) as stored:
            original = {k: stored[k] for k in stored.files}
        for index in indices:
            entry = manifest["cases"][index]
            case_id = entry["case_id"]
            arrays = {a["name"]: original[case_id + "/in_" + a["name"]] for a in entry["args"]}
            new_entry, saved = fuzz.run_case(entry, arrays)
            manifest["cases"][index] = new_entry
            for member, value in saved.items():
                bundle[case_id + "/" + member] = value
            rows += 1
        write_if_changed(bundle_path, bundle, changed)
    text = ec.canonical_dumps(manifest)
    if text.encode("utf-8") != bytes_of(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        changed.append(path)
    gg.write_sha256sums(set_dir)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--fuzz-dirs", nargs="*", default=["fuzz", "fuzz_s2"])
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()
    changed = []
    for set_name, x64 in (("x64_off", False), ("x64_on", True)):
        print("parity", set_name, regen_parity(args.goldens, set_name, x64, changed))
        for name in args.fuzz_dirs:
            root = os.path.join(args.goldens, name)
            if os.path.isdir(root):
                print(name, set_name, regen_fuzz(root, set_name, x64, changed))
    relative = sorted(os.path.relpath(p, args.goldens) for p in changed)
    if args.manifest_out:
        with open(args.manifest_out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(relative) + "\n")
    print(len(relative), "files changed")


if __name__ == "__main__":
    main()
