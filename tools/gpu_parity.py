import argparse
import collections
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))


def write_cases(goldens, out, modulus, remainder, modes, suite="parity"):
    import numpy as np

    with open(out, "w", encoding="utf-8") as sink:
        for set_name in ("x64_off", "x64_on"):
            set_dir = os.path.join(goldens, suite, set_name)
            with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as fh:
                cases = json.load(fh)["cases"]
            for case in cases:
                if zlib.adler32(case["case_id"].encode("utf-8")) % modulus != remainder:
                    continue
                inputs = []
                single = os.path.join(set_dir, "inputs", case["case_id"] + ".npz")
                if os.path.exists(single):
                    source, prefix = single, ""
                else:
                    source = os.path.join(set_dir, "ops", case["op"] + ".npz")
                    prefix = case["case_id"] + "/in_"
                with np.load(source) as z:
                    for arg in case["args"]:
                        value = z[prefix + arg["name"]]
                        inputs.append(
                            {
                                "name": arg["name"],
                                "dtype": value.dtype.name,
                                "shape": [int(d) for d in value.shape],
                                "hex": value.tobytes().hex(),
                            }
                        )
                row = {
                    "set": set_name,
                    "case": case["case_id"],
                    "op": case["op"],
                    "params": case["params"],
                    "inputs": inputs,
                    "modes": modes,
                }
                sink.write(json.dumps(row, sort_keys=True) + "\n")


def reference(cases, out, shard, shards, device_inputs=False):
    sys.path.insert(0, HERE)
    import numpy as np
    import jax
    import parity_registry as registry

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                done.add((row["set"], row["case"], row["mode"]))
    with open(cases, encoding="utf-8") as fh, open(out, "a", encoding="utf-8") as sink:
        for index, line in enumerate(fh):
            if index % shards != shard:
                continue
            case = json.loads(line)
            jax.config.update("jax_enable_x64", case["set"] == "x64_on")
            names = [i["name"] for i in case["inputs"]]
            arrays = [
                np.frombuffer(bytes.fromhex(i["hex"]), dtype=np.dtype(i["dtype"])).reshape(i["shape"])
                for i in case["inputs"]
            ]
            op = registry.OPS[case["op"]]

            def fn(*xs, _op=op, _names=names, _params=case["params"]):
                return _op(**dict(zip(_names, xs)), p=_params)

            for mode in case["modes"]:
                if (case["set"], case["case"], mode) in done:
                    continue
                row = {"set": case["set"], "case": case["case"], "mode": mode}
                try:
                    if mode == "jit":
                        result = jax.jit(fn)(*arrays)
                    elif device_inputs:
                        result = fn(*[jax.numpy.asarray(a) for a in arrays])
                    else:
                        result = fn(*arrays)
                    outs = list(result) if isinstance(result, (tuple, list)) else [result]
                    row["outs"] = []
                    for i, value in enumerate(outs):
                        array = np.asarray(value, order="C")
                        row["outs"].append(
                            {
                                "name": "out" + str(i),
                                "dtype": array.dtype.name,
                                "shape": [int(d) for d in array.shape],
                                "hex": array.tobytes().hex(),
                            }
                        )
                except Exception as error:
                    row.pop("outs", None)
                    row["error"] = type(error).__name__ + ": " + str(error)
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()
            if index % 40 == 0:
                jax.clear_caches()


def load(paths):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(row["set"], row["case"], row["mode"])] = row
    return rows


WIDTH = {"float16": ("<H", 2), "float32": ("<I", 4), "float64": ("<Q", 8), "complex64": ("<I", 4), "complex128": ("<Q", 8)}


def ulp_gap(dtype, a, b):
    if dtype not in WIDTH:
        return None
    fmt, width = WIDTH[dtype]
    top = 1 << (8 * width - 1)

    def rank(v):
        return v if v < top else top - v

    gap = 0
    count = 0
    for i in range(0, len(a), width):
        x = struct.unpack_from(fmt, a, i)[0]
        y = struct.unpack_from(fmt, b, i)[0]
        if x != y:
            count += 1
            gap = max(gap, abs(rank(x) - rank(y)))
    return count, len(a) // width, gap


def classify(ref, ours):
    if "error" in ref or "error" in ours:
        if "error" in ref and "error" in ours:
            if ref["error"] == ours["error"]:
                return "both_raise", ""
            if ref["error"].split(":")[0] == ours["error"].split(":")[0]:
                return "both_raise_class", ""
            return "both_raise_other", ours["error"][:120]
        if "error" in ref:
            return "only_jax_raises", ref["error"][:120]
        return "only_ojax_raises", ours["error"][:160]
    if len(ref["outs"]) != len(ours["outs"]):
        return "arity", f'{len(ref["outs"])} vs {len(ours["outs"])}'
    details = []
    kind = "identical"
    for r, o in zip(ref["outs"], ours["outs"]):
        if r["dtype"] != o["dtype"]:
            kind = "dtype"
            details.append(f'{r["name"]} {r["dtype"]} vs {o["dtype"]}')
        elif r["shape"] != o["shape"]:
            kind = "shape"
            details.append(f'{r["name"]} {r["shape"]} vs {o["shape"]}')
        elif r["hex"] != o["hex"]:
            if kind == "identical":
                kind = "bytes"
            gap = ulp_gap(r["dtype"], bytes.fromhex(r["hex"]), bytes.fromhex(o["hex"]))
            details.append(f'{r["name"]} {r["dtype"]} {gap}')
    return kind, "; ".join(details)


def compare(ref_paths, our_paths, verbose):
    refs = load(ref_paths)
    ours = load(our_paths)
    totals = collections.Counter()
    families = collections.Counter()
    missing = 0
    rows = []
    for key in sorted(refs):
        if key not in ours:
            missing += 1
            continue
        kind, detail = classify(refs[key], ours[key])
        totals[(key[2], kind)] += 1
        if kind not in ("identical", "both_raise", "both_raise_class"):
            family = key[1].split("_")[0] + "_" + key[1].split("_")[1] if "_" in key[1] else key[1]
            families[(key[2], kind, family)] += 1
            rows.append(f"{key[2]}\t{key[0]}\t{key[1]}\t{kind}\t{detail}")
    if verbose:
        for row in rows:
            print(row[:400])
    print("reference rows", len(refs), "ojax rows", len(ours), "missing", missing)
    for key, count in sorted(totals.items()):
        print(key[0], key[1], count)
    for key, count in families.most_common(40):
        print("  ", key[0], key[1], key[2], count)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cases")
    p.add_argument("--goldens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modulus", type=int, default=15)
    p.add_argument("--remainder", type=int, default=0)
    p.add_argument("--modes", default="jit,eager")
    p.add_argument("--suite", default="parity")
    p = sub.add_parser("reference")
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--device-inputs", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--ours", nargs="+", required=True)
    p.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.command == "cases":
        write_cases(args.goldens, args.out, args.modulus, args.remainder, args.modes.split(","), args.suite)
    elif args.command == "reference":
        reference(args.cases, args.out, args.shard, args.shards, args.device_inputs)
    else:
        compare(args.ref, args.ours, args.verbose)


if __name__ == "__main__":
    main()
