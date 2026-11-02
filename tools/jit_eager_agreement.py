#!/usr/bin/env python3
import argparse
import array
import ast
import json
import os
import re
import struct
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def needs_concrete_ops(root):
    with open(os.path.join(root, "test/golden/parity_cases.ml"), encoding="utf-8") as fh:
        source = fh.read()
    match = re.search(r"let needs_concrete op =\s*List\.mem op\s*\[(.*?)\]", source, re.S)
    if not match:
        sys.exit("needs_concrete not found in test/golden/parity_cases.ml")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def read_member(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as fh:
        raw = fh.read()
    if raw[:6] != b"\x93NUMPY":
        sys.exit("bad npy member in " + path)
    if raw[6] == 1:
        size = struct.unpack("<H", raw[8:10])[0]
        start = 10
    else:
        size = struct.unpack("<I", raw[8:12])[0]
        start = 12
    header = ast.literal_eval(raw[start : start + size].decode("latin1"))
    if header["fortran_order"]:
        sys.exit("fortran order member in " + path)
    return header["descr"], tuple(header["shape"]), raw[start + size :]


def as_doubles(descr, body):
    if descr == "<f8" or descr == "<c16":
        out = array.array("d")
        out.frombytes(body)
        return list(out)
    if descr == "<f4" or descr == "<c8":
        out = array.array("f")
        out.frombytes(body)
        return list(out)
    if descr == "<f2":
        return [v[0] for v in struct.iter_unpack("<e", body)]
    if descr == "<V2":
        return [struct.unpack("<f", b"\x00\x00" + body[i : i + 2])[0] for i in range(0, len(body), 2)]
    sys.exit("not a float descr " + descr)


def as_ints(descr, body):
    code = {"|b1": "<B", "|u1": "<B", "|i1": "<b", "<u2": "<H", "<i2": "<h", "<u4": "<I", "<i4": "<i", "<u8": "<q", "<i8": "<q"}
    return [v[0] for v in struct.iter_unpack(code[descr], body)]


def kind_of(descr):
    if descr in ("<c8", "<c16"):
        return "complex"
    if descr in ("<f2", "<f4", "<f8", "<V2"):
        return "float"
    return "int"


def same_bits(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def differing(left, right):
    left_descr, left_shape, left_body = left
    right_descr, right_shape, right_body = right
    if left_shape != right_shape or kind_of(left_descr) != kind_of(right_descr):
        return -1
    if kind_of(left_descr) == "int":
        return sum(a != b for a, b in zip(as_ints(left_descr, left_body), as_ints(right_descr, right_body)))
    return sum(not same_bits(a, b) for a, b in zip(as_doubles(left_descr, left_body), as_doubles(right_descr, right_body)))


def raw_differs(left, right):
    return left != right


def load_cases(goldens, module, mode):
    with open(os.path.join(goldens, module, mode, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def compare_mode(root, goldens, mode):
    concrete = needs_concrete_ops(root)
    jit_cases = {c["case_id"]: c for c in load_cases(goldens, "parity_jit", mode)}
    rows = []
    for case in load_cases(goldens, "parity", mode):
        entry = jit_cases[case["case_id"]]
        eager_path = os.path.join(goldens, "parity", mode, "outputs", case["case_id"] + ".npz")
        jit_path = os.path.join(goldens, "parity_jit", mode, "outputs", case["case_id"] + ".npz")
        for output in case["outputs"]:
            row = case["case_id"] + ":" + output["name"]
            if "jit_error" in entry:
                rows.append((row, "jit_error", 0, False))
            elif case["op"] in concrete:
                diff = differing(read_member(eager_path, output["name"]), read_member(jit_path, output["name"]))
                rows.append((row, "needs_concrete", diff, False))
            else:
                eager = read_member(eager_path, output["name"])
                jitted = read_member(jit_path, output["name"])
                rows.append((row, "compared", differing(eager, jitted), raw_differs(eager, jitted)))
    return rows


def agreement(root=ROOT, goldens=None):
    goldens = goldens or os.path.join(root, "goldens")
    counts = {}
    for mode in ("x64_off", "x64_on"):
        rows = compare_mode(root, goldens, mode)
        counts[mode] = sum(1 for _, source, diff, _ in rows if source != "compared" or diff == 0)
    return counts


def main():
    parser = argparse.ArgumentParser(description="count the parity_jit rows where JAX's jit and eager goldens are bit-identical")
    parser.add_argument("--goldens", default=None)
    parser.add_argument("--list", action="store_true", help="print every row where the two goldens differ")
    parser.add_argument("--log", default=None, help="an OJAX_EXACT_LOG file to check the row names against")
    args = parser.parse_args()
    goldens = args.goldens or os.path.join(ROOT, "goldens")
    for mode in ("x64_off", "x64_on"):
        rows = compare_mode(ROOT, goldens, mode)
        compared = [r for r in rows if r[1] == "compared"]
        agree = sum(1 for r in compared if r[2] == 0)
        errors = sum(1 for r in rows if r[1] == "jit_error")
        concrete = [r for r in rows if r[1] == "needs_concrete"]
        count = agree + errors + len(concrete)
        print(f"parity_jit_eager:{mode}\t{count}\t{len(rows)}")
        print(f"  compared bit by bit {len(compared)}, identical {agree}, differing {len(compared) - agree}")
        raw_agree = sum(1 for r in compared if not r[3])
        print(f"  compared as raw npz bytes {len(compared)}, identical {raw_agree}, differing {len(compared) - raw_agree}")
        print(f"  reference is the eager golden, so identical by construction: jit_error {errors}, needs_concrete {len(concrete)}")
        print(f"  needs_concrete rows whose jit golden exists and differs: {[r[0] for r in concrete if r[2] != 0]}")
        if args.log:
            suite = "parity_jit_eager:" + mode
            with open(args.log, encoding="utf-8") as fh:
                logged = {f[0] for f in (line.rstrip("\n").split("\t") for line in fh) if len(f) >= 5 and f[4] == suite}
            names = {r[0] for r in rows}
            print(f"  row names against {os.path.basename(args.log)}: {len(logged)} logged, only here {sorted(names - logged)}, only there {sorted(logged - names)}")
        if args.list:
            for row, _, diff, raw in compared:
                if diff != 0 or raw:
                    print(f"  differs\t{row}\t{diff}\traw={raw}")


if __name__ == "__main__":
    main()
