#!/usr/bin/env python3
"""Split the both-raise rows of the parity, coverage, apicov and fuzz text references into identical, JAX defect with proof and differing."""
import argparse
import collections
import glob
import json
import os
import re
import sys

FIELDS = ("text_error", "jaxpr_error", "structure_error")
TRACED_AT = re.compile(r"while tracing the function (\S+) at (\S+:\d+) for ")
FROM_LINE = re.compile(r"\n    from line (.*)")
SETS = ("parity", "coverage", "apicov", "fuzz")
LOC_CALLSITE = re.compile(r'at loc\("[^"]*"\(callsite\(')


def load(paths):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["set"], r["case"])] = r
    return rows


def python_locations(message):
    return [m.group(2) for m in TRACED_AT.finditer(message)] + [m.group(1) for m in FROM_LINE.finditer(message)]


def without_locations(message):
    message = TRACED_AT.sub(lambda m: f"while tracing the function {m.group(1)} at <loc> for ", message)
    return FROM_LINE.sub("\n    from line <loc>", message)


CALLBACK = re.compile(r"callback=_FlatCallback\(callback_func=<function (\S+) at 0x[0-9A-Fa-fX]+>, in_tree=PyTreeDef\(.*\)\)$")


def without_callsites(message):
    out, i = [], 0
    while True:
        m = LOC_CALLSITE.search(message, i)
        if m is None:
            out.append(message[i:])
            return "".join(out)
        start = message.index('"(callsite(', m.start()) + 1
        out.append(message[i:start])
        depth, j, quoted = 0, start, False
        while j < len(message):
            ch = message[j]
            if ch == '"':
                quoted = not quoted
            elif not quoted and ch == "(":
                depth += 1
            elif not quoted and ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1


def callsite_files(message):
    return sorted(set(re.findall(r'\("(/[^"]+\.py)":\d+', message)))


def callback_repr_only(jax_text, ojax_text):
    a, b = jax_text.splitlines(), ojax_text.splitlines()
    if len(a) != len(b):
        return None
    names = []
    for x, y in zip(a, b):
        if x == y:
            continue
        m = CALLBACK.search(x.strip())
        if m is None or y.strip() != "callback=<host callback>" or x[: len(x) - len(x.lstrip())] != y[: len(y) - len(y.lstrip())]:
            return None
        names.append(m.group(1))
    return names or None


def classify(ref, ours):
    out = []
    for key, r in sorted(ref.items()):
        if key[0].split("/")[0] not in SETS:
            continue
        o = ours.get(key, {})
        if "jaxpr" in r and "jaxpr" in o and r["jaxpr"] != o["jaxpr"]:
            names = callback_repr_only(r["jaxpr"], o["jaxpr"])
            if names:
                proof = (
                    "the only differing jaxpr line is the callback parameter, which JAX prints as the repr of a Python object: "
                    + "_FlatCallback(callback_func=<function " + names[0] + " at 0x...>, in_tree=...), a Python function's qualified name in the generator "
                    + "(tools/apicov_registry.py) and a CPython object address; ojax's callback is an OCaml closure, printed <host callback>"
                )
                out.append((key, "jaxpr", "defect", "c", proof))
    for key, r in sorted(ref.items()):
        if key[0].split("/")[0] not in SETS:
            continue
        o = ours.get(key, {})
        for field in FIELDS:
            if field not in r or field not in o:
                continue
            jax_text, ojax_text = r[field], o[field]
            if jax_text == ojax_text:
                out.append((key, field, "identical", "", ""))
                continue
            theirs = python_locations(jax_text)
            if theirs and without_locations(jax_text) == without_locations(ojax_text):
                mine = python_locations(ojax_text)
                proof = (
                    "JAX names the Python source location of the traced function ("
                    + ", ".join(theirs)
                    + ", the reference generator's own file) in its ConcretizationTypeError or tracing note; ojax prints "
                    + (", ".join(mine) if mine else "no location")
                    + " there, and the texts are equal once the location after 'while tracing the function <name> at' and 'from line' is removed"
                )
                out.append((key, field, "defect", "c", proof))
            elif LOC_CALLSITE.search(jax_text) and without_callsites(jax_text) == ojax_text:
                proof = (
                    "JAX's MLIR verifier diagnostic carries a loc(...) callsite chain of Python frames ("
                    + ", ".join(callsite_files(jax_text))
                    + ": the generator's and JAX's Python files, lines, columns and Python function names); "
                    + "ojax prints the same loc without a callsite, and the texts are equal once the callsite chain after the quoted loc name is removed"
                )
                out.append((key, field, "defect", "c", proof))
            else:
                out.append((key, field, "differ", "", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True, help="reference jsonl files (text/reference.jsonl, coverage/text_ref.jsonl, apicov/text_ref.jsonl, fuzz/text/reference.jsonl)")
    ap.add_argument("--ours", nargs="+", required=True, help="text_dump output jsonl files")
    ap.add_argument("--write", help="write the defect rows as a tsv")
    args = ap.parse_args()
    ref = load(args.ref)
    ours = load([p for pattern in args.ours for p in glob.glob(pattern)])
    rows = classify(ref, ours)
    counts = collections.Counter((key[0].split("/")[0], field, verdict) for key, field, verdict, _, _ in rows)
    for (group, field, verdict), n in sorted(counts.items()):
        sys.stdout.write(f"{group}\t{field}\t{verdict}\t{n}\n")
    for key, field, verdict, _, _ in rows:
        if verdict == "differ":
            sys.stdout.write(f"differ\t{key[0]}\t{key[1]}\t{field}\n")
    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write("set\tcase\tfield\tkind\tproof\n")
            for key, field, verdict, kind, proof in rows:
                if verdict == "defect":
                    fh.write(f"{key[0]}\t{key[1]}\t{field}\t{kind}\t{proof}\n")


if __name__ == "__main__":
    main()
