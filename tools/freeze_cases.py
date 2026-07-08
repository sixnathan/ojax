import json
import os
import sys

import emit_common as ec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_source(module):
    path = os.path.join(ROOT, "tools", "cases", module + ".json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def freeze_jaxpr(module):
    records = load_source(module)
    cases = sorted(records, key=lambda c: c["case_id"])
    doc = {
        "schema_version": 1,
        "module": module,
        "num_generated_cases": ec.NUM,
        "cases": cases,
    }
    dst = os.path.join(ROOT, "spec", module + ".cases.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(doc))
    return dst, len(cases)


def freeze(module):
    if module in ("jaxpr", "ad", "batching"):
        return freeze_jaxpr(module)
    records = load_source(module)
    explicit = []
    generated = []
    for rec in records:
        expanded = ec.expand_record(rec)
        if "args" in rec:
            explicit.extend(expanded)
        else:
            generated.extend(expanded)
    chosen = explicit + ec.sample(generated)
    chosen.sort(key=ec.sort_key)
    cases = ec.assign_ids(chosen)
    doc = {
        "schema_version": 1,
        "module": module,
        "num_generated_cases": ec.NUM,
        "cases": cases,
    }
    dst = os.path.join(ROOT, "spec", module + ".cases.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(doc))
    return dst, len(cases)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: freeze_cases.py <module>")
    dst, n = freeze(sys.argv[1])
    sys.stdout.write(dst + " " + str(n) + "\n")


if __name__ == "__main__":
    main()
