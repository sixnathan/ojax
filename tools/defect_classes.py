#!/usr/bin/env python3
"""Classify differing rows of a run_goldens log with tools/defect_classes.tsv; defect rows are counted apart from the rest, never dropped."""
import argparse
import collections
import csv
import os
import re
import sys

SAME = {"same", "both_raise"}
PRECISION_NAMES = {"highest", "float32", "high", "bfloat16_3x", "tensorfloat32", "default", "bfloat16", "fastest"}
DTYPE_LIST = re.compile(r"subtypes of ([A-Za-z_]+(?:, [A-Za-z_]+)+)")
PRECISION_GOT = re.compile(r"but got \(?'?([^',()]+?)'?(?:, Precision\.DEFAULT\))?\.$")


def sorted_dtype_lists(text):
    return DTYPE_LIST.sub(lambda m: "subtypes of " + ", ".join(sorted(m.group(1).split(", "))), text)


def dtype_order_only(rest):
    parts = rest.split("\t")
    return len(parts) >= 3 and parts[1] != parts[2] and sorted_dtype_lists(parts[1]) == sorted_dtype_lists(parts[2])


def unknown_precision_string(rest):
    parts = rest.split("\t")
    got = PRECISION_GOT.search(parts[1].replace("\\n", "\n")) if len(parts) >= 2 else None
    return got is not None and got.group(1) not in PRECISION_NAMES


PRECISION_PAIR = re.compile(r"but got \(\('([a-z_0-9]+)', '([a-z_0-9]+)'\), Precision\.DEFAULT\)\.$")


def precision_pair_as_enums(rest):
    parts = rest.split("\t")
    if len(parts) < 3:
        return False
    got = PRECISION_PAIR.search(parts[1])
    if got is None or not {got.group(1), got.group(2)} <= PRECISION_NAMES:
        return False
    enums = PRECISION_PAIR.sub(lambda m: f"but got ((Precision.{m.group(1).upper()}, Precision.{m.group(2).upper()}), Precision.DEFAULT).", parts[1])
    return enums == parts[2]


CHECKS = {"accepted_dtypes_order": dtype_order_only, "invalid_precision_string": unknown_precision_string, "precision_pair_repr": precision_pair_as_enums}


def load_classes(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["suite_re"] = re.compile(r["suite"])
        r["row_re"] = re.compile(r["row"])
        r["text_re"] = re.compile(r["jax_text"])
    return rows


def classify(classes, suite, row, rest):
    for c in classes:
        if c["suite_re"].search(suite) and c["row_re"].search(row) and c["text_re"].search(rest):
            check = CHECKS.get(c["class"])
            if check is None or check(rest):
                return c["class"]
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--classes", default=os.path.join(here, "defect_classes.tsv"))
    parser.add_argument("--list", action="store_true", help="print every differing row with its class")
    args = parser.parse_args()
    classes = load_classes(args.classes)
    counts = collections.Counter()
    by_suite = collections.Counter()
    with open(args.log, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[5] in SAME:
                continue
            suite, row, rest = parts[4], parts[0], "\t".join(parts[5:])
            name = classify(classes, suite, row, rest) or "not_a_defect"
            counts[name] += 1
            by_suite[(suite.split(":")[0], name)] += 1
            if args.list:
                print(f"{name}\t{suite}\t{row}")
    for name, n in sorted(counts.items()):
        print(f"{name}\t{n}", file=sys.stderr if args.list else sys.stdout)
    for (suite, name), n in sorted(by_suite.items()):
        print(f"{suite}\t{name}\t{n}", file=sys.stderr if args.list else sys.stdout)


if __name__ == "__main__":
    main()
