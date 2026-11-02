#!/usr/bin/env python3
"""Classify differing rows of a run_goldens log with tools/jax_defects.tsv; defect rows are counted apart from the rest, never dropped."""
import argparse
import collections
import csv
import os
import re
import sys

SAME = {"same", "both_raise"}


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
            return c["class"]
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--classes", default=os.path.join(here, "jax_defects.tsv"))
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
