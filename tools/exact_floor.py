import argparse
import collections
import os

import jit_eager_agreement
import run_goldens

SHADOW = "parity_jit_eager:"


def main():
    parser = argparse.ArgumentParser(
        description="turn OJAX_EXACT_LOG files into test/golden/exact_floor.tsv lines"
    )
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--goldens", default=None)
    args = parser.parse_args()
    root = jit_eager_agreement.ROOT
    goldens = args.goldens or os.environ.get("OJAX_GOLDENS") or run_goldens.find_goldens(root)
    ceiling = jit_eager_agreement.agreement(root, goldens)
    identical = collections.Counter()
    total = collections.Counter()
    for path in args.logs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5 or fields[4] in ("compare", "ratchet_selftest"):
                    continue
                total[fields[4]] += 1
                identical[fields[4]] += fields[2] == "0"
    for suite in sorted(total):
        floor = identical[suite]
        if suite.startswith(SHADOW):
            floor = min(floor, ceiling[suite[len(SHADOW) :]])
        print(f"{suite}\t{floor}\t{total[suite]}")


if __name__ == "__main__":
    main()
