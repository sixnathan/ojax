import argparse
import collections


def main():
    parser = argparse.ArgumentParser(
        description="turn OJAX_EXACT_LOG files into test/golden/exact_floor.tsv lines"
    )
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    identical = collections.Counter()
    total = collections.Counter()
    for path in args.logs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5 or fields[4] == "compare":
                    continue
                total[fields[4]] += 1
                identical[fields[4]] += fields[2] == "0"
    for suite in sorted(total):
        print(f"{suite}\t{identical[suite]}\t{total[suite]}")


if __name__ == "__main__":
    main()
