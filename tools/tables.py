import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defect_classes
import jax_defects
import seed_defects

KINDS = ["a", "b", "c", "d", "c,d"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_listed(root, seed):
    listed = jax_defects.read_defects(os.path.join(root, jax_defects.DEFECTS))
    for extra in jax_defects.TRACK_DEFECTS:
        listed.update(jax_defects.read_defects(os.path.join(root, extra)))
    if seed is not None:
        listed.update(seed_defects.read_seed_defects(REPO, seed))
    return listed


def read_rows(logs):
    rows = {}
    for log in logs:
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 6:
                    rows[(f[4], f[0])] = f
    return rows


def is_ok(f):
    return f[5] in ("same", "both_raise")


def family(suite):
    return suite.split(":")[0]


def function_of(row):
    if row.startswith("compose_"):
        return "compose:" + row.split(":")[-1]
    return re.sub(r"_[0-9a-f]{8}_.*", "", row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=REPO)
    parser.add_argument("--seed", default=None, help="apply the rows of this checkout's test/golden/seed_defects.tsv proven on this seed")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("out")
    args = parser.parse_args()
    listed = load_listed(args.root, args.seed)
    classes = defect_classes.load_classes(os.path.join(args.root, "tools", "defect_classes.tsv"))

    def verdict(suite, row, f):
        if (suite, row) in listed:
            return ("defect", listed[(suite, row)], "list")
        if is_ok(f):
            return ("identical", "", "")
        c = defect_classes.classify(classes, suite, row, "\t".join(f[5:]))
        if c:
            category = next(x["category"] for x in classes if x["class"] == c)
            return ("defect", category, c)
        return ("differ", "", "")

    by_suite = collections.defaultdict(collections.Counter)
    named = []
    for (suite, row), f in sorted(read_rows(args.logs).items()):
        v, kind, source = verdict(suite, row, f)
        b = by_suite[suite]
        b["rows"] += 1
        if v == "defect":
            b["defect"] += 1
            b["defect_" + kind] += 1
            if is_ok(f):
                b["defect_identical_by_chance"] += 1
        else:
            b[v] += 1
        if f[5] == "same":
            b["same"] += 1
        if f[5] == "both_raise":
            b["both_raise"] += 1
        if not is_ok(f) and not suite.startswith("parity_jit_eager"):
            named.append([suite, row, v, (kind + " " + source).strip(), f[5]] + f[6:9])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "by_suite.tsv"), "w", encoding="utf-8") as o:
        o.write("suite\trows\tidentical\tsame\tboth_raise\tdefect\tdefect_a\tdefect_b\tdefect_c\tdefect_d\tdefect_cd\tdefect_identical_by_chance\tdiffer\n")
        for suite in sorted(by_suite):
            b = by_suite[suite]
            o.write("\t".join([suite] + [str(b[x]) for x in ["rows", "identical", "same", "both_raise", "defect"]] + [str(b["defect_" + k]) for k in KINDS] + [str(b["defect_identical_by_chance"]), str(b["differ"])]) + "\n")
    with open(os.path.join(args.out, "rows_named.tsv"), "w", encoding="utf-8") as w:
        w.write("suite\trow\tverdict\tdefect_kind_and_class\tlog_kind\tfields\n")
        for r in named:
            w.write("\t".join(r) + "\n")
    counts = collections.Counter((family(r[0]), r[2], r[3] if r[2] == "defect" else r[4].split(":")[0], function_of(r[1])) for r in named)
    with open(os.path.join(args.out, "by_kind.txt"), "w", encoding="utf-8") as w:
        for k, v in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], -x[1])):
            w.write(f"{v}\t" + "\t".join(k) + "\n")
    total = collections.Counter()
    for suite, b in by_suite.items():
        if not suite.startswith(("parity_jit_eager", "compare", "ratchet_selftest")):
            total.update(b)
    print("total (no shadow, compare, selftest):", dict(total))


if __name__ == "__main__":
    main()
