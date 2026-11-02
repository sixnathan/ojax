#!/usr/bin/env python3
import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_goldens

SUITES = ("fuzz:x64_off", "fuzz:x64_on")
CODES = {-1: "ojax_raises", -2: "jax_raises", -3: "dtype", -4: "shape", -5: "arity", -6: "size", -7: "lost", -8: "both_raise_type"}
VARIANT = re.compile(r"_(edge|wide|shape_small|shape_other)$")
OP = re.compile(r"^(.+?)_(?:[0-9x_]*)[0-9a-f]{8}_float(?:32|64)_")


def op_of(case):
    match = OP.match(case)
    return match.group(1) if match else case


def expected_rows(goldens):
    rows = {}
    for suite in SUITES:
        path = os.path.join(goldens, "fuzz", suite.split(":")[1], "manifest.json")
        with open(path, encoding="utf-8") as fh:
            for case in json.load(fh)["cases"]:
                for mode in ("eager", "jit"):
                    outs = [o["name"] for o in case.get(mode, [])] if mode in case else ["both_raise"]
                    rows[(suite, case["case_id"], mode)] = outs
    return rows


def parse_log(path):
    seen = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or not fields[4].startswith("fuzz:"):
                continue
            case, mode, out = fields[0].rsplit(":", 2)
            seen[(fields[4], case, mode, out)] = fields
    return seen


def report(goldens, log, out_path, top):
    expected = expected_rows(goldens)
    seen = parse_log(log)
    logged = collections.Counter((s, c, m) for s, c, m, _ in seen)
    per_op = collections.Counter()
    per_class = collections.Counter()
    per_variant = collections.Counter()
    total = identical = both_raise = 0
    rows = []
    for key, outs in sorted(expected.items()):
        suite, case, mode = key
        if logged[key] == 0:
            if mode == "jit" and outs == ["both_raise"]:
                continue
            total += 1
            rows.append((suite, case, mode, "lost", -7, 0, "lost"))
            per_op[op_of(case)] += 1
            per_class["lost"] += 1
            per_variant[VARIANT.search(case).group(1)] += 1
    for (suite, case, mode, out), fields in sorted(seen.items()):
        total += 1
        differing = int(fields[2])
        cls = fields[5] if len(fields) > 5 else ""
        if differing == 0:
            identical += 1
            both_raise += out == "both_raise"
            continue
        rows.append((suite, case, mode, out, differing, fields[3], cls))
        per_op[op_of(case)] += 1
        for name in (cls.split(":")[0].split(",") if cls else [CODES.get(differing, "?")]):
            per_class[name] += 1
        per_variant[VARIANT.search(case).group(1)] += 1
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    sys.stdout.write(f"fuzz rows {total}, identical {identical} (of which both raise {both_raise}), differing {len(rows)}\n")
    sys.stdout.write("by class: " + json.dumps(per_class.most_common()) + "\n")
    sys.stdout.write("by variant: " + json.dumps(per_variant.most_common()) + "\n")
    sys.stdout.write("by operation:\n")
    for op, count in per_op.most_common(top):
        sys.stdout.write(f"  {count} {op}\n")


def text_report(goldens, directory, jobs):
    os.makedirs(directory, exist_ok=True)
    exe = os.path.join(ROOT, "_build/default/tools/text_dump/text_dump.exe")
    workers = [
        subprocess.Popen(
            ["nice", "-n", "10", exe, goldens, os.path.join(directory, str(i) + ".jsonl"), str(i), str(jobs), "fuzz"],
            cwd=os.path.dirname(exe),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for i in range(jobs)
    ]
    for worker in workers:
        worker.wait()
    command = [
        sys.executable,
        "-I",
        os.path.join(HERE, "compare_text.py"),
        "compare",
        "--ref",
        os.path.join(goldens, "fuzz", "text", "reference.jsonl"),
        "--ours",
    ] + [os.path.join(directory, str(i) + ".jsonl") for i in range(jobs)] + ["--out", os.path.join(directory, "report.json"), "--top", "12"]
    subprocess.run(command, cwd=directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--log", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--stall-timeout", type=float, default=120)
    args = parser.parse_args()
    goldens = os.environ.get("OJAX_GOLDENS") or run_goldens.find_goldens(ROOT)
    if args.text:
        text_report(goldens, os.path.abspath(args.text), args.jobs)
        return
    status = 0
    if not args.report_only:
        command = [
            sys.executable,
            os.path.join(HERE, "run_goldens.py"),
            "--jobs",
            str(args.jobs),
            "--filter",
            args.filter or "^fuzz:",
            "--log",
            args.log,
            "--stall-timeout",
            str(args.stall_timeout),
        ]
        status = subprocess.run(command).returncode
    report(goldens, args.log, args.rows, args.top)
    floors = run_goldens.read_floors(os.path.join(ROOT, "test/golden/exact_floor.tsv"))
    counts = run_goldens.tally(args.log)
    for suite in SUITES:
        if suite in floors and counts[suite] < floors[suite]:
            sys.stdout.write(f"RATCHET {suite}: {counts[suite]} bit-identical, floor {floors[suite]}\n")
            status = status or 1
    sys.exit(status)


if __name__ == "__main__":
    main()
