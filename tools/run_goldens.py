#!/usr/bin/env python3
import argparse
import collections
import fcntl
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import row_ratchet

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LISTED = re.compile(r"^(\S+)\s+(\d+)\s")


def slot_cores():
    spec = os.environ.get("OJAX_SLOT_CORES", "2-17")
    low, _, high = spec.partition("-")
    return list(range(int(low), int(high or low) + 1))


def try_slot(cores):
    directory = os.environ.get("OJAX_SLOT_DIR", "/tmp/ojax_slots")
    os.makedirs(directory, exist_ok=True)
    for core in cores:
        fd = os.open(os.path.join(directory, str(core)), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return core, fd
        except OSError:
            os.close(fd)
    return None


def checkout_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_checkout(root):
    common = subprocess.run(
        ["git", "-C", root, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return os.path.dirname(common) if common else root


def find_goldens(root):
    for base in (root, main_checkout(root)):
        path = os.path.join(base, "goldens")
        if os.path.isdir(path):
            return path
    sys.exit("no goldens directory found")


def list_cases(exe, cwd, env):
    out = subprocess.run([exe, "list"], cwd=cwd, env=env, capture_output=True, text=True).stdout
    cases = []
    for line in out.splitlines():
        m = LISTED.match(ANSI.sub("", line).strip())
        if m and m.group(1) != "exact_ratchet":
            cases.append((m.group(1), int(m.group(2))))
    return cases


def make_jobs(cases, count):
    size = max(1, len(cases) // count)
    jobs = []
    for start in range(0, len(cases), size):
        by_group = collections.OrderedDict()
        for group, index in cases[start : start + size]:
            by_group.setdefault(group, []).append(index)
        jobs.extend((g, min(v), max(v)) for g, v in by_group.items())
    return jobs


def read_floors(path):
    floors = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                floors[parts[0]] = int(parts[1])
    return floors


def tally(log):
    counts = collections.Counter()
    with open(log) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[2] == "0":
                counts[parts[4]] += 1
            if len(parts) >= 6 and parts[5] == "same":
                counts["raw:" + parts[4]] += 1
    return counts


def is_fuzz_suite(group):
    return group.startswith(("fuzz:", "fuzz_s2:", "option_fuzz:", "compose_fuzz:"))


def row_classes(log):
    groups = collections.defaultdict(collections.Counter)
    with open(log, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            suite = parts[4]
            if suite.startswith("parity_jit_eager:") or suite in ("compare", "ratchet_selftest"):
                continue
            if suite.startswith("fuzz:"):
                group = "fuzz"
            elif suite.startswith("fuzz_s2:"):
                group = "fuzz_s2"
            elif suite.startswith("option_fuzz:"):
                group = "option_fuzz"
            elif suite.startswith("text_option_fuzz:"):
                group = "option_fuzz_error_text"
            elif suite.startswith("compose_fuzz:"):
                group = "compose_fuzz"
            else:
                group = "fixed"
            counter = groups[group]
            counter["rows"] += 1
            counter["identical"] += parts[2] == "0"
            counter["raw"] += parts[5] == "same"
            counter["both_raise"] += parts[5] == "both_raise"
            counter["both_raise_type"] += parts[5].startswith("both_raise_type:")
            counter["both_raise_typed"] += parts[5] == "both_raise" and len(parts) > 6 and parts[6] == "typed"
            counter["both_raise_authored"] += parts[5] == "both_raise" and len(parts) > 6 and parts[6] == "authored"
            counter["empty"] += parts[1] == "0" and parts[5] == "same"
    return groups


def print_row_classes(log):
    print("set\trows\tidentical\traw\tboth_raise\t(typed\tauthored)\tboth_raise_type\tempty_arrays")
    groups = row_classes(log)
    for name in ("fixed", "fuzz", "fuzz_s2", "compose_fuzz", "option_fuzz", "option_fuzz_error_text"):
        c = groups.get(name)
        if c:
            print(
                f"{name}\t{c['rows']}\t{c['identical']}\t{c['raw']}\t{c['both_raise']}\t{c['both_raise_typed']}\t"
                f"{c['both_raise_authored']}\t{c['both_raise_type']}\t{c['empty']}"
            )


def float_nan_uses(root):
    uses = []
    for directory, _, files in os.walk(os.path.join(root, "src")):
        for name in files:
            if name.endswith(".ml"):
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as f:
                    if re.search(r"\bFloat\.nan\b", f.read()):
                        uses.append(os.path.relpath(path, root))
    return uses


def rows_by_name(log):
    rows = collections.defaultdict(list)
    with open(log, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                rows[(parts[4], parts[0])].append(line.rstrip("\n"))
    return {k: sorted(v) for k, v in rows.items()}


def order_differences(normal_log, log):
    normal = rows_by_name(normal_log)
    other = rows_by_name(log)
    differ = []
    for key in sorted(set(normal) | set(other)):
        if normal.get(key) != other.get(key):
            differ.append((key, normal.get(key), other.get(key)))
    return differ


def main():
    root = checkout_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--no-ratchet", action="store_true")
    parser.add_argument("--stall-timeout", type=float, default=0)
    parser.add_argument("--split-failed", action="store_true")
    parser.add_argument("--mem-limit-gb", type=float, default=0)
    parser.add_argument("--failed-log", default=None)
    parser.add_argument("--order", default="normal", help="normal, reverse or shuffle:SEED, applied to the cases of each job in one process")
    parser.add_argument("--compare-log", default=None, help="a log of the same cases in normal order; rows that differ are printed")
    args = parser.parse_args()

    cwd = os.path.join(root, "_build/default/test/golden")
    env = dict(os.environ, OJAX_GOLDENS=os.environ.get("OJAX_GOLDENS") or find_goldens(root))
    work = tempfile.mkdtemp(prefix="run_goldens_", dir=os.environ.get("TMPDIR"))
    exe = os.path.join(work, "manifest.exe")
    shutil.copy(os.path.join(cwd, "manifest.exe"), exe)

    cases = list_cases(exe, cwd, env)
    if not cases:
        sys.exit("no test cases listed")
    if args.filter:
        cases = [c for c in cases if re.search(args.filter, c[0])]
    jobs = make_jobs(cases, args.jobs * 2)

    def bound_fuzz_memory():
        limit = int((args.mem_limit_gb or 12) * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    started = time.time()
    running = []
    failed = []
    next_job = 0
    cores = slot_cores()
    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < args.jobs:
            slot = try_slot(cores)
            if slot is None:
                break
            group, low, high = jobs[next_job]
            scratch = os.path.join(work, str(next_job))
            os.makedirs(os.path.join(scratch, "a"))
            job_log = os.path.join(scratch, "exact.log")
            job_env = dict(env, OJAX_EXACT_LOG=job_log, OJAX_TEST_ORDER=args.order)
            out = open(os.path.join(scratch, "out"), "w")
            proc = subprocess.Popen(
                ["nice", "-n", "10", "taskset", "-c", str(slot[0]), exe, "test", "-o", os.path.join(scratch, "a"), "^" + re.escape(group) + "$", f"{low}..{high}"],
                cwd=cwd,
                env=job_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=bound_fuzz_memory if is_fuzz_suite(group) else None,
            )
            running.append((proc, next_job, out, slot[1], job_log, time.time()))
            next_job += 1
        for item in running[:]:
            is_fuzz = is_fuzz_suite(jobs[item[1]][0])
            stall = args.stall_timeout or (60 if is_fuzz else 0)
            if stall and item[0].poll() is None:
                moved = os.path.getmtime(item[4]) if os.path.exists(item[4]) else item[5]
                if time.time() - max(moved, item[5]) > stall:
                    item[0].kill()
            if item[0].poll() is not None:
                item[2].close()
                os.close(item[3])
                if item[0].returncode != 0:
                    group, low, high = jobs[item[1]]
                    if (args.split_failed or is_fuzz) and low < high:
                        middle = (low + high) // 2
                        jobs.append((group, low, middle))
                        jobs.append((group, middle + 1, high))
                        if os.path.exists(item[4]):
                            os.remove(item[4])
                    elif is_fuzz:
                        print(f"FUZZ LOST {group} {low}")
                    else:
                        failed.append(item[1])
                running.remove(item)
        time.sleep(0.05)

    log = os.path.abspath(args.log) if args.log else os.path.join(work, "exact.log")
    with open(log, "w") as merged:
        for index in range(len(jobs)):
            part = os.path.join(work, str(index), "exact.log")
            if os.path.exists(part):
                with open(part) as f:
                    merged.write(f.read())

    if args.failed_log:
        with open(args.failed_log, "w") as sink:
            for index in failed:
                with open(os.path.join(work, str(index), "out"), errors="replace") as f:
                    for line in f:
                        clean = ANSI.sub("", line)
                        if "[FAIL]" in clean or "[failure]" in clean or "[exception]" in clean:
                            sink.write(clean.strip()[:400] + "\n")

    for index in failed[:20]:
        group, low, high = jobs[index]
        print(f"FAILED {group} {low}..{high}")
        with open(os.path.join(work, str(index), "out"), errors="replace") as f:
            for line in f:
                clean = ANSI.sub("", line)
                if "[FAIL]" in clean or "[failure]" in clean:
                    print("  " + clean.strip()[:200])

    order_differ = []
    if args.compare_log:
        order_differ = order_differences(args.compare_log, log)
        for (suite, name), before, after in order_differ[:200]:
            print(f"ORDER {args.order} {suite} {name}")
            for line in before or ["(missing)"]:
                print("  normal: " + line[:300])
            for line in after or ["(missing)"]:
                print("  " + args.order + ": " + line[:300])
        print(f"order {args.order}: {len(order_differ)} rows differ from {args.compare_log}")

    counts = tally(log)
    fell = []
    if not args.no_ratchet and not args.filter:
        floors = read_floors(os.path.join(root, "test/golden/exact_floor.tsv"))
        ran = {c[0] for c in cases}
        ran |= {s.replace("parity_jit:", "parity_jit_eager:") for s in ran}
        ran |= {"raw:" + s for s in ran}
        fell = [(s, counts[s], f) for s, f in sorted(floors.items()) if s in ran and counts[s] < f]
    for suite, got, floor in fell:
        print(f"RATCHET {suite}: {got} bit-identical, floor {floor}")
    row_fell = 0
    if not args.no_ratchet and not args.filter:
        row_fell = row_ratchet.report(row_ratchet.row_failures(root, log))
        row_fell += row_ratchet.report(row_ratchet.run_text_check(root, env["OJAX_GOLDENS"], min(args.jobs, 8)))

    totals = collections.Counter()
    with open(log) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                totals[parts[4]] += 1
    shadow = [s for s in totals if s.startswith("parity_jit_eager:")]
    second_seed = [s for s in totals if s.startswith("fuzz_s2:")]
    option_rows = [s for s in totals if s.startswith(("option_fuzz:", "text_option_fuzz:"))]
    composed = [s for s in totals if s.startswith("compose_fuzz:")]
    uncounted = set(shadow) | set(second_seed) | set(option_rows) | set(composed) | {"compare", "ratchet_selftest"}
    identical = sum(n for s, n in counts.items() if s not in uncounted and not s.startswith("raw:"))
    rows = sum(n for s, n in totals.items() if s not in uncounted)
    raw_identical = sum(n for s, n in counts.items() if s.startswith("raw:") and s[4:] not in uncounted)
    print_row_classes(log)
    if second_seed:
        print(f"fuzz_s2, not in the totals below: {sum(counts[s] for s in second_seed)}/{sum(totals[s] for s in second_seed)} bit-identical, {sum(counts['raw:' + s] for s in second_seed)} in raw bytes")
    for s in sorted(option_rows):
        print(f"{s}, not in the totals below: {counts[s]}/{totals[s]} bit-identical, {counts['raw:' + s]} in raw bytes")
    if composed:
        print(f"compose_fuzz, not in the totals below: {sum(counts[s] for s in composed)}/{sum(totals[s] for s in composed)} bit-identical, {sum(counts['raw:' + s] for s in composed)} in raw bytes")
    print(f"raw npz bytes and dtype: {raw_identical}/{rows} identical")
    print(f"{len(cases)} tests, {len(jobs)} jobs, {time.time() - started:.0f}s, {identical}/{rows} bit-identical")
    if shadow:
        print(f"parity_jit against the eager goldens: {sum(counts[s] for s in shadow)}/{sum(totals[s] for s in shadow)} bit-identical, {sum(counts['raw:' + s] for s in shadow)} in raw bytes")
    shutil.rmtree(work, ignore_errors=True)
    nan_uses = float_nan_uses(root)
    if nan_uses:
        print("Float.nan is 0x7ff8000000000001, JAX's nan is 0x7ff8000000000000; use Float_format.nan in: " + ", ".join(nan_uses))
    sys.exit(1 if failed or fell or row_fell or nan_uses or order_differ else 0)


if __name__ == "__main__":
    main()
