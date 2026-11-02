#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREFIXES = ("fuzz:", "fuzz_s2:", "compose_fuzz:", "text_compose_fuzz:", "custom_ad:", "index_fuzz:", "text_index_fuzz:", "apply_transforms:", "text_apply_transforms:", "keys_arrays:", "jit_opts:", "jit_opts_notices:", "user_dtype:", "user_dtype_notices:", "docs_p2:", "option_fuzz:", "text_option_fuzz:", "option_pinned:", "text_option_pinned:", "put_modes:", "text_put_modes:", "text_fuzz:", "text_fuzz_s2:", "coverage:")
BASELINE_DIR = os.path.join("test", "golden", "identical_rows")
DROPPED = "dropped.tsv"
TEXT_GROUPS = {"parity": "text_parity", "jaxpr": "text_parity", "coverage": "text_coverage", "apicov": "text_apicov", "fuzz": "text_fuzz"}
DETAIL_GROUPS = {"text_errors": ("text_error", "jaxpr_error", "structure_error"), "text_structure": ("structure",)}
FIELDS = dict({name: ("text", "jaxpr") for name in TEXT_GROUPS.values()}, **DETAIL_GROUPS)
SHOWN = 40


def is_ratcheted(suite):
    return suite.startswith(PREFIXES)


def baseline_path(root, suite):
    return os.path.join(root, BASELINE_DIR, suite.replace(":", "_") + ".tsv")


def read_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row, raw = line.rstrip("\n").split("\t")
            rows[row] = raw == "1"
    return rows


def read_log(path):
    suites = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 6 and is_ratcheted(fields[4]):
                suites.setdefault(fields[4], {})[fields[0]] = (fields[2] == "0", fields[5] == "same")
    return suites


def row_failures(root, log, complete=True):
    failures = []
    for suite, seen in sorted(read_log(log).items()):
        path = baseline_path(root, suite)
        if not os.path.exists(path):
            failures.append(f"{suite}: no baseline {os.path.relpath(path, root)}")
            continue
        for row, raw in sorted(read_rows(path).items()):
            got = seen.get(row)
            if got is None:
                if complete:
                    failures.append(f"{suite} {row}: did not run")
            elif not got[0]:
                failures.append(f"{suite} {row}: no longer identical")
            elif raw and not got[1]:
                failures.append(f"{suite} {row}: no longer identical in raw bytes")
    return failures


def load_jsonl(paths):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                rows[(row["set"], row["case"])] = row
    return rows


def verdict(ref, ours, field):
    if ours is None:
        return "-"
    if field in ref and field in ours:
        return "i" if ref[field] == ours[field] else "-"
    if field not in ref and field not in ours:
        return "b"
    return "-"


def error_verdict(ref, ours, field):
    if field not in ref:
        return "n"
    if ours is None or field not in ours:
        return "-"
    return "i" if ref[field] == ours[field] else "-"


def reference_paths(goldens):
    return [
        os.path.join(goldens, "text", "reference.jsonl"),
        os.path.join(goldens, "coverage", "text_ref.jsonl"),
        os.path.join(goldens, "apicov", "text_ref.jsonl"),
        os.path.join(goldens, "fuzz", "text", "reference.jsonl"),
    ]


def dump_text(root, goldens, out, jobs):
    exe = os.path.join(root, "_build", "default", "tools", "text_dump", "text_dump.exe")
    if not os.path.exists(exe):
        sys.exit("row ratchet: build tools/text_dump/text_dump.exe first")
    os.makedirs(out, exist_ok=True)
    procs = []
    logs = []
    paths = []
    for tail, name in (([], "fixed"), (["fuzz"], "fuzz")):
        for shard in range(jobs):
            path = os.path.join(out, f"{name}{shard}.jsonl")
            paths.append(path)
            sink = open(path + ".out", "w")
            logs.append(sink)
            procs.append(
                subprocess.Popen(
                    ["nice", "-n", "10", exe, goldens, path, str(shard), str(jobs)] + tail,
                    cwd=os.path.dirname(exe),
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                )
            )
    for proc in procs:
        proc.wait()
    for sink in logs:
        sink.close()
    return paths


def text_verdicts(goldens, ours):
    refs = load_jsonl(reference_paths(goldens))
    mine = load_jsonl(ours)
    groups = {}
    for key, ref in refs.items():
        name = TEXT_GROUPS[key[0].split("/")[0]]
        got = mine.get(key)
        groups.setdefault(name, {})[key] = (verdict(ref, got, "text"), verdict(ref, got, "jaxpr"))
        errors = DETAIL_GROUPS["text_errors"]
        if any(field in ref for field in errors):
            groups.setdefault("text_errors", {})[key] = tuple(error_verdict(ref, got, f) for f in errors)
        groups.setdefault("text_structure", {})[key] = (verdict(ref, got, "structure"),)
    return groups


def read_text_baseline(root, name):
    rows = {}
    with open(os.path.join(root, BASELINE_DIR, name + ".tsv"), encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 + len(FIELDS[name]):
                sys.exit(f"row ratchet: bad line in {name}.tsv: {line.rstrip()}")
            rows[(fields[0], fields[1])] = tuple(fields[2:])
    return rows


def text_failures(root, goldens, ours):
    current = text_verdicts(goldens, ours)
    failures = []
    for name in sorted(FIELDS):
        path = os.path.join(root, BASELINE_DIR, name + ".tsv")
        if not os.path.exists(path):
            failures.append(f"{name}: no baseline {os.path.relpath(path, root)}")
            continue
        for key, wanted in sorted(read_text_baseline(root, name).items()):
            got = current.get(name, {}).get(key, ("-",) * len(FIELDS[name]))
            for field, want, have in zip(FIELDS[name], wanted, got):
                if want not in ("-", "n") and want != have:
                    failures.append(f"{name} {key[0]} {key[1]} {field}: was {want}, now {have}")
    return failures, current


def report(failures):
    for line in failures[:SHOWN]:
        print("ROW RATCHET " + line)
    if len(failures) > SHOWN:
        print(f"ROW RATCHET and {len(failures) - SHOWN} more")
    return len(failures)


def run_text_check(root, goldens, jobs):
    for path in reference_paths(goldens):
        if not os.path.exists(path):
            return [f"text: reference {path} is missing"]
    with tempfile.TemporaryDirectory(prefix="text_ratchet_", dir=os.environ.get("TMPDIR")) as out:
        ours = dump_text(root, goldens, out, jobs)
        failures, current = text_failures(root, goldens, ours)
    print_text_counts(current)
    return failures


def print_text_counts(current):
    for name, rows in sorted(current.items()):
        if name in DETAIL_GROUPS:
            continue
        text = sum(1 for r in rows.values() if r[0] == "i")
        jaxpr = sum(1 for r in rows.values() if r[1] == "i")
        both = sum(1 for r in rows.values() if r[0] == "b")
        print(f"{name}: {len(rows)} cases, text identical {text}, jaxpr identical {jaxpr}, text both raise {both}")
    by_group = {}
    for key, verdicts in current.get("text_errors", {}).items():
        counts = by_group.setdefault(key[0].split("/")[0], {})
        for field, v in zip(DETAIL_GROUPS["text_errors"], verdicts):
            if v != "n":
                tally = counts.setdefault(field, [0, 0])
                tally[0] += 1
                tally[1] += v == "i"
    for group, counts in sorted(by_group.items()):
        parts = ", ".join(f"{field} {c[1]} of {c[0]}" for field, c in sorted(counts.items()))
        print(f"text_errors {group}: JAX raises, raw message identical: {parts}")
    structure = {}
    for key, (v,) in current.get("text_structure", {}).items():
        tally = structure.setdefault(key[0].split("/")[0], [0, 0, 0])
        tally[0] += 1
        tally[1] += v == "i"
        tally[2] += v == "b"
    for group, c in sorted(structure.items()):
        print(f"text_structure {group}: {c[0]} cases, identical {c[1]}, both raise {c[2]}")


def read_proofs(path):
    proofs = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row, _, note = line.rstrip("\n").partition("\t")
                if row and note.strip():
                    proofs[row] = note.strip()
    return proofs


def merge_rows(old, seen):
    merged = dict(old)
    dropped = []
    for row, raw in old.items():
        got = seen.get(row)
        if got is None or not got[0] or (raw and not got[1]):
            dropped.append(row)
    for row, got in seen.items():
        if got[0]:
            merged[row] = got[1]
    return merged, dropped


def write_atomic(path, lines):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    os.replace(path + ".tmp", path)


def refresh(root, log, goldens, text_dir, proof_file, jobs):
    proofs = read_proofs(proof_file)
    writes = {}
    missing = []
    used = {}
    os.makedirs(os.path.join(root, BASELINE_DIR), exist_ok=True)
    if log:
        for suite, seen in sorted(read_log(log).items()):
            path = baseline_path(root, suite)
            old = read_rows(path) if os.path.exists(path) else {}
            merged, dropped = merge_rows(old, seen)
            for row in dropped:
                key = suite + " " + row
                if key in proofs:
                    used[key] = proofs[key]
                    if not seen.get(row, (False, False))[0]:
                        del merged[row]
                    else:
                        merged[row] = False
                else:
                    missing.append(key)
            writes[path] = [f"{r}\t{1 if merged[r] else 0}\n" for r in sorted(merged)]
    if text_dir is not None:
        with tempfile.TemporaryDirectory(prefix="text_ratchet_", dir=os.environ.get("TMPDIR")) as out:
            ours = dump_text(root, goldens, text_dir or out, jobs)
            current = text_verdicts(goldens, ours)
        for name, rows in sorted(current.items()):
            path = os.path.join(root, BASELINE_DIR, name + ".tsv")
            old = read_text_baseline(root, name) if os.path.exists(path) else {}
            fields = FIELDS[name]
            merged = {k: v for k, v in rows.items() if any(x != "-" for x in v)}
            for key, wanted in old.items():
                got = rows.get(key, ("-",) * len(fields))
                for index, field in enumerate(fields):
                    if wanted[index] not in ("-", "n") and wanted[index] != got[index]:
                        tag = f"{name} {key[0]} {key[1]} {field}"
                        if tag in proofs:
                            used[tag] = proofs[tag]
                        else:
                            missing.append(tag)
            writes[path] = ["\t".join(k + v) + "\n" for k, v in sorted(merged.items())]
    if missing:
        print(f"refused: {len(missing)} rows would be dropped without a proof note")
        for row in missing[:SHOWN]:
            print("  " + row)
        print("give --proof-file with lines: <suite or text group> <row>[ <field>]<TAB><proof note>")
        return 1
    for path, lines in writes.items():
        write_atomic(path, lines)
    if used:
        with open(os.path.join(root, BASELINE_DIR, DROPPED), "a", encoding="utf-8") as fh:
            for key, note in sorted(used.items()):
                fh.write(f"{key}\t{note}\n")
    print(f"refreshed {len(writes)} baselines, dropped {len(used)} rows with proof notes")
    return 0


def main():
    parser = argparse.ArgumentParser(description="by-name ratchet of rows known identical to JAX")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--log", default=None, help="an OJAX_EXACT_LOG or run_goldens --log file")
    p.add_argument("--text", action="store_true", help="also dump and check StableHLO and jaxpr text")
    p.add_argument("--partial", action="store_true", help="do not fail on listed rows missing from the log")
    p.add_argument("--goldens", default=None)
    p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("refresh", help="add rows that are identical now, and refuse to drop one without a proof note")
    p.add_argument("--log", default=None)
    p.add_argument("--text", action="store_true")
    p.add_argument("--proof-file", default=None, help="TSV of '<suite> <row>' or '<group> <set> <case> <field>' and a proof note")
    p.add_argument("--goldens", default=None)
    p.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    goldens = args.goldens or os.environ.get("OJAX_GOLDENS") or os.path.join(ROOT, "goldens")
    if args.command == "refresh":
        sys.exit(refresh(ROOT, args.log, goldens, "" if args.text else None, args.proof_file, args.jobs))
    failures = row_failures(ROOT, args.log, not args.partial) if args.log else []
    if args.text:
        failures += run_text_check(ROOT, goldens, args.jobs)
    print(f"row ratchet: {report(failures)} rows fell")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
