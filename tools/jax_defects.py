#!/usr/bin/env python3
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFECTS = os.path.join("test", "golden", "jax_defects.tsv")
SETS = (("x64_off", False), ("x64_on", True))
KINDS = {
    "a": "JAX is nondeterministic",
    "b": "JAX's jit and eager disagree",
    "c": "text exists only because of Python",
    "d": "input not expressible in OCaml's types",
}
DTYPE_LIST = re.compile(r"subtypes of [A-Za-z_]+(, [A-Za-z_]+)+")
TRACED_AT = re.compile(r"The error occurred while tracing the function \S+ at \S+\.py:\d+")
PY_TYPE_ERRORS = ("'int' object is not iterable", "object of type 'int' has no len()")
MLIR_LOC = re.compile(r"(?:UNKNOWN: |at loc\().*?([^\"\s(]+\.py)\"?:\d+")
REPEATED_AXES = re.compile(r"does not support repeated axes\. Got axes \(")
PY_ADDRESS = re.compile(r"<function ([A-Za-z_0-9]+) at (0x[0-9a-f]+)>")


def jax_tree():
    import jax

    return os.path.dirname(jax.__file__)


def load_option(goldens, set_name):
    with open(os.path.join(goldens, "option_fuzz", set_name, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def load_compose(goldens, set_name):
    with open(os.path.join(goldens, "compose_fuzz", set_name, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def compose_references(goldens):
    refs = {}
    with open(os.path.join(goldens, "compose_fuzz", "text", "reference.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            refs[(row["set"].split("/")[-1], row["case"])] = row
    return refs


def dtype_list_tasks(goldens):
    tasks = []
    for set_name, _ in SETS:
        for case in load_option(goldens, set_name):
            modes = [m for m in ("eager", "jit") if DTYPE_LIST.search(case.get(m + "_error_text", ""))]
            if modes:
                stored = {m: case[m + "_error_text"] for m in modes}
                tasks.append({"kind": "option", "set": set_name, "case": case["case_id"], "modes": modes, "stored": stored})
    refs = compose_references(goldens)
    for set_name, _ in SETS:
        for case in load_compose(goldens, set_name):
            ref = refs.get((set_name, case["case_id"]))
            if ref is None:
                continue
            fields = [f for f in ("jaxpr", "text") if DTYPE_LIST.search(ref.get(f + "_error", ""))]
            if fields:
                stored = {f: ref[f + "_error"] for f in fields}
                tasks.append({"kind": "compose", "set": set_name, "case": case["case_id"], "fields": fields, "stored": stored})
    return tasks


def canonical(text):
    return DTYPE_LIST.sub(lambda m: "subtypes of " + ", ".join(sorted(m.group(0)[len("subtypes of "):].split(", "))), text)


def bundle_arrays(goldens, suite, set_name, op, case_id):
    import numpy as np

    path = os.path.join(goldens, suite, set_name, "ops", op + ".npz")
    prefix = case_id + "/in_"
    with np.load(path) as z:
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def innermost_jax_frame(error, tree):
    frames = traceback.extract_tb(error.__traceback__)
    inside = [f for f in frames if f.filename.startswith(tree)]
    if not inside:
        return ""
    f = inside[-1]
    return f"jax/{os.path.relpath(f.filename, tree)}:{f.lineno}: {(f.line or '').strip()}"


def run_option(goldens, set_name, case, tree):
    import gen_option_fuzz as go
    import jax
    import option_registry as registry

    arrays = bundle_arrays(goldens, "option_fuzz", set_name, case["op"], case["case_id"])
    names = sorted(arrays)
    out = {}
    try:
        devices = {k: jax.numpy.asarray(v) for k, v in arrays.items()}
        go.to_outputs(registry.call(case["module"], case["name"], case["positional"], case["options"], devices))
        out["eager"] = None
    except Exception as error:
        out["eager"] = (str(error), innermost_jax_frame(error, tree))

    def traced(*values):
        return registry.call(case["module"], case["name"], case["positional"], case["options"], dict(zip(names, values)))

    try:
        go.to_outputs(jax.jit(traced)(*[arrays[k] for k in names]))
        out["jit"] = None
    except Exception as error:
        out["jit"] = (str(error), innermost_jax_frame(error, tree))
    return out


def run_compose(goldens, set_name, case):
    import compare_text
    import gen_compose_fuzz as gc
    import jax
    from jax._src import core as jax_core

    if case["op"] == "promote":
        with open(os.path.join(goldens, "compose_fuzz", set_name, "manifest.json"), encoding="utf-8") as fh:
            arrays = gc.make_inputs(case, json.load(fh)["seed"])
    else:
        arrays = bundle_arrays(goldens, "compose_fuzz", set_name, case["op"], case["case_id"])
    values = [arrays[k] for k in sorted(arrays)]
    row = compare_text._reference_case(gc.program(case), values, jax, jax_core)
    return {f: row.get(f + "_error") for f in ("jaxpr", "text")}


def worker(goldens, tasks_path, set_name, out_path):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, HERE)
    import jax

    jax.config.update("jax_enable_x64", set_name == "x64_on")
    tree = jax_tree()
    with open(tasks_path, encoding="utf-8") as fh:
        tasks = [t for t in json.load(fh) if t["set"] == set_name]
    option = {c["case_id"]: c for c in load_option(goldens, set_name)}
    compose = {c["case_id"]: c for c in load_compose(goldens, set_name)}
    results = []
    for task in tasks:
        if task["kind"] == "option":
            got = run_option(goldens, set_name, option[task["case"]], tree)
            for mode in task.get("modes", ("eager", "jit")):
                text = got[mode][0] if got[mode] else None
                frame = got[mode][1] if got[mode] else ""
                results.append(["text_option_fuzz:" + set_name, f"{task['case']}:{mode}:error_text", text, frame])
        else:
            got = run_compose(goldens, set_name, compose[task["case"]])
            for field in task["fields"]:
                results.append(["text_compose_fuzz:" + set_name, f"{task['case']}:{field}:error_text", got[field], ""])
        if len(results) % 200 == 0:
            jax.clear_caches()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)


def spawn(goldens, tasks, out_dir, seeds):
    os.makedirs(out_dir, exist_ok=True)
    tasks_path = os.path.join(out_dir, "tasks.json")
    with open(tasks_path, "w", encoding="utf-8") as fh:
        json.dump(tasks, fh)
    procs = []
    for seed in seeds:
        for set_name, _ in SETS:
            out = os.path.join(out_dir, f"seed{seed}_{set_name}.json")
            env = dict(os.environ, PYTHONHASHSEED=str(seed), OPENBLAS_NUM_THREADS="1", JAX_PLATFORMS="cpu")
            cmd = ["nice", "-n", "10", sys.executable, os.path.abspath(__file__), "worker", goldens, tasks_path, set_name, out]
            procs.append((subprocess.Popen(cmd, env=env), out))
    for proc, out in procs:
        if proc.wait() != 0:
            sys.exit(f"jax_defects: worker for {out} failed")
    return [out for _, out in procs]


def hashseed(goldens, out_dir, seeds):
    tasks = dtype_list_tasks(goldens)
    print(f"{len(tasks)} cases carry a dtype list in JAX's stored text")
    paths = spawn(goldens, tasks, out_dir, seeds)
    stored = {}
    for task in tasks:
        suite = ("text_option_fuzz:" if task["kind"] == "option" else "text_compose_fuzz:") + task["set"]
        for part, text in task["stored"].items():
            row = f"{task['case']}:{part}:error_text"
            stored[(suite, row)] = text
    seen = collections.defaultdict(dict)
    for path in paths:
        seed = os.path.basename(path).split("_")[0][4:]
        with open(path, encoding="utf-8") as fh:
            for suite, row, text, _ in json.load(fh):
                seen[(suite, row)][seed] = text
    varying = {}
    unreproduced = []
    for key, by_seed in sorted(seen.items()):
        want = canonical(stored[key])
        if any(canonical(t or "") != want for t in by_seed.values()):
            unreproduced.append("\t".join(key))
            continue
        lists = sorted({DTYPE_LIST.search(t).group(0) for t in by_seed.values()})
        if len(lists) > 1:
            varying["\t".join(key)] = {"seeds": by_seed, "lists": lists, "stored": DTYPE_LIST.search(stored[key]).group(0)}
    with open(os.path.join(out_dir, "varying.json"), "w", encoding="utf-8") as fh:
        json.dump({"seeds": seeds, "rows": len(seen), "varying": varying, "unreproduced": unreproduced}, fh, indent=1, sort_keys=True)
    print(f"{len(seen)} rows rerun under PYTHONHASHSEED {','.join(map(str, seeds))}: {len(varying)} vary, {len(unreproduced)} not reproduced")


def read_log(path):
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6:
                rows[(f[4], f[0])] = f
    return rows


def wants_frame(suite, f):
    if f[2] == "0":
        return False
    if suite.startswith("text_option_fuzz:") and len(f) >= 7:
        return any(p in f[6] for p in PY_TYPE_ERRORS) or bool(REPEATED_AXES.search(f[6]))
    return suite.startswith("option_fuzz:") and f[5].startswith("no_binding:type:")


def frames_for(goldens, log, out_dir):
    rows = read_log(log)
    tasks = {}
    for (suite, row), f in rows.items():
        if wants_frame(suite, f):
            set_name = suite.split(":", 1)[1]
            case, mode, _ = row.rsplit(":", 2)
            tasks.setdefault((set_name, case), {"kind": "option", "set": set_name, "case": case, "modes": []})["modes"].append(mode)
    paths = spawn(goldens, list(tasks.values()), out_dir, [0])
    frames = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for suite, row, text, frame in json.load(fh):
                case, mode, _ = row.rsplit(":", 2)
                frames[f"{suite.split(':')[1]}\t{case}:{mode}"] = [text, frame]
    with open(os.path.join(out_dir, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump(frames, fh, indent=1, sort_keys=True)
    print(f"{len(frames)} rows rerun for the raising JAX source line")


def jax_line(tree, relpath, needle):
    path = os.path.join(tree, relpath)
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if needle in line:
                return f"jax/_src/{relpath}:{number}: {line.strip()}"
    return f"jax/_src/{relpath}: {needle}"


def clean(text):
    return " ".join(str(text).split())[:300]


def unescape(text):
    return text.replace('\\"', '"')


def strip_traced(text):
    text = re.sub(r"tracing the function .*? for (\w+)\.", r"tracing the function F for \1.", text)
    text = re.sub(r"from line [^\\]*?(?=\\n|$)", "from line L", text)
    return re.sub(r"depends on the value of the argument [^.]*\.", "depends on the value of the argument A.", text)


def strip_loc(text):
    text = re.sub(r"\\n\\t\\tat loc\(.*?(?=\\n|$)", "", text)
    text = re.sub(r"\\n\\tsee current operation: .*?(?=\\n|$)", "", text)
    text = re.sub(r"^UNKNOWN: \S+\.py:\d+:\d+: ", "UNKNOWN: ", text)
    text = re.sub(r"^(UNKNOWN: error: [^\\]*)\\n.*$", r"\1", text)
    return text


def strip_signature(text):
    return re.sub(r"axis (-?\d+) of (argument \w+|args\[\d+\])", r"axis \1 of ARG", text)


def same_after(strip, theirs, ours):
    return strip(unescape(theirs)) == strip(unescape(ours))


def option_cases(goldens):
    cases = {}
    for set_name, _ in SETS:
        for case in load_option(goldens, set_name):
            cases[(set_name, case["case_id"])] = case
    return cases


def eager_jit_differ(goldens, suite, set_name, case, case_id, name):
    import numpy as np

    if case is None:
        return None
    path = os.path.join(goldens, suite, set_name, "ops", case["op"] + ".npz")
    with np.load(path) as z:
        eager, jit = z.files and (case_id + "/eager_" + name, case_id + "/jit_" + name)
        if eager not in z.files or jit not in z.files:
            return None
        a, b = z[eager], z[jit]
    if a.shape != b.shape or a.dtype != b.dtype:
        return (a.size, a.size)
    width = a.dtype.itemsize
    left = a.reshape(-1).view(np.uint8).reshape(-1, width)
    right = b.reshape(-1).view(np.uint8).reshape(-1, width)
    count = int((left != right).any(axis=1).sum())
    return (count, a.size) if count else None


def build(log, varying_path, frames_path, goldens):
    import jax

    tree = os.path.join(os.path.dirname(jax.__file__), "_src")
    traced = jax_line(tree, "api_util.py", 'return f"{fun.__name__} at {filename}:{lineno}"')
    origin = jax_line(tree, "interpreters/partial_eval.py", 'origin = ("The error occurred while tracing the function "')
    axes = jax_line(tree, "numpy/fft.py", "does not support repeated axes. Got axes {axes}.")
    loc = jax_line(tree, "interpreters/mlir.py", "def source_info_to_location(")
    signature = jax_line(tree, "api.py", "ba = inspect.signature(fn).bind(*args, **kwargs)")
    from jax._src.random import prng

    impl_raise = jax_line(tree, "random/prng.py", "end with {impl.key_shape}; got shape={key_data.shape} for {impl=}")
    impl_class = jax_line(tree, "random/prng.py", "class PRNGImpl(NamedTuple):")
    rows = read_log(log)
    out = {}
    with open(varying_path, encoding="utf-8") as fh:
        varying = json.load(fh)
    seeds = ",".join(map(str, varying["seeds"]))
    for key, info in varying["varying"].items():
        suite, row = key.split("\t")
        lists = " | ".join(info["lists"])
        out[(suite, row)] = ("a", f"PYTHONHASHSEED {seeds} give {len(info['lists'])} texts: {lists}; the list is a Python set iterated at jax/_src/lax/lax.py naryop_dtype_rule and unop_dtype_rule")
    frames = {}
    if frames_path:
        with open(frames_path, encoding="utf-8") as fh:
            frames = json.load(fh)
    options = option_cases(goldens)
    from jax._src.lax import linalg as lax_linalg
    from jax._src.lax import lax as lax_lax

    precision_names = {k for k in lax_lax._precision_strings if isinstance(k, str)} | {p.name for p in lax_lax.DotAlgorithmPreset}
    precision_line = jax_line(tree, "lax/lax.py", '"Precision argument must be one of:')

    enums = {
        "implementation": (
            jax_line(tree, "lax/linalg.py", "class EigImplementation(enum.Enum):") + " and " + jax_line(tree, "lax/linalg.py", "class EighImplementation(enum.Enum):"),
            sorted({m.value for m in lax_linalg.EigImplementation} | {m.value for m in lax_linalg.EighImplementation}),
        )
    }
    for (suite, row), f in sorted(rows.items()):
        if f[2] == "0" or (suite, row) in out:
            continue
        if suite.startswith("option_fuzz:") and f[5].startswith("no_binding:type:"):
            set_name = suite.split(":", 1)[1]
            case_id, mode, _ = row.rsplit(":", 2)
            case = options[(set_name, case_id)]
            text, frame = frames.get(f"{set_name}\t{case_id}:{mode}", [None, ""])
            said = f"JAX raises '{clean(text)}' at {frame or 'no JAX frame'}" if text else "JAX returns without checking it"
            opts = clean(json.dumps(case["options"], sort_keys=True))
            if text and ("cannot unpack non-iterable" in text or "is not iterable" in text):
                out[(suite, row)] = ("c", f"Python's own unpacking TypeError for options {opts}: {said}")
            elif f[5] == "no_binding:type:enum":
                bad = [(k, v, enums[k]) for k, v in case["options"].items() if k in enums and isinstance(v, str) and v not in enums[k][1]]
                if bad:
                    k, v, (where, members) = bad[0]
                    out[(suite, row)] = ("d", f"{k}={v!r} is a string outside the enum at {where} (members {', '.join(members)}), so OCaml's variant cannot hold it; options {opts}: {said}")
            continue
        if suite.startswith(("option_fuzz:", "compose_fuzz:", "fuzz:", "fuzz_s2:")) and f[5] == "ulps":
            set_name = suite.split(":", 1)[1]
            case_id, mode, name = row.rsplit(":", 2)
            differ = eager_jit_differ(goldens, suite.split(":")[0], set_name, options.get((set_name, case_id)), case_id, name)
            if differ:
                out[(suite, row)] = ("b", f"JAX's own eager and jit {name} differ in {differ[0]} of {differ[1]} elements (stored bytes in {suite.split(':')[0]}/{set_name}/ops); ojax's {mode} output differs from JAX's {mode} (log: {f[1]} compared, {f[2]} differing)")
            continue
        if suite.startswith("keys_arrays:") and len(f) >= 8 and PY_ADDRESS.search(f[6]) and PY_ADDRESS.sub(r"<function \1>", f[6]) == f[7]:
            name, stored = PY_ADDRESS.search(f[6]).groups()
            live = {fn.__name__: hex(id(fn)) for impl in prng.prngs.values() for fn in impl[1:5]}
            out[(suite, row)] = ("a", f"CPython object addresses: JAX ends the text with {{impl=}} at {impl_raise}, the NamedTuple repr of {impl_class}, which prints every impl function as <function name at 0x...>; the stored golden has {name} at {stored} and this run has it at {live.get(name)}; ojax's text equals JAX's with the addresses removed")
            continue
        if len(f) < 8 or not suite.startswith(("text_option_fuzz:", "text_fuzz:", "text_fuzz_s2:", "text_compose_fuzz:")):
            continue
        theirs, ours = f[6], f[7]
        plain = unescape(theirs)
        if suite.startswith("text_option_fuzz:") and ours.startswith("Precision argument must be one of:"):
            set_name = suite.split(":", 1)[1]
            case = options.get((set_name, row.rsplit(":", 2)[0]))
            value = case["options"].get("precision") if case else None
            if isinstance(value, str) and value not in precision_names:
                out[(suite, row)] = ("d", f"precision={value!r} is a string outside JAX's precision names ({', '.join(sorted(precision_names))}) checked at {precision_line}, so OCaml's precision type cannot hold it and the harness raises before the call; JAX raises '{clean(plain)}'")
                continue
        if TRACED_AT.search(plain) and same_after(strip_traced, theirs, ours):
            out[(suite, row)] = ("c", f"identical once the Python function, its file and parameter names are removed: {clean(TRACED_AT.search(plain).group(0))}; from {traced} and {origin}")
        elif "of argument " in plain and same_after(strip_signature, theirs, ours):
            out[(suite, row)] = ("c", f"identical once the Python function's parameter names are removed: {clean(re.search(r'axis -?[0-9]+ of argument [A-Za-z_0-9]+', plain).group(0))}; from {signature}")
        elif MLIR_LOC.search(plain) and same_after(strip_loc, theirs, ours):
            out[(suite, row)] = ("c", f"identical once JAX's MLIR loc into Python files is removed: {clean(MLIR_LOC.search(plain).group(0))}; locations from {loc}")
        elif any(p in plain for p in PY_TYPE_ERRORS):
            set_name = suite.split(":", 1)[1]
            case_id, mode, _ = row.rsplit(":", 2)
            got = frames.get(f"{set_name}\t{case_id}:{mode}")
            where = got[1] if got and got[1] else "frame not rerun"
            out[(suite, row)] = ("c", f"Python's own TypeError '{clean(plain)}' raised by CPython at {where}")
        elif REPEATED_AXES.search(plain) and plain.replace("Got axes (", "Got axes [").replace(").", "].") == unescape(ours):
            out[(suite, row)] = ("d", f"JAX prints the user's Python tuple with {{axes}}: '{clean(plain)}' at {axes}; OCaml's axes are one int list type and the list form is pinned by the reference unit test")
    return out


def write_defects(path, defects):
    lines = ["suite\trow\tkind\tproof\n"]
    for (suite, row), (kind, proof) in sorted(defects.items()):
        lines.append(f"{suite}\t{row}\t{kind}\t{proof}\n")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    os.replace(path + ".tmp", path)


def read_defects(path):
    defects = {}
    if not os.path.exists(path):
        return defects
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) == 4 and f[2] in KINDS and f[3]:
                defects[(f[0], f[1])] = f[2]
            elif line.strip():
                sys.exit(f"jax_defects: bad line in {path}: {line.rstrip()}")
    return defects


def report(root, log, out=sys.stdout):
    defects = read_defects(os.path.join(root, DEFECTS))
    by_suite = collections.defaultdict(collections.Counter)
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            suite, row = f[4], f[0]
            same = f[2] == "0"
            kind = defects.get((suite, row))
            c = by_suite[suite]
            c["rows"] += 1
            if kind is None:
                c["identical" if same else "differ"] += 1
            else:
                c["defect"] += 1
                c["defect_" + kind] += 1
                c["defect_identical" if same else "defect_differ"] += 1
    marked = [s for s, c in by_suite.items() if c["defect"]]
    print("defect policy: suite\trows\tidentical\tdefect (a b c d; of them identical by chance)\tdiffering", file=out)
    for suite in sorted(by_suite):
        c = by_suite[suite]
        if c["defect"] or (c["differ"] and suite not in ("compare",) and not suite.startswith("parity_jit_eager:")):
            kinds = " ".join(str(c["defect_" + k]) for k in "abcd")
            print(f"defect policy: {suite}\t{c['rows']}\t{c['identical']}\t{c['defect']} ({kinds}; {c['defect_identical']})\t{c['differ']}", file=out)
    listed = set(defects)
    seen = {(s, r) for s in marked for r in ()}
    del seen
    return listed


def main():
    parser = argparse.ArgumentParser(description="JAX-defect rows: proofs and reporting, rows are never removed")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("hashseed", help="rerun JAX under several PYTHONHASHSEED values on every text row with a dtype list")
    p.add_argument("--goldens", default=os.path.join(ROOT, "goldens"))
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4,5")
    p = sub.add_parser("frames", help="rerun JAX on differing text rows for the JAX source line that raises")
    p.add_argument("--goldens", default=os.path.join(ROOT, "goldens"))
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build", help="write test/golden/jax_defects.tsv from the proofs")
    p.add_argument("--log", required=True)
    p.add_argument("--varying", required=True)
    p.add_argument("--frames", default=None)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--goldens", default=os.path.join(ROOT, "goldens"))
    p = sub.add_parser("report")
    p.add_argument("--log", required=True)
    p = sub.add_parser("worker")
    p.add_argument("goldens")
    p.add_argument("tasks")
    p.add_argument("set_name")
    p.add_argument("out")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.goldens, args.tasks, args.set_name, args.out)
    elif args.command == "hashseed":
        hashseed(os.path.abspath(args.goldens), os.path.abspath(args.out), [int(s) for s in args.seeds.split(",")])
    elif args.command == "frames":
        frames_for(os.path.abspath(args.goldens), args.log, os.path.abspath(args.out))
    elif args.command == "build":
        defects = build(args.log, args.varying, args.frames, os.path.abspath(args.goldens))
        write_defects(os.path.join(args.root, DEFECTS), defects)
        print(f"{len(defects)} rows: " + ", ".join(f"{k} {sum(1 for v in defects.values() if v[0] == k)}" for k in "abcd"))
    else:
        report(ROOT, args.log)


if __name__ == "__main__":
    main()
