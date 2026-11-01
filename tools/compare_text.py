import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = (("x64_off", False), ("x64_on", True))


def _structure(jaxpr, jax_core):
    parts = []
    for eqn in jaxpr.eqns:
        nested = []
        for key in sorted(eqn.params):
            value = eqn.params[key]
            items = value if isinstance(value, (tuple, list)) else [value]
            for item in items:
                inner = getattr(item, "jaxpr", item)
                if isinstance(inner, jax_core.Jaxpr):
                    nested.append(_structure(inner, jax_core))
        name = eqn.primitive.name
        parts.append(name + ("(" + " | ".join(nested) + ")" if nested else ""))
    return " ".join(parts)


def _reference_case(fn, args, jax, jax_core):
    row = {}
    fn.__name__ = "f"
    try:
        closed = jax.make_jaxpr(fn)(*args)
        row["jaxpr"] = str(closed)
        row["structure"] = _structure(closed.jaxpr, jax_core)
    except Exception as exc:
        row["jaxpr_error"] = str(exc)
        row["structure_error"] = str(exc)
    try:
        row["text"] = jax.jit(fn).lower(*args).as_text()
    except Exception as exc:
        row["text_error"] = str(exc)
    return row


def reference(goldens, out, shard, shards):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, HERE)
    import numpy as np
    import jax
    from jax._src import core as jax_core
    import parity_registry as registry
    import gen_goldens as gg

    with open(out, "w", encoding="utf-8") as sink:
        for set_name, x64 in SETS:
            jax.config.update("jax_enable_x64", x64)
            set_dir = os.path.join(goldens, "parity", set_name)
            with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as fh:
                cases = json.load(fh)["cases"]
            for i, case in enumerate(cases):
                if shards > 1 and i % shards != shard:
                    continue
                names = [a["name"] for a in case["args"]]
                with np.load(os.path.join(set_dir, "inputs", case["case_id"] + ".npz")) as z:
                    arrays = [z[n] for n in names]
                op = registry.OPS[case["op"]]
                params = case["params"]

                def fn(*xs, _op=op, _names=names, _params=params):
                    return _op(**dict(zip(_names, xs)), p=_params)

                row = _reference_case(fn, arrays, jax, jax_core)
                row.update({"set": "parity/" + set_name, "case": case["case_id"]})
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()
                if i % 50 == 0:
                    jax.clear_caches()
            if shard != 0:
                continue
            with open(os.path.join(gg.ROOT, "spec", "jaxpr.cases.json"), encoding="utf-8") as fh:
                jaxpr_cases = json.load(fh)["cases"]
            for case in jaxpr_cases:
                args = [np.zeros(tuple(a["shape"]), np.dtype(a["dtype"])) for a in case["in_avals"]]
                base = gg.JAXPR_FNS[case["fn"]]

                def fn(*xs, _base=base):
                    return _base(*xs)

                row = _reference_case(fn, args, jax, jax_core)
                row.update({"set": "jaxpr/" + set_name, "case": case["case_id"]})
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()


def load_rows(paths):
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(row["set"], row["case"])] = row
    return rows


def _first_alternative(body):
    depth = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and body.startswith(" | ", i):
            return body[:i]
    return body


def flatten_calls(structure):
    out = structure
    for name in ("jit", "custom_jvp_call", "custom_vjp_call"):
        pattern = re.compile(r"(?<![\w])" + name + r"\(")
        while True:
            match = pattern.search(out)
            if match is None:
                break
            depth = 1
            i = match.end()
            while depth:
                if out[i] == "(":
                    depth += 1
                elif out[i] == ")":
                    depth -= 1
                i += 1
            body = _first_alternative(out[match.end() : i - 1])
            out = out[: match.start()] + body + out[i:]
    return " ".join(out.split())


def first_difference(a, b):
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return i, x, y
    if len(la) != len(lb):
        i = min(len(la), len(lb))
        return i, la[i] if i < len(la) else "<end>", lb[i] if i < len(lb) else "<end>"
    return None


def _shape_token(line):
    line = re.sub(r"\d+", "N", line.strip())
    line = re.sub(r"%\w+", "%v", line)
    return line[:70]


def _operation(line):
    match = re.search(r"(?:= )?(stablehlo\.\w+|chlo\.\w+|func\.\w+|call|return|module|func)", line)
    return match.group(1) if match else line.strip()[:30]


def _text_cause(ref, ours):
    diff = first_difference(ref, ours)
    if diff is None:
        return "identical"
    _, a, b = diff
    return _operation(a) + " vs " + _operation(b)


def _structure_cause(ref, ours):
    ta, tb = ref.split(), ours.split()
    for x, y in zip(ta, tb):
        if x != y:
            return x.split("(")[0] + " vs " + y.split("(")[0]
    if len(ta) != len(tb):
        extra = ta[len(tb)] if len(ta) > len(tb) else tb[len(ta)]
        side = "ref extra " if len(ta) > len(tb) else "ours extra "
        return side + extra.split("(")[0]
    return "identical"


def _jaxpr_cause(ref, ours):
    diff = first_difference(ref, ours)
    if diff is None:
        return "identical"
    _, a, b = diff
    return _shape_token(a)[:40] + " vs " + _shape_token(b)[:40]


def _op_of(case):
    match = re.match(r"(.+?)_(?:\d|_|x)*[0-9a-f]{8}_\w+$", case)
    return match.group(1) if match else case


def compare(ref_paths, our_paths, out, top):
    ref = load_rows(ref_paths)
    ours = load_rows(our_paths)
    report = {"sets": {}, "causes": {}, "failing": collections.defaultdict(list)}
    kinds = (
        ("text", _text_cause),
        ("jaxpr", _jaxpr_cause),
        ("structure", _structure_cause),
        ("structure_flat", _structure_cause),
    )
    totals = collections.Counter()
    by_set = collections.defaultdict(collections.Counter)
    causes = {kind: collections.Counter() for kind, _ in kinds}
    examples = {kind: {} for kind, _ in kinds}
    ops = {kind: collections.defaultdict(set) for kind, _ in kinds}
    for key, r in sorted(ref.items()):
        o = ours.get(key)
        group = key[0].split("/")[0]
        totals["cases"] += 1
        by_set[group]["cases"] += 1
        for kind, cause_fn in kinds:
            field = "structure" if kind == "structure_flat" else kind
            if o is None:
                verdict = "ojax_missing"
            elif field not in r and field not in o:
                verdict = "both_error"
            elif field not in r:
                verdict = "jax_error_only"
            elif field not in o:
                verdict = "ojax_error_only"
            else:
                a, b = r[field], o[field]
                if kind == "structure_flat":
                    a, b = flatten_calls(a), flatten_calls(b)
                verdict = "identical" if a == b else "different"
            by_set[group][kind + ":" + verdict] += 1
            totals[kind + ":" + verdict] += 1
            if verdict == "different":
                cause = cause_fn(a, b)
            elif verdict == "ojax_error_only":
                cause = "ojax error: " + re.sub(r"\d+", "N", o.get(field + "_error", ""))[:70]
            else:
                continue
            causes[kind][cause] += 1
            examples[kind].setdefault(cause, key)
            ops[kind][cause].add(_op_of(key[1]))
    report = {
        "totals": dict(sorted(totals.items())),
        "by_group": {k: dict(sorted(v.items())) for k, v in sorted(by_set.items())},
        "causes": {
            kind: [
                {"cause": c, "cases": n, "ops": len(ops[kind][c]), "example": examples[kind][c]}
                for c, n in counter.most_common()
            ]
            for kind, counter in causes.items()
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    for group, counter in sorted(by_set.items()):
        sys.stdout.write(group + " cases=" + str(counter["cases"]) + "\n")
        for kind, _ in kinds:
            cells = " ".join(
                v + "=" + str(counter[kind + ":" + v])
                for v in ("identical", "different", "ojax_error_only", "jax_error_only", "both_error", "ojax_missing")
                if counter[kind + ":" + v]
            )
            sys.stdout.write("  " + kind + ": " + cells + "\n")
    for kind, _ in kinds:
        sys.stdout.write("top causes " + kind + "\n")
        for c, n in causes[kind].most_common(top):
            sys.stdout.write("  " + str(n) + " [" + str(len(ops[kind][c])) + " ops] " + c + "\n")


def show(ref_paths, our_paths, set_name, case, field):
    r = load_rows(ref_paths)[(set_name, case)]
    o = load_rows(our_paths)[(set_name, case)]
    for label, row in (("JAX", r), ("OJAX", o)):
        sys.stdout.write("---- " + label + "\n")
        sys.stdout.write(str(row.get(field, row.get(field + "_error"))) + "\n")


def arrays_reference(goldens, out):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax.numpy as jnp

    set_dir = os.path.join(goldens, "parity", "x64_off")
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    with open(out, "w", encoding="utf-8") as sink:
        for case in cases:
            with np.load(os.path.join(set_dir, "inputs", case["case_id"] + ".npz")) as z:
                for name in z.files:
                    arr = jnp.asarray(z[name])
                    row = {"case": case["case_id"], "arg": name, "repr": repr(arr), "str": str(arr)}
                    sink.write(json.dumps(row, sort_keys=True) + "\n")


def arrays_compare(ref_path, our_path):
    def load(path):
        with open(path, encoding="utf-8") as fh:
            return {(r["case"], r["arg"]): r for r in map(json.loads, fh)}

    ref, ours = load(ref_path), load(our_path)
    totals = collections.Counter()
    by_dtype = collections.Counter()
    samples = []
    for key, r in sorted(ref.items()):
        o = ours.get(key, {})
        totals["arrays"] += 1
        for field in ("repr", "str"):
            same = r[field] == o.get(field)
            totals[field + "_identical"] += same
            if field == "repr" and not same:
                match = re.search(r"dtype=(\w+)", r["repr"])
                by_dtype[match.group(1) if match else "default"] += 1
                if len(samples) < 6:
                    samples.append((r["repr"][:160], str(o.get("repr", o.get("error")))[:160]))
    sys.stdout.write(json.dumps(totals, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(by_dtype.most_common(), sort_keys=True) + "\n")
    for a, b in samples:
        sys.stdout.write("JAX  " + a + "\nOJAX " + b + "\n")


def errors_compare(ref_paths, our_paths):
    ref, ours = load_rows(ref_paths), load_rows(our_paths)
    both = same = 0
    samples = []
    for key, r in sorted(ref.items()):
        o = ours.get(key, {})
        if "text_error" in r and "text_error" in o:
            both += 1
            if r["text_error"] == o["text_error"]:
                same += 1
            elif len(samples) < 6:
                samples.append((r["text_error"][:170], o["text_error"][:170]))
    sys.stdout.write("both_error=" + str(both) + " identical_message=" + str(same) + "\n")
    for a, b in samples:
        sys.stdout.write("JAX  " + a.replace("\n", " ") + "\nOJAX " + b.replace("\n", " ") + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reference")
    p.add_argument("--goldens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p = sub.add_parser("compare")
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--ours", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=30)
    p = sub.add_parser("show")
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--ours", nargs="+", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--field", default="text")
    p = sub.add_parser("arrays-reference")
    p.add_argument("--goldens", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("arrays-compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--ours", required=True)
    p = sub.add_parser("errors")
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--ours", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "arrays-reference":
        arrays_reference(args.goldens, args.out)
    elif args.command == "arrays-compare":
        arrays_compare(args.ref, args.ours)
    elif args.command == "errors":
        errors_compare(args.ref, args.ours)
    elif args.command == "reference":
        reference(args.goldens, args.out, args.shard, args.shards)
    elif args.command == "compare":
        compare(args.ref, args.ours, args.out, args.top)
    else:
        show(args.ref, args.ours, args.set, args.case, args.field)


if __name__ == "__main__":
    main()
