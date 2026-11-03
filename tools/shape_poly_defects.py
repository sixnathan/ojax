import argparse
import json
import re

SCOPE = re.compile(r"Using symbolic scope (\d+) created at (.*?) with constraints:")
PROOF = ("JAX prints a SymbolicScope as f\"{id(self)} created at {loc}\" (jax/_src/export/shape_poly.py:1039): "
         "id() is a CPython object address and loc the Python file, line and function that made the scope")


NDIFF = "so the diff in the metadata at these pytree nodes is"
NDIFF_PROOF = ("the lines after 'so the diff in the metadata at these pytree nodes is' are the output of Python's "
               "difflib.ndiff (jax/_src/tree_util.py:1347 in _prefix_error), a Python standard-library text format "
               "with intraline '?' hint lines; ojax prints the same message with plain '- ' and '+ ' lines")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in json.load(fh)["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--rerun", action="append", default=[], help="SEED=PATH of a rerun of the generator")
    ap.add_argument("--suite", default="shape_poly_exprs:x64_off")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.rows)
    reruns = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in args.rerun]
    lines = ["suite\trow\tkind\tproof"]
    for rid, r in sorted(rows.items()):
        msg = r.get("message", "")
        if NDIFF in msg:
            lines.append(f"{args.suite}\t{rid}\tc\t{NDIFF_PROOF}; JAX's diff: {msg.split(NDIFF, 1)[1].strip()!r}")
            continue
        m = SCOPE.search(msg)
        if not m:
            continue
        ids = [f"PYTHONHASHSEED 0 (golden): {m.group(1)}"]
        for seed, other in reruns:
            m2 = SCOPE.search(other[rid].get("message", ""))
            if m2 is None or SCOPE.sub("", other[rid]["message"]) != SCOPE.sub("", r["message"]):
                raise SystemExit("rerun differs outside the scope text: " + rid)
            ids.append(f"PYTHONHASHSEED {seed}: {m2.group(1)}")
        if len(set(i.split(": ")[1] for i in ids)) < 2:
            raise SystemExit("scope id did not change across reruns: " + rid)
        lines.append(f"{args.suite}\t{rid}\ta\t{PROOF}; scope ids {', '.join(ids)}; location {m.group(2)}")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(len(lines) - 1, "rows")


if __name__ == "__main__":
    main()
