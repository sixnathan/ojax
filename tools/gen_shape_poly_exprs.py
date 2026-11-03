import argparse
import json
import os
import random

import gen_goldens as gg
import jax
from jax import export
from jax._src import core

MODULE = "shape_poly_exprs"

SCOPES = [
    ("a, b", []),
    ("a, b, c", []),
    ("a, b", ["a >= b"]),
    ("a, b", ["b >= 2", "a <= 10"]),
    ("a, b", ["a >= 2*b"]),
    ("a, b, c", ["a + b >= c", "c >= 3"]),
    ("n, m", ["n == 2*m"]),
    ("a, b", ["max(a, b) <= 8"]),
    ("a, b", ["mod(a, 2) == 0"]),
    ("a, b", ["floordiv(a, 2) >= b"]),
]

BINOPS = ["add", "sub", "mul", "floordiv", "mod", "max", "min"]
UNOPS = ["neg", "pow2", "pow3"]
CMPS = ["eq", "ne", "ge", "gt", "le", "lt"]


def gen_expr(rng, nvars, depth):
    r = rng.random()
    if depth == 0 or r < 0.3:
        if rng.random() < 0.7:
            return ["v", rng.randrange(nvars)]
        return ["c", rng.choice([-3, -1, 0, 1, 2, 3, 4, 5])]
    if r < 0.4:
        return [rng.choice(UNOPS), gen_expr(rng, nvars, depth - 1)]
    op = rng.choice(BINOPS)
    lhs = gen_expr(rng, nvars, depth - 1)
    if op in ("floordiv", "mod") and rng.random() < 0.6:
        rhs = ["c", rng.choice([1, 2, 3, 4])]
    else:
        rhs = gen_expr(rng, nvars, depth - 1)
    return [op, lhs, rhs]


def ev(e, dims):
    k = e[0]
    if k == "v":
        return dims[e[1]]
    if k == "c":
        return e[1]
    if k == "neg":
        return -ev(e[1], dims)
    if k == "pow2":
        return ev(e[1], dims) ** 2
    if k == "pow3":
        return ev(e[1], dims) ** 3
    a, b = ev(e[1], dims), ev(e[2], dims)
    if k == "add":
        return a + b
    if k == "sub":
        return a - b
    if k == "mul":
        return a * b
    if k == "floordiv":
        return a // b
    if k == "mod":
        return a % b
    if k == "max":
        return core.max_dim(a, b)
    if k == "min":
        return core.min_dim(a, b)
    if k == "eq":
        return a == b
    if k == "ne":
        return a != b
    if k == "ge":
        return a >= b
    if k == "gt":
        return a > b
    if k == "le":
        return a <= b
    if k == "lt":
        return a < b
    raise ValueError(k)


def outcome(fn):
    try:
        v = fn()
        if isinstance(v, bool):
            return {"kind": "bool", "text": str(v)}
        return {"kind": "dim", "text": str(v)}
    except Exception as ex:
        return {"kind": "raise", "class": type(ex).__name__, "message": str(ex)}


SPECS = [
    ("a, b", None), ("a, ...", (None, 3, 4)), ("...", (2, 3)), ("_, a", (5, None)),
    ("a + b, a - b", None), ("2 * a", None), ("a*b*c", None), ("3*a^2", None),
    ("mod(a, 2)", None), ("floordiv(a, b)", None), ("max(a, b, c)", None), ("min(a)", None),
    ("non_negative(a - 3)", None), ("b - 1", None), ("a, _", (None, 4)), ("a, 3", (None, 4)),
    ("_", None), ("a, ...", None), ("...", None), ("", None), ("a,", None), ("(a, b)", None),
    ("a b", None), ("1a", None), ("a_1, b2", None), ("-a", None), ("a * -2", None),
    ("a, a", (3, 4)), ("a, 2*a", (None, None)), ("max(a, 0)", None), ("4", (4,)), ("4", (5,)),
    ("a, b", (2,)), ("a ^ 2", None), ("a**2", None), ("floordiv(a)", None), ("mod(a, b, c)", None),
    ("a + ", None), ("a, , b", None), ("a .. b", None),
]


ARGS_SPECS = [
    ([[2, 3], [4]], ["a, _", "b"]),
    ([[2, 3], [4]], "a, ..."),
    ([[2, 3]], None),
    ([[2, 3], [4]], ["a, b"]),
    ([[2, 3], [2]], ["a, 3", "a"]),
    ([[2, 3], [2]], ["a, 4", "a"]),
    ({"x": [2, 3]}, {"x": "b, _"}),
    ([[2, 3]], ["_, _"]),
    ([[2, 3]], [None]),
    ([[2, 3], [5]], ["a, b", "a + b"]),
    ([[2, 3], [4]], ["a, b", "c", "d"]),
    ({"x": [2, 3], "y": [1]}, {"x": "b, _"}),
    ({"x": [2, 3]}, {"z": "b, _"}),
    ([[2, 3]], ["a, b, c"]),
    ([[]], [""]),
]


def args_tree(desc):
    if isinstance(desc, dict):
        return {k: jax.ShapeDtypeStruct(tuple(v), jax.numpy.float32) for k, v in desc.items()}
    return tuple(jax.ShapeDtypeStruct(tuple(v), jax.numpy.float32) for v in desc)


def specs_tree(desc):
    if isinstance(desc, list):
        return tuple(specs_tree(d) for d in desc)
    if isinstance(desc, dict):
        return {k: specs_tree(v) for k, v in desc.items()}
    return desc


def args_specs_rows(rows):
    for i, (args, specs) in enumerate(ARGS_SPECS):
        r = outcome(lambda: str(jax.tree.map(lambda s: str(s.shape), export.symbolic_args_specs(args_tree(args), specs_tree(specs)))))
        r.update({"id": f"args_specs_{i:03d}", "spec": "", "constraints": [], "expr": ["args_specs", args, specs]})
        rows.append(r)


def spec_rows(rng, rows):
    for i, (spec, like) in enumerate(SPECS):
        r = outcome(lambda: str(export.symbolic_shape(spec, like=like)))
        r.update({"id": f"spec_{i:03d}", "spec": spec, "constraints": [], "expr": ["parse", spec, like]})
        rows.append(r)
    for r0 in list(rows):
        if r0["kind"] == "dim" and not r0["id"].startswith("spec") and rng.random() < 0.5:
            names = r0["spec"]
            text = names + ", " + r0["text"]
            r = outcome(lambda: str(export.symbolic_shape(text)))
            r.update({"id": "reparse_" + r0["id"], "spec": text, "constraints": [], "expr": ["parse", text, None]})
            rows.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--count", type=int, default=40)
    args = ap.parse_args()
    gg.preflight()
    rng = random.Random(args.seed)
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    os.makedirs(base, exist_ok=True)
    rows = []
    for si, (spec, constraints) in enumerate(SCOPES):
        nvars = len(spec.split(","))
        for k in range(args.count):
            e = gen_expr(rng, nvars, 3)
            if rng.random() < 0.35:
                e = [rng.choice(CMPS), e, gen_expr(rng, nvars, 2)]
            r = outcome(lambda: ev(e, export.symbolic_shape(spec, constraints=tuple(constraints))))
            r.update({"id": f"s{si}_{k:03d}", "spec": spec, "constraints": constraints, "expr": e})
            rows.append(r)
    args_specs_rows(rows)
    spec_rows(rng, rows)
    with open(os.path.join(base, "rows.json"), "w") as fh:
        json.dump({"jax": jax.__version__, "seed": args.seed, "rows": rows}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
