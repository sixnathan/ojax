import json
import os
import re

import numpy as np

ALLOWED_DTYPES = (
    "bool",
    "int2",
    "int4",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint2",
    "uint4",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float4_e2m1fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

EXACT_DTYPES = frozenset(
    d for d in ALLOWED_DTYPES if d == "bool" or d.startswith("int") or d.startswith("uint")
)

WIDE64_DTYPES = frozenset(("float64", "int64", "uint64", "complex128"))

NUM = int(os.getenv("JAX_NUM_GENERATED_CASES", "10"))

SANITIZE_RE = re.compile(r"[ \"'\[\](){}<>=,._]+")


def sanitize(s):
    return SANITIZE_RE.sub("_", s)


def canonical_dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"


def params_key(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def is_exact_dtype(dtype):
    return dtype in EXACT_DTYPES


def is_wide64(dtype):
    return dtype in WIDE64_DTYPES


def _validate_dtype(dtype):
    if dtype not in ALLOWED_DTYPES:
        raise SystemExit("unknown dtype " + repr(dtype))
    return dtype


def _arg(index, shape, dtype, rng, weak):
    return {
        "name": "arg" + str(index),
        "shape": [int(d) for d in shape],
        "dtype": _validate_dtype(dtype),
        "rng": rng,
        "weak": bool(weak),
    }


def expand_record(rec):
    op = rec["op"]
    primitive = rec["primitive"]
    params = rec.get("params", {})
    if "args" in rec:
        args = [
            _arg(i, a["shape"], a["dtype"], a["rng"], a.get("weak", False))
            for i, a in enumerate(rec["args"])
        ]
        case = {
            "op": op,
            "primitive": primitive,
            "nargs": len(args),
            "args": args,
            "params": params,
        }
        if "tol_widen" in rec:
            case["tol_widen"] = rec["tol_widen"]
        if "x64_off_only" in rec:
            case["x64_off_only"] = rec["x64_off_only"]
        return [case]
    nargs = rec["nargs"]
    rng = rec["rng"]
    out = []
    for dtype in rec["dtypes"]:
        for shape in rec["shapes"]:
            args = [_arg(i, shape, dtype, rng, False) for i in range(nargs)]
            out.append(
                {
                    "op": op,
                    "primitive": primitive,
                    "nargs": nargs,
                    "args": args,
                    "params": params,
                }
            )
    return out


def arg_dtype_key(args):
    return ",".join(a["dtype"] + ("w" if a["weak"] else "") for a in args)


def arg_shape_key(args):
    return json.dumps([a["shape"] for a in args], separators=(",", ":"))


def sort_key(c):
    return (
        c["op"],
        c["primitive"],
        arg_dtype_key(c["args"]),
        arg_shape_key(c["args"]),
        params_key(c["params"]),
    )


def _shape_slug(shape):
    return "x".join(str(d) for d in shape) if shape else "s"


def base_slug(c):
    shapes = "_".join(_shape_slug(a["shape"]) for a in c["args"])
    dtypes = "_".join(a["dtype"] + ("w" if a["weak"] else "") for a in c["args"])
    return sanitize(c["op"] + "__" + shapes + "__" + dtypes)


def assign_ids(cases):
    counts = {}
    out = []
    for c in cases:
        base = base_slug(c)
        occ = counts.get(base, 0)
        counts[base] = occ + 1
        cc = dict(c)
        cc["case_id"] = base + "__" + "%03d" % occ
        out.append(cc)
    return out


def choice(n, m):
    return np.random.RandomState(42).choice(n, size=m, replace=False)


def sample(cands):
    n = len(cands)
    m = min(n, NUM)
    return [cands[i] for i in choice(n, m)]
