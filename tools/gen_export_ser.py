import argparse
import hashlib
import json
import os

import gen_goldens as gg
import jax
import jax.numpy as jnp
import numpy as np
from jax import export
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

MODULE = "export_ser"


def sds(shape, dtype, scope=None, constraints=()):
    if isinstance(shape, str):
        shape = export.symbolic_shape(shape, constraints=constraints, scope=scope)
    return jax.ShapeDtypeStruct(shape, np.dtype(dtype))


def f_sin(x):
    return jnp.sin(x)


def f_add_mul(x, y):
    return x + y, x * y


def f_sum0(x):
    return jnp.sum(jnp.sin(x), axis=0)


def f_dict(d):
    return {"s": d["a"] - d["b"], "p": [d["a"], jnp.cos(d["b"])]}


def f_matmul(x, y):
    return x @ y


def f_reshape(x):
    return jnp.reshape(x, (-1,)) * 2.0


def f_concat(x, y):
    return jnp.concatenate([x, y], axis=0)


def f_int(x):
    return x * 3 + 1


def f_unused(x, y):
    return jnp.exp(y)


def f_cumsum(x):
    return jnp.cumsum(x, axis=1)


def f_none(x):
    return (x, None)


CASES = [
    ("sin_f32_3", f_sin, lambda: ((sds((3,), "float32"),), {})),
    ("sin_f32_2x4", f_sin, lambda: ((sds((2, 4), "float32"),), {})),
    ("add_mul", f_add_mul, lambda: ((sds((3,), "float32"), sds((3,), "float32")), {})),
    ("add_mul_i32", f_add_mul, lambda: ((sds((2, 2), "int32"), sds((2, 2), "int32")), {})),
    ("dict_in", f_dict, lambda: (({"a": sds((2,), "float32"), "b": sds((2,), "float32")},), {})),
    ("matmul", f_matmul, lambda: ((sds((2, 3), "float32"), sds((3, 4), "float32")), {})),
    ("unused_arg", f_unused, lambda: ((sds((2,), "float32"), sds((3,), "float32")), {})),
    ("int_affine", f_int, lambda: ((sds((5,), "int32"),), {})),
    ("none_out", f_none, lambda: ((sds((2,), "float32"),), {})),
    ("sym_sum0", f_sum0, lambda: ((sds("b, 3", "float32"),), {})),
    ("sym_sin_2d", f_sin, lambda: ((sds("b, c", "float32"),), {})),
    ("sym_reshape", f_reshape, lambda: ((sds("b, 4", "float32"),), {})),
    ("sym_cumsum", f_cumsum, lambda: ((sds("b, c", "float32"),), {})),
]


def sym_pair():
    scope = export.SymbolicScope()
    return ((sds("b, 2", "float32", scope), sds("b, 2", "float32", scope)), {})


def sym_concat():
    scope = export.SymbolicScope()
    return ((sds("a, 3", "float32", scope), sds("b, 3", "float32", scope)), {})


def sym_constrained():
    return ((sds("b, 3", "float32", constraints=("b >= 2",)),), {})


CASES += [
    ("sym_add_mul", f_add_mul, sym_pair),
    ("sym_concat", f_concat, sym_concat),
    ("sym_constrained", f_sum0, sym_constrained),
]

CALLS = {
    "sin_f32_3": [[((3,), "float32")]],
    "sin_f32_2x4": [[((2, 4), "float32")], [((2, 3), "float32")], [((4, 2), "float32")], [((8,), "float32")], [((2, 4), "float64")]],
    "add_mul": [[((3,), "float32"), ((3,), "float32")], [((3,), "float32")]],
    "add_mul_i32": [[((2, 2), "int32"), ((2, 2), "int32")], [((2, 2), "int32"), ((2, 2), "float32")]],
    "dict_in": [[((2,), "float32"), ((2,), "float32")]],
    "matmul": [[((2, 3), "float32"), ((3, 4), "float32")]],
    "unused_arg": [[((2,), "float32"), ((3,), "float32")]],
    "int_affine": [[((5,), "int32")]],
    "none_out": [[((2,), "float32")]],
    "sym_sum0": [[((4, 3), "float32")], [((1, 3), "float32")], [((0, 3), "float32")], [((4, 2), "float32")], [((4,), "float32")], [((4, 3), "int32")]],
    "sym_sin_2d": [[((2, 5), "float32")], [((3, 1), "float32")]],
    "sym_reshape": [[((3, 4), "float32")]],
    "sym_cumsum": [[((2, 3), "float32")]],
    "sym_add_mul": [[((3, 2), "float32"), ((3, 2), "float32")], [((3, 2), "float32"), ((4, 2), "float32")]],
    "sym_concat": [[((2, 3), "float32"), ((4, 3), "float32")]],
    "sym_constrained": [[((3, 3), "float32")], [((1, 3), "float32")], [((2, 3), "float32")]],
}


def make_input(shape, dtype, k):
    n = int(np.prod(shape)) if shape else 1
    if np.dtype(dtype).kind in "iu":
        vals = (np.arange(n) * (k + 2) - 3).astype(dtype)
    else:
        vals = (np.linspace(-1.5, 2.0, n) + 0.25 * k).astype(dtype)
    return vals.reshape(shape)


def enc_array(a):
    a = np.asarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "hex": a.tobytes().hex()}


def enc_input(a):
    out = enc_array(a)
    out["values"] = [float(v) if a.dtype.kind == "f" else int(v) for v in a.reshape(-1)]
    return out


def tree_str(t):
    return str(t)


def fields_text(e):
    lines = [
        "fun_name=" + e.fun_name,
        "in_tree=" + tree_str(e.in_tree),
        "out_tree=" + tree_str(e.out_tree),
        "in_avals=" + ";".join(a.str_short() for a in e.in_avals),
        "out_avals=" + ";".join(a.str_short() for a in e.out_avals),
        "platforms=" + ",".join(e.platforms),
        "nr_devices=" + str(e.nr_devices),
        "calling_convention_version=" + str(e.calling_convention_version),
        "module_kept_var_idx=" + ",".join(str(i) for i in e.module_kept_var_idx),
        "uses_global_constants=" + str(e.uses_global_constants),
        "disabled_safety_checks=" + ",".join(str(c) for c in e.disabled_safety_checks),
        "ordered_effects=" + str(len(e.ordered_effects)),
        "unordered_effects=" + str(len(e.unordered_effects)),
        "in_shardings=" + ",".join(str(s) for s in e._in_named_shardings),
        "out_shardings=" + ",".join(str(s) for s in e._out_named_shardings),
        "has_vjp=" + str(e.has_vjp()),
    ]
    return "\n".join(lines)


def plain_text(e):
    with mlir.make_ir_context() as ctx:
        m = ir.Module.parse(e.mlir_module())
        return m.operation.get_asm(enable_debug_info=False)


def leaves_to_args(name, leaves):
    if name == "dict_in":
        return ({"a": leaves[0], "b": leaves[1]},)
    return tuple(leaves)


def call_row(e, name, leaves):
    try:
        out = e.call(*leaves_to_args(name, leaves))
        flat = [o for o in jax.tree_util.tree_leaves(out)]
        return {"kind": "ok", "outputs": [enc_array(o) for o in flat]}
    except Exception as ex:
        return {"kind": "error", "class": type(ex).__name__, "message": str(ex)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    os.makedirs(base, exist_ok=True)
    rows = []
    for name, fn, spec in CASES:
        a, kw = spec()
        e = export.export(jax.jit(fn))(*a, **kw)
        ser = bytes(e.serialize())
        ser_again = bytes(e.serialize())
        back = export.deserialize(bytearray(ser))
        reser = bytes(back.serialize())
        row = {
            "id": name,
            "serialized": ser.hex(),
            "sha256": hashlib.sha256(ser).hexdigest(),
            "deterministic": ser == ser_again,
            "reserialized_same": reser == ser,
            "fields": fields_text(back),
            "text": plain_text(back),
            "calls": [],
        }
        ser_vjp = bytes(e.serialize(vjp_order=1)) if not name.startswith("sym") and name not in ("add_mul_i32", "int_affine", "none_out") else None
        if ser_vjp is not None:
            row["serialized_vjp1"] = ser_vjp.hex()
            row["reserialized_vjp1_same"] = bytes(export.deserialize(bytearray(ser_vjp)).serialize(vjp_order=1)) == ser_vjp
        for k, sig in enumerate(CALLS[name]):
            leaves = [make_input(shape, dt, k + j) for j, (shape, dt) in enumerate(sig)]
            r = call_row(back, name, leaves)
            r["inputs"] = [enc_input(x) for x in leaves]
            row["calls"].append(r)
        rows.append(row)
    with open(os.path.join(base, "cases.json"), "w") as fh:
        json.dump({"jax": jax.__version__, "rows": rows}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
