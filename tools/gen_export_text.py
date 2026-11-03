import argparse
import json
import os

import gen_goldens as gg
import jax
import jax.numpy as jnp
import numpy as np
from jax import export, lax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

MODULE = "export_text"


def conv_same(x, w):
    return lax.conv(x, w, (1, 1), "SAME")


def conv_valid_stride(x, w):
    return lax.conv(x, w, (2, 1), "VALID")


def conv_valid(x, w):
    return lax.conv(x, w, (1, 1), "VALID")


def cumsum1(x):
    return jnp.cumsum(x, axis=1)


def maxpool(x):
    return lax.reduce_window(x, -jnp.inf, lax.max, (2, 2), (1, 1), "VALID")


def sumpool_stride(x):
    return lax.reduce_window(x, 0.0, lax.add, (2, 2), (2, 1), "VALID")


def sumpool_pad(x):
    return lax.reduce_window(x, 0.0, lax.add, (2, 2), (1, 1), ((1, 0), (0, 0)))


def maxpool_grad(x):
    return jax.grad(lambda y: maxpool(y).sum())(x)


def batch_dot(a, b):
    return jnp.einsum("bij,bjk->bik", a, b)


def matmul(a, b):
    return a @ b


def dot_highest(a, b):
    return jnp.dot(a, b, precision=lax.Precision.HIGHEST)


CASES = [
    ("conv_same", conv_same, [(1, 1, 5, 5), (1, 1, 3, 3)]),
    ("conv_valid_stride", conv_valid_stride, [(1, 1, 5, 5), (1, 1, 3, 3)]),
    ("conv_valid", conv_valid, [(1, 1, 5, 5), (1, 1, 3, 3)]),
    ("cumsum1", cumsum1, [(2, 3)]),
    ("maxpool", maxpool, [(4, 4)]),
    ("sumpool_stride", sumpool_stride, [(4, 4)]),
    ("sumpool_pad", sumpool_pad, [(4, 4)]),
    ("maxpool_grad", maxpool_grad, [(4, 4)]),
    ("batch_dot", batch_dot, [(2, 2, 3), (2, 3, 2)]),
    ("matmul", matmul, [(2, 3), (3, 4)]),
    ("dot_highest", dot_highest, [(2, 3), (3, 4)]),
]


def plain_text(e):
    with mlir.make_ir_context():
        return ir.Module.parse(e.mlir_module()).operation.get_asm(enable_debug_info=False)


def rows_for(dtype):
    rows = []
    for name, f, shapes in CASES:
        args = [jax.ShapeDtypeStruct(s, np.dtype(dtype)) for s in shapes]
        row = {"id": name, "dtype": dtype, "shapes": [list(s) for s in shapes]}
        try:
            row["text"] = plain_text(export.export(jax.jit(f))(*args))
        except Exception as error:
            row["error_class"] = type(error).__name__
            row["message"] = str(error)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    os.makedirs(base, exist_ok=True)
    for x64 in (False, True):
        jax.config.update("jax_enable_x64", x64)
        rows = rows_for("float32")
        if x64:
            rows += rows_for("float64")
        with open(os.path.join(base, ("x64_on" if x64 else "x64_off") + ".json"), "w") as fh:
            json.dump({"jax": jax.__version__, "x64": x64, "rows": rows}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
