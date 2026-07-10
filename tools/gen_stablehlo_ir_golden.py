import json
import struct
import sys

import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

DTYPES = [
    ("f32", np.float32, "F32"),
    ("f64", np.float64, "F64"),
    ("i32", np.int32, "I32"),
    ("i64", np.int64, "I64"),
    ("i1", np.bool_, "Bool"),
    ("ui32", np.uint32, "Uint32"),
]

SHAPES = [[], [3], [2, 3], [4, 5, 6]]

FLOAT_CASES_F32 = [
    0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 0.5, 0.25, 2.5, 10.0, 100.0, 255.0,
    -2.75, 0.333333, 1.5, 7.0, 42.0, 0.001, 1000.0, 100000.0, 1000000.0,
    12345.0, 123456.0, 1234567.0, 16777216.0, 1e-10, 1e20, 6.022e23, 1e-45,
    3.14159265358979, 2.718281828, 123456.789, float("inf"), float("-inf"),
    float("nan"),
]

FLOAT_CASES_F64 = [
    0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.5, 2.5, 100.0, 255.0, -2.75, 0.001,
    1000.0, 1234567.0, 1e-10, 1e20, 3.14159265358979, 2.718281828, 123456.789,
    float("inf"), float("-inf"), float("nan"),
]

INT_CASES = [
    ("i32", np.int32, 0), ("i32", np.int32, 1), ("i32", np.int32, -3),
    ("i32", np.int32, 127), ("i32", np.int32, 2147483647),
    ("i64", np.int64, 5), ("i64", np.int64, -100),
    ("ui32", np.uint32, 7), ("ui32", np.uint32, 4000000000),
]


def scalar_attr(ctx, value, npdt, mlir_t):
    arr = np.asarray(value, npdt).reshape(())
    attr = ir.DenseElementsAttr.get(arr, type=ir.RankedTensorType.get([], mlir_t))
    text = str(attr)
    return text[text.find("<") + 1 : text.find(">")]


def f32_bits(v):
    return "0x%08x" % (struct.unpack("<I", struct.pack("<f", np.float32(v)))[0])


def f64_bits(v):
    return "0x%016x" % (struct.unpack("<Q", struct.pack("<d", v))[0])


def main():
    out = {}
    with ir.Context() as ctx, ir.Location.unknown():
        stablehlo.register_dialect(ctx)
        mlir_types = {
            "f32": ir.F32Type.get(),
            "f64": ir.F64Type.get(),
            "i32": ir.IntegerType.get_signless(32),
            "i64": ir.IntegerType.get_signless(64),
            "i1": ir.IntegerType.get_signless(1),
            "ui32": ir.IntegerType.get_unsigned(32),
        }

        element_types = {}
        tensor_types = []
        for name, _npdt, dtag in DTYPES:
            element_types[dtag] = name
            mt = mlir_types[name]
            for shape in SHAPES:
                t = str(ir.RankedTensorType.get(shape, mt))
                tensor_types.append({"dtype": dtag, "shape": shape, "text": t})
        out["element_types"] = element_types
        out["tensor_types"] = sorted(
            tensor_types, key=lambda e: (e["dtype"], e["shape"])
        )

        floats = []
        for v in FLOAT_CASES_F32:
            floats.append(
                {
                    "dtype": "F32",
                    "bits": f32_bits(v),
                    "text": scalar_attr(ctx, v, np.float32, mlir_types["f32"]),
                }
            )
        for v in FLOAT_CASES_F64:
            floats.append(
                {
                    "dtype": "F64",
                    "bits": f64_bits(v),
                    "text": scalar_attr(ctx, v, np.float64, mlir_types["f64"]),
                }
            )
        out["float_literals"] = sorted(
            floats, key=lambda e: (e["dtype"], e["bits"])
        )

        ints = []
        for name, npdt, v in INT_CASES:
            ints.append(
                {
                    "dtype": name,
                    "value": str(int(v)),
                    "text": scalar_attr(ctx, v, npdt, mlir_types[name]),
                }
            )
        out["int_literals"] = sorted(
            ints, key=lambda e: (e["dtype"], e["value"])
        )

        bools = []
        for v in (True, False):
            bools.append(
                {
                    "value": v,
                    "text": scalar_attr(ctx, v, np.bool_, mlir_types["i1"]),
                }
            )
        out["bool_literals"] = sorted(bools, key=lambda e: e["value"])

    json.dump(
        out, sys.stdout, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
