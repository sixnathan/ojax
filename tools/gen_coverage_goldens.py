import json
import os
import sys

import compare_text as ct
import coverage_registry as registry
import emit_common as ec
import gen_goldens as gg
import gen_parity_goldens as gp
import jax
import numpy as np
from jax._src import core as jax_core

MODULE = "coverage"


def expand():
    cases = []
    for op, inputs, params in registry.all_cases():
        for dtype in ["float32", "float64"]:
            cases.append(
                {
                    "case_id": gp.case_id(op, inputs, params, dtype),
                    "op": op,
                    "dtype": dtype,
                    "inputs": inputs,
                    "params": params,
                    "primitive": "coverage." + op,
                }
            )
    cases.sort(key=lambda c: c["case_id"])
    return cases


def text_reference(base, out):
    with open(out, "w", encoding="utf-8") as sink:
        for set_name, x64 in ct.SETS:
            jax.config.update("jax_enable_x64", x64)
            set_dir = os.path.join(base, set_name)
            with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as fh:
                cases = json.load(fh)["cases"]
            for i, case in enumerate(cases):
                names = [a["name"] for a in case["args"]]
                with np.load(os.path.join(set_dir, "inputs", case["case_id"] + ".npz")) as z:
                    arrays = [z[n] for n in names]
                op = registry.OPS[case["op"]]
                params = case["params"]

                def fn(*xs, _op=op, _names=names, _params=params):
                    return _op(**dict(zip(_names, xs)), p=_params)

                row = ct._reference_case(fn, arrays, jax, jax_core)
                row.update({"set": MODULE + "/" + set_name, "case": case["case_id"]})
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                if i % 50 == 0:
                    jax.clear_caches()


def main():
    gg.preflight()
    gp.registry = registry
    gp.MODULE = MODULE
    cases = expand()
    with open(os.path.join(gg.ROOT, "spec", MODULE + ".cases.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps({"cases": cases}))
    base = os.path.join(gg.ROOT, "goldens", MODULE)
    n_off = gp.gen_set(cases, False, os.path.join(base, "x64_off"))
    n_on = gp.gen_set(cases, True, os.path.join(base, "x64_on"))
    text_reference(base, os.path.join(base, "text_ref.jsonl"))
    sys.stdout.write(MODULE + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
