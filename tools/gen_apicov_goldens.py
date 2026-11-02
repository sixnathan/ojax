import json
import os
import sys

import apicov_registry as registry
import compare_text as ct
import emit_common as ec
import gen_coverage_goldens as gcg
import gen_goldens as gg
import gen_parity_goldens as gp
import jax
import numpy as np
from jax._src import core as jax_core

MODULE = "apicov"


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
                    "primitive": MODULE + "." + op,
                }
            )
    cases.sort(key=lambda c: c["case_id"])
    return cases


def main():
    gg.preflight()
    gp.registry = registry
    gp.MODULE = MODULE
    gcg.registry = registry
    gcg.MODULE = MODULE
    cases = expand()
    with open(os.path.join(gg.ROOT, "spec", MODULE + ".cases.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps({"cases": cases}))
    base = os.path.join(gg.ROOT, "goldens", MODULE)
    n_off = gp.gen_set(cases, False, os.path.join(base, "x64_off"))
    n_on = gp.gen_set(cases, True, os.path.join(base, "x64_on"))
    gcg.text_reference(base, os.path.join(base, "text_ref.jsonl"))
    sys.stdout.write(MODULE + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
