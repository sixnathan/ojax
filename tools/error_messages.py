#!/usr/bin/env python3
"""Count both-raise fuzz rows whose raw message equals JAX's. Run from outside the repo with reference JAX."""
import argparse
import codecs
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gen_fuzz_goldens as fuzz
import jax
import numpy as np
import parity_registry as registry


def ojax_messages(log, name):
    rows = {}
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 10 and f[4].startswith(name + ":") and f[0].endswith(":both_raise"):
                rows[(f[4], f[0])] = codecs.decode(f[9], "unicode_escape")
    return rows


def jax_message(call):
    try:
        call()
    except Exception as error:
        return str(error)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--name", default="fuzz")
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()
    ojax = ojax_messages(args.log, args.name)
    total = same = 0
    shown = 0
    for set_name, x64 in (("x64_off", False), ("x64_on", True)):
        jax.config.update("jax_enable_x64", x64)
        set_dir = os.path.join(args.goldens, args.name, set_name)
        with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as fh:
            cases = json.load(fh)["cases"]
        suite = args.name + ":" + set_name
        by_op = {}
        for case in cases:
            if "eager_error" in case or "jit_error" in case:
                by_op.setdefault(case["op"], []).append(case)
        for op_name, group in sorted(by_op.items()):
            op = registry.OPS[op_name]
            with np.load(os.path.join(set_dir, "ops", op_name + ".npz")) as stored:
                for case in group:
                    names = sorted(a["name"] for a in case["args"])
                    arrays = {k: stored[case["case_id"] + "/in_" + k] for k in names}
                    params = case["params"]

                    def eager():
                        devices = {k: jax.numpy.asarray(v) for k, v in arrays.items()}
                        return op(**devices, p=params)

                    def jitted():
                        return jax.jit(lambda *v: op(**dict(zip(names, v)), p=params))(*[arrays[k] for k in names])

                    for mode, call in (("eager", eager), ("jit", jitted)):
                        key = (suite, case["case_id"] + ":" + mode + ":both_raise")
                        if key not in ojax or mode + "_error" not in case:
                            continue
                        message = jax_message(call)
                        total += 1
                        if message == ojax[key]:
                            same += 1
                        elif shown < args.show:
                            shown += 1
                            print(key[1], "\n  JAX ", repr(message)[:200], "\n  OJAX", repr(ojax[key])[:200])
            jax.clear_caches()
    print(f"{args.name}: both-raise rows {total}, raw message identical {same}")


if __name__ == "__main__":
    main()
