import argparse
import json
import os

import numpy as np


def cases(jax, jnp):
    from jax.extend import core as jc
    from jax._src import core as sc
    from jax._src import source_info_util

    def strip(j):
        empty = source_info_util.new_source_info()
        return j.replace(eqns=[e.replace(source_info=empty) for e in j.eqns])

    def var(shape, dtype):
        return sc.Var(sc.ShapedArray(shape, np.dtype(dtype)))

    small = strip(jax.make_jaxpr(lambda x: jnp.sin(x) * 2.0)(np.ones(3, np.float32)).jaxpr)

    def chain(x):
        for _ in range(30):
            x = jnp.sin(x)
        return x

    long = strip(jax.make_jaxpr(chain)(np.ones(3, np.float32)).jaxpr)
    e0, e1 = small.eqns
    bad = var((3,), "int32")
    out = {}
    out["good"] = small
    out["out_type"] = small.replace(eqns=[e0.replace(outvars=[bad]), e1.replace(invars=[bad, e1.invars[1]])])
    out["out_shape"] = small.replace(eqns=[e0.replace(outvars=[var((2,), "float32")])] + [e1])
    out["undefined"] = small.replace(eqns=[e1])
    out["rebound"] = small.replace(eqns=[e0, e0, e1])
    out["rebound_input"] = small.replace(invars=[small.invars[0], small.invars[0]])
    out["undefined_out"] = small.replace(outvars=[var((3,), "float32")])
    for k in (0, 5, 15, 29):
        e = long.eqns[k]
        eqns = list(long.eqns)
        eqns[k] = e.replace(outvars=[var((3,), "float64")])
        out[f"long_{k}"] = long.replace(eqns=eqns)
    out["long_undefined_25"] = long.replace(eqns=long.eqns[:20] + long.eqns[21:])
    return {k: (lambda j=j: jc.check_jaxpr(j)) for k, j in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(args.root or os.path.join(here, "..", "goldens"), "jaxpr_check")
    os.makedirs(base, exist_ok=True)
    rows = []
    for name, thunk in cases(jax, jnp).items():
        try:
            thunk()
            row = {"kind": "ok", "text": ""}
        except Exception as e:
            row = {"kind": "raise", "class": type(e).__name__, "message": str(e), "text": ""}
        row["id"] = name
        rows.append(row)
    with open(os.path.join(base, "rows.json"), "w", encoding="utf-8") as fh:
        json.dump({"jax": jax.__version__, "rows": rows}, fh, indent=1, sort_keys=True)
    print("jaxpr_check", len(rows), "rows")


if __name__ == "__main__":
    main()
