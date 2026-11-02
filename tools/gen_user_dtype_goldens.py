import argparse
import contextlib
import io
import json
import warnings

import numpy as np

DTYPES = ["float64", "float32", "float16", "int64", "int32", "uint64",
          "uint32", "complex128"]

FUNCTIONS = [
    "bits", "randint", "uniform", "normal", "truncated_normal",
    "exponential", "cauchy", "laplace", "logistic", "gumbel", "pareto",
    "rayleigh", "weibull_min", "lognormal", "triangular", "wald",
    "geometric", "rademacher", "gamma", "loggamma", "beta", "chisquare",
    "t", "f", "generalized_normal", "poisson", "binomial", "maxwell",
    "double_sided_maxwell", "ball", "orthogonal",
]


def encode(out):
    a = np.asarray(out)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "hex": a.tobytes().hex()}


def run_case(thunk):
    notices = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(notices):
                out = thunk()
            result = {"kind": "ok", "outs": [encode(out)]}
        except Exception as e:
            result = {"kind": "raise", "class": type(e).__name__,
                      "message": str(e)}
    result["warnings"] = [[w.category.__name__, str(w.message)]
                          for w in caught]
    result["notices"] = [l for l in notices.getvalue().splitlines() if l]
    return result


def call(jax, name, dtype):
    r = jax.random
    k = r.key(1)
    s = (3,)
    calls = {
        "bits": lambda: r.bits(k, s, dtype),
        "randint": lambda: r.randint(k, s, 0, 10, dtype),
        "uniform": lambda: r.uniform(k, s, dtype),
        "normal": lambda: r.normal(k, s, dtype),
        "truncated_normal": lambda: r.truncated_normal(k, -1.0, 1.0, s, dtype),
        "exponential": lambda: r.exponential(k, s, dtype),
        "cauchy": lambda: r.cauchy(k, s, dtype),
        "laplace": lambda: r.laplace(k, s, dtype),
        "logistic": lambda: r.logistic(k, s, dtype),
        "gumbel": lambda: r.gumbel(k, s, dtype),
        "pareto": lambda: r.pareto(k, 2.0, s, dtype),
        "rayleigh": lambda: r.rayleigh(k, 1.5, s, dtype),
        "weibull_min": lambda: r.weibull_min(k, 1.5, 2.0, s, dtype),
        "lognormal": lambda: r.lognormal(k, 0.5, s, dtype),
        "triangular": lambda: r.triangular(k, 0.0, 0.5, 1.0, s, dtype),
        "wald": lambda: r.wald(k, 1.5, s, dtype),
        "geometric": lambda: r.geometric(k, 0.5, s, dtype),
        "rademacher": lambda: r.rademacher(k, s, dtype),
        "gamma": lambda: r.gamma(k, 2.0, s, dtype),
        "loggamma": lambda: r.loggamma(k, 2.0, s, dtype),
        "beta": lambda: r.beta(k, 2.0, 3.0, s, dtype),
        "chisquare": lambda: r.chisquare(k, 3.0, s, dtype),
        "t": lambda: r.t(k, 3.0, s, dtype),
        "f": lambda: r.f(k, 2.0, 3.0, s, dtype),
        "generalized_normal": lambda: r.generalized_normal(k, 2.0, s, dtype),
        "poisson": lambda: r.poisson(k, 2.0, s, dtype),
        "binomial": lambda: r.binomial(k, 5.0, 0.5, s, dtype),
        "maxwell": lambda: r.maxwell(k, s, dtype),
        "double_sided_maxwell":
            lambda: r.double_sided_maxwell(k, 0.0, 1.0, s, dtype),
        "ball": lambda: r.ball(k, 2, 2, s, dtype),
        "orthogonal": lambda: r.orthogonal(k, 3, (), dtype),
    }
    return calls[name]


LINALG = ["cholesky", "lu", "qr", "eigh", "eig", "svd", "schur",
          "hessenberg", "tridiagonal", "tridiagonal_solve",
          "triangular_solve"]


def linalg_call(jax, name, dtype, mode):
    import jax.numpy as jnp
    from jax.lax import linalg as L
    a = jnp.asarray(np.eye(3) + 0.1, dtype=dtype)
    calls = {
        "cholesky": lambda a: L.cholesky(a),
        "lu": lambda a: L.lu(a)[0],
        "qr": lambda a: L.qr(a)[0],
        "eigh": lambda a: L.eigh(a)[0],
        "eig": lambda a: L.eig(a)[0],
        "svd": lambda a: L.svd(a)[1],
        "schur": lambda a: L.schur(a)[0],
        "hessenberg": lambda a: L.hessenberg(a)[0],
        "tridiagonal": lambda a: L.tridiagonal(a)[0],
        "tridiagonal_solve": lambda a: L.tridiagonal_solve(a[0], a[1], a[2], a),
        "triangular_solve": lambda a: L.triangular_solve(a, a, left_side=True,
                                                        lower=True),
    }
    f = calls[name]
    if mode == "jit":
        f = jax.jit(f)
    return lambda: f(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    rows = []
    for name in FUNCTIONS:
        for dtype in DTYPES:
            r = run_case(call(jax, name, np.dtype(dtype)))
            r.update({"id": f"user_dtype_{name}_{dtype}",
                      "group": "user_dtype", "fn": name, "dtype": dtype})
            rows.append(r)
    for name in LINALG:
        for dtype in ["float16", "bfloat16"]:
            for mode in ["eager", "jit"]:
                import jax.numpy as jnp
                r = run_case(linalg_call(jax, name, jnp.dtype(dtype), mode))
                r.update({"id": f"linalg_half_{name}_{dtype}_{mode}",
                          "group": "linalg_half", "fn": name, "dtype": dtype,
                          "mode": mode})
                rows.append(r)
    with open(args.out, "w") as fh:
        json.dump({"jax": jax.__version__, "x64": args.x64, "rows": rows},
                  fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
