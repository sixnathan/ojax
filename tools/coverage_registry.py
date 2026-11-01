import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

SINGLE_PRECISION_ONLY = set()
LOOSE_TOLERANCE = {}


def compares_at_single_precision(op, arrays):
    return False


def _k(p):
    return jr.key(p["seed"])


def _sh(p):
    return tuple(p["shape"])


def _random_ops():
    ops = {}
    for name in "normal exponential cauchy laplace logistic".split():
        ops["cv_r_" + name] = lambda a, p, _n=name: getattr(jr, _n)(_k(p), _sh(p), dtype=p["dtype"])
    ops["cv_r_uniform"] = lambda a, p: jr.uniform(_k(p), _sh(p), dtype=p["dtype"], minval=-1.0, maxval=3.0)
    ops["cv_r_truncated_normal"] = lambda a, p: jr.truncated_normal(_k(p), -1.0, 2.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_randint"] = lambda a, p: jr.randint(_k(p), _sh(p), -3, 50, dtype=p["dtype"])
    ops["cv_r_gumbel"] = lambda a, p: jr.gumbel(_k(p), _sh(p), dtype=p["dtype"], mode=p.get("mode"))
    ops["cv_r_bernoulli"] = lambda a, p: jr.bernoulli(_k(p), 0.4, _sh(p), mode=p["mode"])
    ops["cv_r_gamma"] = lambda a, p: jr.gamma(_k(p), 2.5, _sh(p), dtype=p["dtype"])
    ops["cv_r_loggamma"] = lambda a, p: jr.loggamma(_k(p), 2.5, _sh(p), dtype=p["dtype"])
    ops["cv_r_beta"] = lambda a, p: jr.beta(_k(p), 2.0, 3.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_chisquare"] = lambda a, p: jr.chisquare(_k(p), 3.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_t"] = lambda a, p: jr.t(_k(p), 4.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_f"] = lambda a, p: jr.f(_k(p), 3.0, 5.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_pareto"] = lambda a, p: jr.pareto(_k(p), 2.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_rayleigh"] = lambda a, p: jr.rayleigh(_k(p), 1.5, _sh(p), dtype=p["dtype"])
    ops["cv_r_wald"] = lambda a, p: jr.wald(_k(p), 1.5, _sh(p), dtype=p["dtype"])
    ops["cv_r_lognormal"] = lambda a, p: jr.lognormal(_k(p), 0.5, _sh(p), dtype=p["dtype"])
    ops["cv_r_geometric"] = lambda a, p: jr.geometric(_k(p), 0.3, _sh(p), dtype=p["dtype"])
    ops["cv_r_poisson"] = lambda a, p: jr.poisson(_k(p), 3.0, _sh(p), dtype=p["dtype"])
    ops["cv_r_rademacher"] = lambda a, p: jr.rademacher(_k(p), _sh(p), dtype=p["dtype"])
    ops["cv_r_binomial"] = lambda a, p: jr.binomial(_k(p), 10.0, 0.3, _sh(p), dtype=p["dtype"])
    ops["cv_r_dirichlet"] = lambda a, p: jr.dirichlet(_k(p), jnp.asarray([1.0, 2.0, 3.0], p["dtype"]), _sh(p), dtype=p["dtype"])
    ops["cv_r_permutation"] = lambda a, p: jr.permutation(_k(p), a, axis=p.get("axis", 0), independent=p.get("independent", False))
    ops["cv_r_choice"] = lambda a, p: jr.choice(_k(p), a, _sh(p), replace=p.get("replace", True), axis=p.get("axis", 0))
    ops["cv_r_choice_p"] = lambda a, b, p: jr.choice(_k(p), a, _sh(p), replace=p.get("replace", True), p=jnp.abs(b), axis=p.get("axis", 0), mode=p.get("mode"))
    ops["cv_r_choice_int"] = lambda a, p: jr.choice(_k(p), p["n"], _sh(p), replace=p.get("replace", True))
    ops["cv_r_categorical"] = lambda a, p: jr.categorical(_k(p), a, axis=p.get("axis", -1), shape=None if "shape" not in p else _sh(p), replace=p.get("replace", True), mode=p.get("mode"))
    ops["cv_r_multinomial"] = lambda a, p: jr.multinomial(_k(p), p["n"], jnp.abs(a) / jnp.sum(jnp.abs(a), axis=-1, keepdims=True), shape=None if "shape" not in p else _sh(p), dtype=p.get("dtype"), unroll=p.get("unroll", 1))
    return ops


def _random_cases():
    r = lambda shape: {"a": {"kind": "real", "shape": shape}}
    one = r([1])
    out = []
    for dt in ["float16", "float32", "bfloat16"]:
        for op in "normal uniform truncated_normal exponential cauchy laplace logistic gumbel gamma loggamma beta chisquare t f pareto rayleigh wald lognormal rademacher binomial".split():
            if op == "rademacher":
                continue
            out.append(("cv_r_" + op, one, {"seed": 21, "shape": [3], "dtype": dt}))
        out.append(("cv_r_dirichlet", one, {"seed": 22, "shape": [2], "dtype": dt}))
    out.append(("cv_r_normal", one, {"seed": 23, "shape": [3], "dtype": "complex64"}))
    for dt in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        out.append(("cv_r_randint", one, {"seed": 24, "shape": [4], "dtype": dt}))
    for dt in ["int8", "int16", "int32"]:
        out.append(("cv_r_rademacher", one, {"seed": 25, "shape": [4], "dtype": dt}))
        out.append(("cv_r_geometric", one, {"seed": 26, "shape": [4], "dtype": dt}))
        out.append(("cv_r_poisson", one, {"seed": 27, "shape": [4], "dtype": dt}))
    for mode in ["low", "high", "highest"]:
        for dt in ["float32", "float16"]:
            out.append(("cv_r_gumbel", one, {"seed": 28, "shape": [5], "dtype": dt, "mode": mode}))
    for mode in ["low", "high"]:
        out.append(("cv_r_bernoulli", one, {"seed": 29, "shape": [6], "mode": mode}))
    for shape, axis, ind in [([5], 0, False), ([3, 4], 0, False), ([3, 4], 1, False), ([3, 4], -1, False), ([3, 4], 0, True), ([3, 4], 1, True)]:
        out.append(("cv_r_permutation", r(shape), {"seed": 30, "axis": axis, "independent": ind}))
    for shape, axis, draw, rep in [([5], 0, [2], True), ([5], 0, [3], False), ([3, 4], 0, [2], True), ([3, 4], 1, [2], False), ([3, 4], 1, [2, 2], True), ([3, 4], 0, [1, 2], False)]:
        out.append(("cv_r_choice", r(shape), {"seed": 31, "shape": draw, "axis": axis, "replace": rep}))
    two = lambda shape, n: {"a": {"kind": "real", "shape": shape}, "b": {"kind": "positive", "shape": [n]}}
    for shape, n, axis, draw, rep, mode in [([5], 5, 0, [3], True, None), ([5], 5, 0, [3], False, None), ([3, 4], 4, 1, [2], True, None), ([3, 4], 3, 0, [2], False, None), ([5], 5, 0, [2], False, "high")]:
        params = {"seed": 32, "shape": draw, "axis": axis, "replace": rep}
        if mode:
            params["mode"] = mode
        out.append(("cv_r_choice_p", two(shape, n), params))
    for n, draw, rep in [(6, [3], True), (6, [3], False), (6, [2, 2], True)]:
        out.append(("cv_r_choice_int", one, {"seed": 33, "n": n, "shape": draw, "replace": rep}))
    for logits, params in [
        ([5], {}),
        ([3, 4], {}),
        ([3, 4], {"axis": 0}),
        ([3, 4], {"shape": [2, 3]}),
        ([4], {"shape": [2]}),
        ([5], {"shape": [3], "replace": False}),
        ([3, 4], {"shape": [2, 3], "replace": False}),
        ([3, 4], {"shape": [2, 4], "axis": 0, "replace": False}),
        ([5], {"mode": "high"}),
    ]:
        out.append(("cv_r_categorical", r(logits), dict(params, seed=34)))
    for shape, params in [([3], {"n": 10.0}), ([2, 3], {"n": 5.0}), ([3], {"n": 10.0, "shape": [2, 3]}), ([3], {"n": 4.0, "unroll": 2}), ([3], {"n": 4.0, "dtype": "float32"})]:
        out.append(("cv_r_multinomial", r(shape), dict(params, seed=35)))
    return out


def _gather_ops():
    return {
        "cv_take_flags": lambda a, p: jnp.take(a, jnp.asarray(p["idx"]), axis=p.get("axis"), mode=p.get("mode"), unique_indices=p.get("unique", False), indices_are_sorted=p.get("sorted", False)),
    }


def _gather_cases():
    r = lambda shape: {"a": {"kind": "real", "shape": shape}}
    out = []
    for idx, unique, srt, mode in [([0, 2], True, False, None), ([0, 2], False, True, None), ([0, 2], True, True, None), ([0, 2], True, True, "clip"), ([2, 0], True, False, "wrap"), ([0, 2], False, True, "fill")]:
        params = {"idx": idx, "unique": unique, "sorted": srt, "axis": 0}
        if mode:
            params["mode"] = mode
        out.append(("cv_take_flags", r([4, 2]), params))
    return out


OPS = {}
OPS.update(_random_ops())
OPS.update(_gather_ops())


def all_cases():
    return _random_cases() + _gather_cases()
