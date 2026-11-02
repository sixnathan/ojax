import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

SINGLE_PRECISION_ONLY = set()
LOOSE_TOLERANCE = {}


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


def _transform_ops():
    def f_scalar(x):
        return jnp.sum(jnp.sin(x) * x), x * 2.0

    def f_vec(x):
        return jnp.sin(x) * x, x * 2.0

    def f_holo(x):
        return jnp.sum(x * x * x)

    return {
        "cv_t_grad_aux": lambda a, p: jax.grad(f_scalar, has_aux=True)(a),
        "cv_t_vg_aux": lambda a, p: jax.value_and_grad(f_scalar, has_aux=True)(a),
        "cv_t_jvp_aux": lambda a, p: jax.jvp(f_vec, (a,), (jnp.ones_like(a),), has_aux=True),
        "cv_t_vjp_aux": lambda a, p: (lambda r: (r[0], r[1](jnp.ones_like(a))[0], r[2]))(jax.vjp(f_vec, a, has_aux=True)),
        "cv_t_linearize_aux": lambda a, p: (lambda r: (r[0], r[1](jnp.ones_like(a)), r[2]))(jax.linearize(f_vec, a, has_aux=True)),
        "cv_t_jacfwd_aux": lambda a, p: jax.jacfwd(f_vec, has_aux=True)(a),
        "cv_t_jacrev_aux": lambda a, p: jax.jacrev(f_vec, has_aux=True)(a),
        "cv_t_hessian_aux": lambda a, p: jax.hessian(f_scalar, has_aux=True)(a),
        "cv_t_jacfwd": lambda a, p: jax.jacfwd(lambda x: jnp.sin(x) * x)(a),
        "cv_t_jacrev": lambda a, p: jax.jacrev(lambda x: jnp.sin(x) * x)(a),
        "cv_t_hessian": lambda a, p: jax.hessian(lambda x: jnp.sum(jnp.sin(x) * x))(a),
        "cv_t_jacfwd_scalar": lambda a, p: jax.jacfwd(lambda x: jnp.sin(x) * x)(a),
        "cv_t_jacrev_scalar": lambda a, p: jax.jacrev(lambda x: jnp.sin(x) * x)(a),
        "cv_t_jacfwd_pair": lambda a, p: jax.jacfwd(lambda t: (t[0] * t[1], jnp.sin(t[1])))((a, a * 2.0)),
        "cv_t_jacrev_pair": lambda a, p: jax.jacrev(lambda t: (t[0] * t[1], jnp.sin(t[1])))((a, a * 2.0)),
        "cv_t_grad_holo": lambda a, p: jax.grad(f_holo, holomorphic=True)(a),
        "cv_t_vg_holo": lambda a, p: jax.value_and_grad(f_holo, holomorphic=True)(a),
    }


def _transform_cases():
    r = {"a": {"kind": "real", "shape": [3]}}
    c = {"a": {"kind": "complex", "shape": [3]}}
    out = [(op, r, {}) for op in "cv_t_grad_aux cv_t_vg_aux cv_t_jvp_aux cv_t_vjp_aux cv_t_linearize_aux cv_t_jacfwd_aux cv_t_jacrev_aux cv_t_hessian_aux".split()]
    out += [("cv_t_grad_holo", c, {}), ("cv_t_vg_holo", c, {})]
    out += [(op, r, {}) for op in "cv_t_jacfwd cv_t_jacrev cv_t_hessian cv_t_jacfwd_pair cv_t_jacrev_pair".split()]
    m = {"a": {"kind": "real", "shape": [2, 2]}}
    out += [("cv_t_jacfwd", m, {}), ("cv_t_jacrev", m, {}), ("cv_t_hessian", m, {})]
    s0 = {"a": {"kind": "real", "shape": []}}
    out += [("cv_t_jacfwd_scalar", s0, {}), ("cv_t_jacrev_scalar", s0, {})]
    return out


def _sdn(update_window=(), inserted=(0,), to_operand=(0,)):
    from jax.lax import ScatterDimensionNumbers as S

    return S(update_window_dims=tuple(update_window), inserted_window_dims=tuple(inserted), scatter_dims_to_operand_dims=tuple(to_operand))


def _gdn():
    from jax.lax import GatherDimensionNumbers as G

    return G(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))


def _lax_ops():
    from jax import lax

    def scatter(name):
        fn = getattr(lax, name)
        idx = lambda p: jnp.asarray(p["idx"]).reshape(-1, 1)
        return lambda a, p: fn(a, idx(p), a[: len(p["idx"])] * 2.0, _sdn(), indices_are_sorted=p.get("sorted", False), unique_indices=p.get("unique", False))

    ops = {"cv_l_" + n: scatter(n) for n in "scatter scatter_add scatter_sub scatter_mul scatter_min scatter_max".split()}
    ops["cv_l_gather"] = lambda a, p: lax.gather(a, jnp.asarray(p["idx"]).reshape(-1, 1), _gdn(), (1,), mode=p["mode"], indices_are_sorted=p.get("sorted", False), unique_indices=p.get("unique", False))
    ops["cv_l_dynamic_slice"] = lambda a, p: lax.dynamic_slice(a, (jnp.argmax(a) - 1,), (2,), allow_negative_indices=p["allow"])
    ops["cv_l_dynamic_slice_in_dim"] = lambda a, p: lax.dynamic_slice_in_dim(a, jnp.argmax(a) - 1, 2, 0, allow_negative_indices=p["allow"])
    ops["cv_l_dynamic_update_slice"] = lambda a, p: lax.dynamic_update_slice(a, a[:2], (jnp.argmax(a) - 1,), allow_negative_indices=p["allow"])
    ops["cv_l_dynamic_update_slice_in_dim"] = lambda a, p: lax.dynamic_update_slice_in_dim(a, a[:2], jnp.argmax(a) - 1, 0, allow_negative_indices=p["allow"])
    ops["cv_l_fori_unroll"] = lambda a, p: lax.fori_loop(0, p["n"], lambda i, x: x * 2.0, a, unroll=p["unroll"])
    ops["cv_l_collapse_from"] = lambda a, p: lax.collapse(a, p["start"])
    return ops


def _lax_cases():
    r = lambda shape: {"a": {"kind": "real", "shape": shape}}
    out = []
    for n in "scatter scatter_add scatter_sub scatter_mul scatter_min scatter_max".split():
        for srt, unique in [(True, False), (False, True), (True, True)]:
            out.append(("cv_l_" + n, r([5]), {"idx": [0, 2], "sorted": srt, "unique": unique}))
    for mode in ["promise_in_bounds", "clip", "fill"]:
        for srt, unique in [(True, False), (False, True), (True, True)]:
            out.append(("cv_l_gather", r([5]), {"idx": [0, 2], "mode": mode, "sorted": srt, "unique": unique}))
    for allow in [True, False]:
        for op in "dynamic_slice dynamic_slice_in_dim dynamic_update_slice dynamic_update_slice_in_dim".split():
            out.append(("cv_l_" + op, r([5]), {"allow": allow}))
    out.append(("cv_l_fori_unroll", r([3]), {"n": 4, "unroll": 2}))
    out.append(("cv_l_collapse_from", r([2, 3, 4]), {"start": 1}))
    out.append(("cv_l_collapse_from", r([2, 3, 4]), {"start": -2}))
    return out


def _flat(fn):
    def call(*args, **kw):
        leaves = jax.tree_util.tree_leaves(fn(*args, **kw))
        return leaves[0] if len(leaves) == 1 else tuple(leaves)

    return call


OPS.update({k: _flat(v) for k, v in _transform_ops().items()})
OPS.update(_lax_ops())
_prior_all_cases = all_cases


def all_cases():
    return _prior_all_cases() + _transform_cases() + _lax_cases()


def _misc_ops():
    from jax import nn, ops as jops
    from jax.scipy import stats as jst

    seg = lambda name: lambda a, p: getattr(jops, name)(a, jnp.asarray(p["ids"]), num_segments=p.get("n"), indices_are_sorted=p.get("sorted", False), unique_indices=p.get("unique", False), bucket_size=p.get("bucket"), mode=p.get("mode"))
    out = {"cv_s_" + n: seg("segment_" + n) for n in "sum prod max min".split()}
    out["cv_sem"] = lambda a, p: jst.sem(a, axis=p.get("axis", 0), ddof=p.get("ddof", 1), nan_policy=p.get("nan_policy", "propagate"), keepdims=p.get("keepdims", False))
    out["cv_take_along"] = lambda a, p: jnp.take_along_axis(a, jnp.argsort(a, axis=0) - p["shift"], axis=0, mode=p.get("mode"), wrap_negative_indices=p["wrap"])
    out["cv_tree_map_leaf"] = lambda a, p: tuple(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: (x[0] + x[1]) if isinstance(x, tuple) else x * 2.0, {"p": (a, a * 3.0), "q": a}, is_leaf=lambda x: isinstance(x, tuple))))
    out["cv_x_maxwell"] = lambda a, p: jr.maxwell(_k(p), _sh(p), dtype=p["dtype"])
    out["cv_x_double_sided_maxwell"] = lambda a, p: jr.double_sided_maxwell(_k(p), 0.5, 2.0, _sh(p), dtype=p["dtype"])
    out["cv_x_ball"] = lambda a, p: jr.ball(_k(p), p["d"], shape=_sh(p), dtype=p["dtype"])
    out["cv_x_orthogonal"] = lambda a, p: jr.orthogonal(_k(p), p["n"], shape=_sh(p), dtype=p["dtype"])
    init = nn.initializers
    named = {
        "normal": lambda d: init.normal(dtype=d),
        "uniform": lambda d: init.uniform(dtype=d),
        "truncated_normal": lambda d: init.truncated_normal(dtype=d),
        "glorot_normal": lambda d: init.glorot_normal(dtype=d),
        "glorot_uniform": lambda d: init.glorot_uniform(dtype=d),
        "he_normal": lambda d: init.he_normal(dtype=d),
        "he_uniform": lambda d: init.he_uniform(dtype=d),
        "lecun_normal": lambda d: init.lecun_normal(dtype=d),
        "lecun_uniform": lambda d: init.lecun_uniform(dtype=d),
        "orthogonal": lambda d: init.orthogonal(dtype=d),
        "delta_orthogonal": lambda d: init.delta_orthogonal(dtype=d),
    }
    for name, make in named.items():
        out["cv_i_" + name] = lambda a, p, _m=make: _m(p["dtype"])(_k(p), _sh(p))
    return out


def _misc_cases():
    r = lambda shape: {"a": {"kind": "real", "shape": shape}}
    one = r([1])
    out = []
    ids = [0, 2, 1, 2, 0, 1]
    for name in "sum prod max min".split():
        op = "cv_s_" + name
        out.append((op, r([6]), {"ids": ids, "n": 3}))
        out.append((op, r([6]), {"ids": ids}))
        out.append((op, r([6, 2]), {"ids": ids, "n": 3}))
        out.append((op, r([6]), {"ids": sorted(ids), "n": 3, "sorted": True}))
        out.append((op, r([3]), {"ids": [0, 1, 2], "n": 3, "unique": True}))
        out.append((op, r([3]), {"ids": [0, 1, 2], "n": 3, "unique": True, "sorted": True}))
        out.append((op, r([6]), {"ids": ids, "n": 2, "mode": "clip"}))
        out.append((op, r([6]), {"ids": ids, "n": 3, "mode": "promise_in_bounds"}))
        out.append((op, r([6]), {"ids": ids, "n": 2, "mode": "drop"}))
        out.append((op, r([6]), {"ids": ids, "n": 3, "bucket": 2}))
        out.append((op, r([6, 2]), {"ids": ids, "n": 3, "bucket": 4}))
    for params in [{}, {"keepdims": True}, {"nan_policy": "omit"}, {"nan_policy": "omit", "keepdims": True}, {"ddof": 0, "nan_policy": "omit"}, {"axis": 1, "keepdims": True}]:
        out.append(("cv_sem", r([3, 4]), params))
    for wrap, shift in [(True, 0), (False, 0), (True, 1), (False, 1)]:
        out.append(("cv_take_along", r([4, 2]), {"wrap": wrap, "shift": shift}))
    out.append(("cv_tree_map_leaf", r([3]), {}))
    for dt in ["float16", "float32"]:
        out.append(("cv_x_maxwell", one, {"seed": 41, "shape": [3], "dtype": dt}))
        out.append(("cv_x_double_sided_maxwell", one, {"seed": 42, "shape": [3], "dtype": dt}))
        out.append(("cv_x_ball", one, {"seed": 43, "shape": [2], "d": 3, "dtype": dt}))
        for name in "normal uniform truncated_normal glorot_normal glorot_uniform he_normal he_uniform lecun_normal lecun_uniform".split():
            out.append(("cv_i_" + name, one, {"seed": 44, "shape": [3, 4], "dtype": dt}))
    out.append(("cv_x_orthogonal", one, {"seed": 45, "n": 3, "shape": [], "dtype": "float32"}))
    out.append(("cv_i_orthogonal", one, {"seed": 46, "shape": [3, 3], "dtype": "float32"}))
    out.append(("cv_i_delta_orthogonal", one, {"seed": 47, "shape": [3, 3, 3], "dtype": "float32"}))
    return out


OPS.update(_misc_ops())
_prior_all_cases_2 = all_cases


def all_cases():
    return _prior_all_cases_2() + _misc_cases()
