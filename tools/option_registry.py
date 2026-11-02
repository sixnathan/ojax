import importlib
import inspect
import math
import zlib

import jax
import jax.numpy as jnp
import numpy as np

MODULES = {
    "jnp": "jax.numpy",
    "linalg": "jax.numpy.linalg",
    "fft": "jax.numpy.fft",
    "lax": "jax.lax",
    "lax_linalg": "jax.lax.linalg",
    "nn": "jax.nn",
    "random": "jax.random",
    "ops": "jax.ops",
    "image": "jax.image",
    "sp_fft": "jax.scipy.fft",
    "sp_linalg": "jax.scipy.linalg",
    "sp_signal": "jax.scipy.signal",
    "sp_special": "jax.scipy.special",
    "sp_ndimage": "jax.scipy.ndimage",
    "sp_cluster": "jax.scipy.cluster.vq",
    "sp_integrate": "jax.scipy.integrate",
    "sp_stats_norm": "jax.scipy.stats.norm",
    "sp_stats_t": "jax.scipy.stats.t",
    "sp_stats_gamma": "jax.scipy.stats.gamma",
    "sp_stats_beta": "jax.scipy.stats.beta",
    "sp_stats_expon": "jax.scipy.stats.expon",
    "sp_stats_cauchy": "jax.scipy.stats.cauchy",
    "sp_stats_laplace": "jax.scipy.stats.laplace",
    "sp_stats_logistic": "jax.scipy.stats.logistic",
    "sp_stats_uniform": "jax.scipy.stats.uniform",
}

NEVER_SAMPLED = {
    "device", "out_sharding", "sharding", "copy", "overwrite_a", "overwrite_b", "overwrite_c", "overwrite_input",
    "overwrite_data", "overwrite_x", "check_finite", "debug", "lwork", "use_magma", "dynamic_shape", "allow_pickle",
    "donate", "may_alias", "src", "backend", "compiler_options", "turbo", "out", "subok", "like", "casting",
    "use_fortran_order", "fft_output", "workers", "plan", "x_is_sorted", "algorithm",
}

SKIP_ANNOTATION = ("Callable", "Mapping", "Sharding", "PRNGSpecDesc", "Sequence[Array")
SKIP_NAMES = {
    "fun", "func", "function", "f", "fn", "body_fun", "cond_fun", "kernel", "impl", "callback", "matvec", "solve",
    "tangent_solve", "pyfunc", "computation", "select_prng", "true_fun", "false_fun", "body", "jvp", "vjp", "cond",
    "branches", "index", "true_operand",
}
ARRAY_LIST = {"arrays", "tup", "operands", "xi", "seq"}

KIND_ORDER = ("real", "int", "bool", "complex")

SQUARE_MODULES = {"linalg", "lax_linalg", "sp_linalg"}

FUNCTION_SAMPLES = {
    ("jnp", "quantile", "method"): ["linear", "lower", "higher", "midpoint", "nearest", "inverted_cdf", "bogus"],
    ("jnp", "percentile", "method"): ["linear", "lower", "higher", "midpoint", "nearest", "inverted_cdf", "bogus"],
    ("jnp", "nanquantile", "method"): ["linear", "lower", "higher", "midpoint", "nearest", "bogus"],
    ("jnp", "nanpercentile", "method"): ["linear", "lower", "higher", "midpoint", "nearest", "bogus"],
    ("jnp", "searchsorted", "method"): ["scan", "scan_unrolled", "sort", "compare_all", "bogus"],
    ("jnp", "pad", "mode"): ["constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty", "bogus"],
    ("jnp", "take", "mode"): ["fill", "clip", "wrap", "promise_in_bounds", "bogus"],
    ("jnp", "take_along_axis", "mode"): ["fill", "clip", "wrap", "promise_in_bounds", "bogus"],
    ("jnp", "choose", "mode"): ["raise", "wrap", "clip", "bogus"],
    ("jnp", "ravel_multi_index", "mode"): ["raise", "wrap", "clip", "bogus"],
    ("jnp", "convolve", "mode"): ["full", "same", "valid", "bogus"],
    ("jnp", "correlate", "mode"): ["full", "same", "valid", "bogus"],
    ("jnp", "gradient", "edge_order"): [1, 2],
    ("jnp", "clip", "min"): [None, -0.5, 0.5],
    ("jnp", "clip", "max"): [None, 0.5, 1.5],
    ("jnp", "interp", "left"): [None, 0.5],
    ("jnp", "interp", "right"): [None, 0.5],
    ("jnp", "interp", "period"): [None, 2.0],
    ("nn", "squareplus", "b"): [4, 0.5],
    ("random", "lognormal", "sigma"): [1.0, 0.5],
    ("sp_linalg", "dft", "scale"): [None, "sqrtn", "n", "bogus"],
    ("sp_signal", "welch", "window"): ["hann"],
    ("sp_signal", "stft", "window"): ["hann"],
    ("sp_signal", "csd", "window"): ["hann"],
    ("sp_signal", "istft", "window"): ["hann"],
    ("linalg", "qr", "mode"): ["reduced", "complete", "r", "raw", "bogus"],
    ("sp_linalg", "qr", "mode"): ["full", "r", "economic", "raw", "bogus"],
    ("sp_linalg", "schur", "output"): ["real", "complex", "bogus"],
    ("sp_linalg", "polar", "side"): ["right", "left", "bogus"],
    ("sp_linalg", "polar", "method"): ["qdwh", "svd", "bogus"],
    ("sp_signal", "convolve", "method"): ["auto", "direct", "fft", "bogus"],
    ("sp_signal", "correlate", "method"): ["auto", "direct", "fft", "bogus"],
    ("sp_ndimage", "map_coordinates", "mode"): ["constant", "nearest", "wrap", "mirror", "reflect", "bogus"],
    ("image", "resize", "method"): ["nearest", "linear", "cubic", "lanczos3", "lanczos5", "bogus"],
    ("nn", "dot_product_attention", "implementation"): ["xla"],
}

BY_NAME = {
    "keepdims": [True, False],
    "precision": [None, "default", "high", "highest", ("high", "default"), "bogus"],
    "preferred_element_type": [None, "float32", "float64", "int32", "complex64"],
    "out_dtype": ["float32", "float64", "int32"],
    "implementation": [None, "xla", "bogus"],
    "lapack_driver": ["gesdd", "gesvd"],
    "max_iterations": [None, 5],
    "subset_by_index": [None, (0, 1), (0, 3), (0, 2)],
    "sort_eig_vals": [True, False],
    "keepdim": [True, False],
    "dtype": ["float32", "float64", "int32", "int64", "uint8", "int16", "complex64", "bool"],
    "order": ["C", "F", "A", "K", "bogus"],
    "side": ["left", "right", "bogus"],
    "norm": [None, "backward", "ortho", "forward", "bogus"],
    "ddof": [0, 1, 2, 7],
    "correction": [0, 1, 0.5],
    "ord": [None, 1, 2, -1, -2, 0, 3, "inf", "-inf", "fro", "nuc"],
    "where": ["$where"],
    "initial": [0.5, 2.0, -1],
    "weights": ["$like"],
    "shift": [0, 1, -2, 7],
    "k": [0, 1, -1, 2],
    "offset": [0, 1, -1, 5],
    "axis1": [0, 1, -1, 2],
    "axis2": [1, 0, -2, 2],
    "uplo": ["L", "U", "bogus"],
    "UPLO": ["L", "U", "bogus"],
    "trans": [0, 1, 2, "N", "T", "C", "bogus"],
    "rcond": [None, 0.5, 1e-3],
    "rtol": [None, 0.5],
    "tol": [None, 0.5],
    "nan": [0.0, 1.5],
    "posinf": [None, 7.5],
    "neginf": [None, -7.5],
    "fill_value": [None, 0.5, -1, 3],
    "size": [None, 1, 3, 8],
    "n": [None, 1, 2, 5],
    "num": [1, 2, 5, 0],
    "num_segments": [None, 1, 4],
    "prepend": ["$scalar"],
    "append": ["$scalar"],
    "descending": [True, False],
    "stable": [True, False],
    "assume_unique": [True, False],
    "invert": [True, False],
    "return_index": [True, False],
    "return_inverse": [True, False],
    "return_counts": [True, False],
    "return_indices": [True, False],
    "return_sign": [True, False],
    "return_scale": [True, False],
    "returned": [True, False],
    "retstep": [True, False],
    "endpoint": [True, False],
    "indexing": ["xy", "ij", "bogus"],
    "sparse": [True, False],
    "base": [2.0, 10.0],
    "s": [None, (2,), (2, 2), (3, 4)],
    "is_causal": [True, False],
    "local_window_size": [None, 1, (1, 0)],
    "return_residual": [True, False],
    "promote_integers": [True, False],
    "equal_nan": [True, False],
    "equal_nans": [True, False],
    "mean": ["$scalar"],
    "b": ["$like"],
    "scale": [None, 0.5, 2.0],
    "loc": [0, 0.5],
    "temperature": [0.5, 2.0],
    "negative_slope": [0.01, 0.2],
    "alpha": [0.5, 1.0],
    "beta": [0.5, 1.0],
    "minval": [0.0, -1.0],
    "maxval": [1.0, 3.0],
    "replace": [True, False],
    "independent": [True, False],
    "total_repeat_length": [None, 7],
    "bias": [True, False],
    "aweights": ["$scalar"],
    "fweights": ["$scalar"],
    "rowvar": [True, False],
    "stop_gradient": [True, False],
    "indices_are_sorted": [True, False],
    "unique_indices": [True, False],
    "strip_zeros": [True, False],
    "trim": ["f", "b", "fb", "bogus"],
    "full": [True, False],
    "compute_uv": [True, False],
    "full_matrices": [True, False],
    "hermitian": [True, False],
    "lower": [True, False],
    "unit_diagonal": [True, False],
    "symmetrize_input": [True, False],
    "sort_eigenvalues": [True, False],
    "pivoting": [True, False],
    "overwrite": [True, False],
    "left_side": [True, False],
    "conjugate_a": [True, False],
    "transpose_a": [True, False],
    "conjugate": [True, False],
    "conjugate_b": [True, False],
    "eigvals_only": [True, False],
    "left": [True, False],
    "right": [True, False],
    "exact": [True, False],
    "repetition": [True, False],
    "reverse": [True, False],
    "dimension": [0, 1, -1],
    "is_stable": [True, False],
    "num_keys": [1],
    "reduction_dimension": [-1, 0],
    "stop_dimension": [None, 2],
    "strides": [None, (1, 1), (2, 1)],
    "stride": [None, 1, 2],
    "shape": [None, (2,), (2, 3), ()],
    "rounding_method": [0, 1],
    "allow_negative_indices": [True, False],
    "transpose_kernel": [True, False],
    "use_consistent_padding": [True, False],
}

REQUIRED_VALUES = {
    "shape": [(2, 3), (3,), (), (1, 2, 2)],
    "newshape": [(3, 2), (-1,), (6,), (4, 2)],
    "new_sizes": [(2, 2)],
    "sizes": [(2,)],
    "new_shape": [(3, 2), (6,), (4, 2)],
    "start": [0.5, 1.0],
    "stop": [2.5, 5.0],
    "N": [0, 1, 3, 4],
    "M": [None, 1, 3],
    "repeats": [1, 2, 0],
    "axis": [0, 1, -1],
    "axes": [(0, 1), (1, 0)],
    "dims": [(0,), (1,), (0, 1)],
    "dimensions": [(0,), (1,)],
    "permutation": [(1, 0), (0, 1)],
    "source": [0, 1],
    "destination": [1, 0],
    "indices_or_sections": [2, 3, (1,)],
    "reps": [2, (2, 1)],
    "pad_width": [1, ((1, 0), (0, 2))],
    "pad_widths": [1],
    "deg": [1, 2],
    "num": [3, 0],
    "num_segments": [4],
    "new_dtype": ["float32", "int32", "complex64", "bool"],
    "out_dtype": ["float32", "int32"],
    "subscripts": ["ij->ji"],
    "n": [1, 2, 3],
    "k": [1, 2],
    "m": [1, 2],
    "x_shape": [(2,)],
}

REQUIRED_BY_FUNCTION = {
    ("lax", "collapse", "start_dimension"): [0, 1],
    ("jnp", "repeat", "repeats"): [1, 2, 0],
    ("jnp", "insert", "obj"): [0, 1],
    ("jnp", "delete", "obj"): [0, 1],
    ("jnp", "rollaxis", "axis"): [0, 1],
    ("jnp", "roll", "shift"): [1, -2, (1, 1)],
    ("jnp", "ravel_multi_index", "dims"): [(4, 4)],
    ("jnp", "unravel_index", "shape"): [(3, 3)],
    ("jnp", "polyfit", "deg"): [1, 2],
    ("jnp", "tril_indices", "n"): [2, 3],
    ("jnp", "triu_indices", "n"): [2, 3],
    ("jnp", "fill_diagonal", "val"): [1.5, 0],
    ("jnp", "put", "ind"): [],
    ("lax", "dynamic_index_in_dim", "index"): [0, 1, 5],
    ("lax", "dynamic_slice_in_dim", "start_index"): [0, 1],
    ("lax", "dynamic_slice_in_dim", "slice_size"): [1, 2],
    ("lax", "dynamic_slice", "start_indices"): [[0, 1], [1, 0]],
    ("lax", "dynamic_slice", "slice_sizes"): [[1, 2], (2, 1)],
    ("lax", "dynamic_update_index_in_dim", "index"): [0, 1],
    ("lax", "dynamic_update_slice_in_dim", "start_index"): [0, 1],
    ("lax", "dynamic_update_slice", "start_indices"): [[0, 0], [1, 0]],
    ("lax", "slice", "start_indices"): [(0, 0), (1, 1)],
    ("lax", "slice", "limit_indices"): [(1, 2), (2, 3)],
    ("lax", "slice_in_dim", "start_index"): [0, 1],
    ("lax", "slice_in_dim", "limit_index"): [1, 2],
    ("lax", "conv_general_dilated", "window_strides"): [(1,), (2,)],
    ("lax", "conv_general_dilated", "padding"): ["SAME", "VALID", [[1, 0]]],
    ("lax", "conv_transpose", "strides"): [(1,), (2,)],
    ("lax", "conv_transpose", "padding"): ["SAME", "VALID"],
    ("lax", "conv_with_general_padding", "window_strides"): [(1,), (2,)],
    ("lax", "conv_with_general_padding", "padding"): [[[1, 0]]],
    ("lax", "conv_with_general_padding", "lhs_dilation"): [None, (2,)],
    ("lax", "conv_with_general_padding", "rhs_dilation"): [None, (2,)],
    ("lax", "conv", "window_strides"): [(1,), (2,)],
    ("lax", "conv", "padding"): ["SAME", "VALID"],
    ("lax", "conv_general_dilated_patches", "filter_shape"): [(2,)],
    ("lax", "conv_general_dilated_patches", "window_strides"): [(1,)],
    ("lax", "conv_general_dilated_patches", "padding"): ["SAME", "VALID"],
    ("lax", "conv_general_dilated_local", "filter_shape"): [(2,)],
    ("lax", "conv_general_dilated_local", "window_strides"): [(1,)],
    ("lax", "conv_general_dilated_local", "padding"): ["SAME", "VALID"],
    ("lax", "full", "fill_value"): [1.5, 0],
    ("lax", "full_like", "fill_value"): [1.5, 0],
    ("jnp", "full", "fill_value"): [1.5, 0],
    ("jnp", "full_like", "fill_value"): [1.5, 0],
    ("jnp", "linspace", "start"): [0.5],
    ("jnp", "linspace", "stop"): [2.5],
    ("jnp", "logspace", "start"): [0.5],
    ("jnp", "logspace", "stop"): [2.5],
    ("jnp", "geomspace", "start"): [0.5],
    ("jnp", "geomspace", "stop"): [2.5],
    ("jnp", "arange", "start"): [0.5, 3, 5.5],
    ("random", "randint", "minval"): [0, -2],
    ("random", "randint", "maxval"): [3, 9],
    ("random", "ball", "d"): [2, 3],
    ("random", "orthogonal", "n"): [2, 3],
    ("random", "binomial", "n"): [3.0, 5.0],
    ("random", "binomial", "p"): [0.5, 0.25],
    ("random", "beta", "a"): [0.5, 2.0],
    ("random", "beta", "b"): [0.5, 2.0],
    ("random", "gamma", "a"): [0.5, 2.0],
    ("random", "loggamma", "a"): [0.5, 2.0],
    ("random", "chisquare", "df"): [1.5, 3.0],
    ("random", "t", "df"): [1.5, 3.0],
    ("random", "f", "dfnum"): [1.5, 3.0],
    ("random", "f", "dfden"): [1.5, 3.0],
    ("random", "generalized_normal", "p"): [1.5, 3.0],
    ("random", "geometric", "p"): [0.5, 0.25],
    ("random", "pareto", "b"): [0.5, 3.0],
    ("random", "poisson", "lam"): [0.5, 3.0],
    ("random", "rayleigh", "scale"): [0.5, 3.0],
    ("random", "wald", "mean"): [0.5, 3.0],
    ("random", "weibull_min", "scale"): [0.5, 3.0],
    ("random", "weibull_min", "concentration"): [0.5, 3.0],
    ("random", "double_sided_maxwell", "loc"): [0.5, 3.0],
    ("random", "double_sided_maxwell", "scale"): [0.5, 3.0],
    ("random", "truncated_normal", "lower"): [-0.5, -3.0],
    ("random", "truncated_normal", "upper"): [0.5, 3.0],
    ("random", "triangular", "left"): [-1.0],
    ("random", "triangular", "mode"): [0.0, 0.5],
    ("random", "triangular", "right"): [1.0],
    ("random", "choice", "a"): [3, 5],
    ("random", "key", "seed"): [0, 7],
}

FORCED_REQUIRED = {
    ("jnp", "clip"): ["arr"],
    ("jnp", "where"): ["x", "y"],
}

LIST_ARGS = {
    ("jnp", "ravel_multi_index", "multi_index"), ("jnp", "select", "condlist"), ("jnp", "select", "choicelist"),
    ("jnp", "lexsort", "keys"), ("jnp", "choose", "choices"),
}

BOOL_ARGS = {("jnp", "select", "condlist"), ("jnp", "where", "condition"), ("jnp", "where", "condlist")}

INT_ARGS = {("jnp", "ravel_multi_index", "multi_index"), ("jnp", "searchsorted", "sorter")}

VARIANT_NAMES = {
    ("lax", "mul"): "mul_with",
    ("jnp", "linspace"): "linspace_v",
    ("jnp", "logspace"): "logspace_v",
    ("jnp", "geomspace"): "geomspace_v",
    ("jnp", "where"): "where_",
    ("jnp", "delete"): "delete_array",
    ("random", "choice"): "choice_array",
    ("random", "permutation"): "permutation_array",
    ("random", "categorical"): "categorical_full",
    ("random", "multinomial"): "multinomial_full",
}

SKIPPED = {
    ("lax", "all_gather"), ("lax", "all_to_all"), ("lax", "psum_scatter"), ("lax", "pmax"), ("lax", "pmin"), ("lax", "pmean"), ("lax", "psum"),
    ("lax", "ragged_dot"), ("lax", "ragged_dot_general"), ("lax", "scaled_dot"), ("lax", "gather"), ("lax", "scatter"), ("lax", "scatter_add"),
    ("lax", "scatter_max"), ("lax", "scatter_min"), ("lax", "scatter_mul"), ("lax", "scatter_sub"), ("lax", "scatter_apply"), ("lax", "dce_sink"),
    ("lax", "conv_shape_tuple"), ("lax", "rng_bit_generator"), ("random", "wrap_key_data"), ("random", "multinomial"),
    ("jnp", "load"), ("jnp", "save"), ("jnp", "savez"), ("jnp", "frombuffer"), ("jnp", "fromstring"), ("jnp", "fromfile"), ("jnp", "fromiter"),
    ("jnp", "fromfunction"), ("jnp", "from_dlpack"), ("jnp", "printoptions"), ("jnp", "set_printoptions"), ("jnp", "get_printoptions"),
    ("jnp", "array_str"), ("jnp", "array_repr"), ("jnp", "dtype"), ("jnp", "printoptions"), ("jnp", "iterable"), ("jnp", "result_type"),
    ("jnp", "issubdtype"), ("jnp", "promote_types"), ("jnp", "can_cast"),
}

SHAPES_BY_FUNCTION = {
    ("lax", "conv_general_dilated"): {"lhs": (1, 1, 5), "rhs": (1, 1, 2)},
    ("lax", "conv"): {"lhs": (1, 1, 5), "rhs": (1, 1, 2)},
    ("lax", "conv_transpose"): {"lhs": (1, 1, 5), "rhs": (1, 1, 2)},
    ("lax", "conv_with_general_padding"): {"lhs": (1, 1, 5), "rhs": (1, 1, 2)},
    ("lax", "conv_general_dilated_patches"): {"lhs": (1, 1, 5)},
    ("lax", "conv_general_dilated_local"): {"lhs": (1, 1, 5), "rhs": (1, 2, 2)},
    ("lax", "sort_key_val"): {"keys": (5,), "values": (5,)},
    ("random", "multivariate_normal"): {"mean": (2,), "cov": (2, 2)},
    ("random", "categorical"): {"logits": (3,)},
    ("random", "dirichlet"): {"alpha": (3,)},
    ("random", "permutation"): {"x": (2, 3)},
    ("jnp", "take"): {"indices": (2,)},
    ("jnp", "repeat"): {},
    ("jnp", "take_along_axis"): {"indices": (2, 3)},
    ("sp_linalg", "lu_solve"): {},
    ("linalg", "vector_norm"): {"x": (2, 3)},
    ("linalg", "cross"): {"x1": (3,), "x2": (3,)},
    ("jnp", "cross"): {"a": (3,), "b": (3,)},
    ("jnp", "polyval"): {"p": (3,), "x": (2,)},
}

BOGUS_STRING = "bogus"


def digest(text):
    return zlib.adler32(text.encode("utf-8"))


def annotation_text(param):
    return "" if param.annotation is inspect.Parameter.empty else str(param.annotation)


def is_array_annotation(text):
    return "Array" in text


def lookup(module_short, name):
    module = importlib.import_module(MODULES[module_short])
    return getattr(module, name, None)


def public_functions(module_short):
    module = importlib.import_module(MODULES[module_short])
    names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    for name in sorted(set(names)):
        obj = getattr(module, name, None)
        if name.startswith("_") or inspect.ismodule(obj) or inspect.isclass(obj) or isinstance(obj, jnp.ufunc):
            continue
        if not callable(obj):
            continue
        if "Primitive" in type(obj).__name__:
            continue
        yield name, obj


def spec_of(module_short, name, obj):
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    required, options, array_options = [], {}, {}
    forced = FORCED_REQUIRED.get((module_short, name), [])
    for param in signature.parameters.values():
        if param.name in forced:
            required.append((param.name, "array"))
            continue
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            if param.kind == param.VAR_POSITIONAL and name in ("broadcast_arrays", "meshgrid", "ix_", "block"):
                required.append((param.name, "star"))
            continue
        text = annotation_text(param)
        if param.default is inspect.Parameter.empty:
            if param.name in SKIP_NAMES and not is_array_annotation(text):
                return None
            if "Callable" in text:
                return None
            if (module_short, name, param.name) in REQUIRED_BY_FUNCTION:
                required.append((param.name, "value"))
            elif param.name in ("key",) and module_short == "random":
                required.append((param.name, "key"))
            elif param.name in REQUIRED_VALUES and not is_array_annotation(text):
                required.append((param.name, "value"))
            elif param.name in ARRAY_LIST or (module_short, name, param.name) in LIST_ARGS:
                required.append((param.name, "list"))
            elif param.name in ("dtype",):
                required.append((param.name, "value"))
            elif text == "" or is_array_annotation(text) or param.name in ("x", "y", "a", "b", "operand", "lhs", "rhs"):
                required.append((param.name, "array"))
            else:
                return None
            continue
        if param.name in NEVER_SAMPLED or param.name.startswith("_"):
            continue
        if any(s in text for s in SKIP_ANNOTATION):
            continue
        values = FUNCTION_SAMPLES.get((module_short, name, param.name))
        if values is None:
            values = BY_NAME.get(param.name)
        if values is None:
            values = generic_values(param, text)
        if values is None:
            continue
        if param.default is not None:
            values = [v for v in values if v is not None]
        if not values:
            continue
        options[param.name] = values
    return {"required": required, "options": options, "module": module_short, "name": name}


def generic_values(param, text):
    default = param.default
    if isinstance(default, bool):
        return [True, False]
    if isinstance(default, int):
        return [default, default + 1, 0, -1]
    if isinstance(default, float):
        return [default, 0.5, 2.0]
    if isinstance(default, str):
        return [default, BOGUS_STRING]
    if param.name == "axis" or param.name == "axes":
        values = [0, 1, -1, 2]
        if default is None:
            values.append(None)
        if "tuple" in text or "Sequence" in text or "Axis" in text or param.name == "axes":
            values += [(0, 1), (1, 0), (0, 0)]
        return values
    if "Sequence[int]" in text and default is None:
        return [None, (1,), (2,)]
    if "int" in text and default is None:
        return [None, 1, 2]
    if "bool" in text and default is None:
        return [None, True, False]
    if "float" in text and default is None:
        return [None, 0.5]
    if is_array_annotation(text) and default is None:
        return [None, 0.5]
    return None


def all_specs():
    specs = []
    for short in MODULES:
        for name, obj in public_functions(short):
            if (short, name) in SKIPPED:
                continue
            spec = spec_of(short, name, obj)
            if spec is not None and spec["options"]:
                specs.append(spec)
    return specs


def sample_case(spec, rng, index):
    module_short = spec["module"]
    kind = KIND_ORDER[int(rng.choice(4, p=[0.7, 0.14, 0.08, 0.08]))] if index > 0 else "real"
    if spec["name"] in ("searchsorted", "digitize", "interp", "polyval", "convolve", "correlate", "trapezoid", "cross"):
        rank_one = True
    else:
        rank_one = module_short in ("sp_signal",) and "2d" not in spec["name"]
    inputs = {}
    params = {}
    positional = []
    override = SHAPES_BY_FUNCTION.get((module_short, spec["name"]), {})
    base_shape = (3, 3) if module_short in SQUARE_MODULES else ((4,) if rank_one else (2, 3))
    for position, (arg, role) in enumerate(spec["required"]):
        if role == "array":
            key = "arg" + str(position)
            if arg == "segment_ids":
                inputs[key] = {"shape": [2], "kind": "int"}
            elif arg in ("indices", "indices_or_sections", "condlist", "condition", "mask") or arg.startswith("bins"):
                inputs[key] = {"shape": list(override.get(arg, base_shape)), "kind": "int" if arg != "condition" and arg != "mask" else "bool"}
            else:
                inputs[key] = {"shape": list(override.get(arg, base_shape)), "kind": kind}
            positional.append({"$arr": key})
        elif role == "list":
            list_kind = "bool" if (module_short, spec["name"], arg) in BOOL_ARGS else "int" if (module_short, spec["name"], arg) in INT_ARGS else kind
            for k in ("arg" + str(position) + "_0", "arg" + str(position) + "_1"):
                inputs[k] = {"shape": list(base_shape), "kind": list_kind}
            positional.append({"$list": ["arg" + str(position) + "_0", "arg" + str(position) + "_1"]})
        elif role == "star":
            for k in ("arg" + str(position) + "_0", "arg" + str(position) + "_1"):
                inputs[k] = {"shape": list(base_shape), "kind": kind}
            positional.append({"$star": ["arg" + str(position) + "_0", "arg" + str(position) + "_1"]})
        elif role == "key":
            positional.append({"$key": int(rng.integers(0, 2**31 - 1))})
        else:
            values = REQUIRED_BY_FUNCTION.get((module_short, spec["name"], arg)) or REQUIRED_VALUES.get(arg) or [0]
            value = values[int(rng.integers(0, len(values)))]
            positional.append(jsonable(value))
    names = sorted(spec["options"])
    if index > 0 and names:
        count = 1 + int(rng.integers(0, min(3, len(names))))
        chosen = list(rng.choice(len(names), size=count, replace=False))
        for pick in chosen:
            option = names[int(pick)]
            values = spec["options"][option]
            value = values[int(rng.integers(0, len(values)))]
            if value == "$where":
                inputs["opt_" + option] = {"shape": list(base_shape), "kind": "bool"}
                params[option] = {"$arr": "opt_" + option}
            elif value == "$like":
                inputs["opt_" + option] = {"shape": list(base_shape), "kind": "positive"}
                params[option] = {"$arr": "opt_" + option}
            elif value == "$scalar":
                inputs["opt_" + option] = {"shape": [], "kind": "positive"}
                params[option] = {"$arr": "opt_" + option}
            else:
                params[option] = jsonable(value)
    return {"positional": positional, "options": params, "inputs": inputs}


def jsonable(value):
    if isinstance(value, tuple):
        return {"$tuple": [jsonable(v) for v in value]}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def from_json(value, arrays):
    if isinstance(value, dict):
        if "$arr" in value:
            return arrays[value["$arr"]]
        if "$list" in value:
            return [arrays[k] for k in value["$list"]]
        if "$tuple" in value:
            return tuple(from_json(v, arrays) for v in value["$tuple"])
        if "$key" in value:
            return jax.random.key(value["$key"])
    if isinstance(value, list):
        return [from_json(v, arrays) for v in value]
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return value


def call(module_short, name, positional, options, arrays):
    fn = lookup(module_short, name)
    args = []
    for item in positional:
        if isinstance(item, dict) and "$star" in item:
            args.extend(arrays[k] for k in item["$star"])
        else:
            args.append(from_json(item, arrays))
    return fn(*args, **{k: from_json(v, arrays) for k, v in options.items()})


def flatten_output(result):
    leaves = [jnp.asarray(leaf) if isinstance(leaf, (bool, int, float, complex)) else leaf for leaf in jax.tree_util.tree_leaves(result)]
    return [np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(jnp.result_type(leaf), jax.dtypes.prng_key) else leaf, order="C") for leaf in leaves]
