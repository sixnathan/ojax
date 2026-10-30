import jax
import jax.numpy as jnp
from jax import lax


def _fft_ops():
    f = jnp.fft
    return {
        "fft": lambda a, p: f.fft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "ifft": lambda a, p: f.ifft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "rfft": lambda a, p: f.rfft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "irfft": lambda a, p: f.irfft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "hfft": lambda a, p: f.hfft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "ihfft": lambda a, p: f.ihfft(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "fftn": lambda a, p: f.fftn(a, s=p.get("s"), axes=p.get("axes"), norm=p.get("norm")),
        "ifftn": lambda a, p: f.ifftn(a, s=p.get("s"), axes=p.get("axes"), norm=p.get("norm")),
        "rfftn": lambda a, p: f.rfftn(a, s=p.get("s"), axes=p.get("axes"), norm=p.get("norm")),
        "irfftn": lambda a, p: f.irfftn(a, s=p.get("s"), axes=p.get("axes"), norm=p.get("norm")),
        "fft2": lambda a, p: f.fft2(a, norm=p.get("norm")),
        "ifft2": lambda a, p: f.ifft2(a, norm=p.get("norm")),
        "rfft2": lambda a, p: f.rfft2(a, norm=p.get("norm")),
        "irfft2": lambda a, p: f.irfft2(a, norm=p.get("norm")),
        "fftshift": lambda a, p: f.fftshift(a, axes=p.get("axes")),
        "ifftshift": lambda a, p: f.ifftshift(a, axes=p.get("axes")),
    }


def _lax_ops():
    return {
        "lax_clamp": lambda a, p: lax.clamp(jnp.asarray(-0.5, a.dtype), a, jnp.asarray(0.5, a.dtype)),
        "lax_add_scalar": lambda a, p: lax.add(a, jnp.asarray(2.0, a.dtype)),
        "lax_expand_dims": lambda a, p: lax.expand_dims(a, p["dims"]),
        "lax_broadcast": lambda a, p: lax.broadcast(a, p["sizes"]),
        "lax_collapse": lambda a, p: lax.collapse(a, p["start"], p["stop"]),
        "lax_slice_in_dim": lambda a, p: lax.slice_in_dim(a, p["start"], p["stop"], stride=p.get("stride", 1), axis=p.get("axis", 0)),
        "lax_index_in_dim": lambda a, p: lax.index_in_dim(a, p["index"], axis=p.get("axis", 0), keepdims=p.get("keepdims", True)),
        "lax_dynamic_slice_in_dim": lambda a, p: lax.dynamic_slice_in_dim(a, p["start"], p["size"], axis=p.get("axis", 0)),
        "lax_dynamic_index_in_dim": lambda a, p: lax.dynamic_index_in_dim(a, p["index"], axis=p.get("axis", 0), keepdims=p.get("keepdims", True)),
        "lax_reduce_custom": lambda a, p: lax.reduce(a, jnp.asarray(0.0, a.dtype), lax.add, p["dims"]),
        "lax_reduce_window": lambda a, p: lax.reduce_window(a, jnp.asarray(0.0, a.dtype), lax.add, p["win"], p["str"], [tuple(x) for x in p["pad"]]),
        "lax_sort": lambda a, p: lax.sort(a, dimension=p.get("dimension", -1)),
        "lax_sort_key_val": lambda a, b, p: lax.sort_key_val(a, b),
        "lax_top_k": lambda a, p: lax.top_k(a, p["k"]),
        "lax_dot": lambda a, b, p: lax.dot(a, b),
        "lax_batch_matmul": lambda a, b, p: lax.batch_matmul(a, b),
        "lax_conv_same": lambda a, b, p: lax.conv(a, b, p["str"], "SAME"),
        "lax_conv_valid": lambda a, b, p: lax.conv(a, b, p["str"], "VALID"),
        "lax_conv_general": lambda a, b, p: lax.conv_general_dilated(a, b, p["str"], p["padding"], lhs_dilation=p.get("ld"), rhs_dilation=p.get("rd"), dimension_numbers=tuple(p["dn"])),
        "lax_conv_transpose": lambda a, b, p: lax.conv_transpose(a, b, p["str"], p["padding"], dimension_numbers=("NCH", "OIH", "NCH"), transpose_kernel=p.get("tk", False)),
        "lax_map": lambda a, p: lax.map(lambda x: x * x + 1.0, a),
        "lax_switch": lambda a, p: lax.switch(p["i"], [lambda x: x + 1.0, lambda x: x * 2.0, lambda x: -x], a),
        "lax_fori_loop": lambda a, p: lax.fori_loop(0, p["n"], lambda i, x: x * 2.0, a),
        "lax_associative_scan": lambda a, p: lax.associative_scan(lax.add, a, axis=p.get("axis", 0)),
        "lax_custom_root": lambda a, p: lax.custom_root(lambda x: x * x - a, jnp.ones_like(a), _newton, lambda g, y: y / g(jnp.ones_like(y))),
        "lax_reciprocal": lambda a, p: lax.reciprocal(a),
        "lax_full_like": lambda a, p: lax.full_like(a, p["v"]),
        "lax_stop_gradient": lambda a, p: jax.grad(lambda x: jnp.sum(lax.stop_gradient(x) * x))(a),
    }


def _sc(v):
    return jnp.sort_complex(v)


def _numpy_ops():
    n = jnp
    return {
        "np_nonzero": lambda a, p: n.nonzero(_binarize(a)),
        "np_nonzero_size": lambda a, p: n.nonzero(_binarize(a), size=p["size"], fill_value=p.get("fill", 0)),
        "np_flatnonzero": lambda a, p: n.flatnonzero(_binarize(a)),
        "np_argwhere": lambda a, p: n.argwhere(_binarize(a)),
        "np_unique_values": lambda a, p: n.unique_values(_rounded(a)),
        "np_unique_counts": lambda a, p: tuple(n.unique_counts(_rounded(a))),
        "np_unique_inverse": lambda a, p: tuple(n.unique_inverse(_rounded(a))),
        "np_unique_all": lambda a, p: tuple(n.unique_all(_rounded(a))),
        "np_unique_size": lambda a, p: n.unique(_rounded(a), size=p["size"], fill_value=p["fill"]),
        "np_unique_axis": lambda a, p: n.unique(_rounded(a), axis=p["axis"]),
        "np_union1d": lambda a, b, p: n.union1d(_rounded(a), _rounded(b)),
        "np_intersect1d": lambda a, b, p: n.intersect1d(_rounded(a), _rounded(b)),
        "np_setdiff1d": lambda a, b, p: n.setdiff1d(_rounded(a), _rounded(b)),
        "np_setxor1d": lambda a, b, p: n.setxor1d(_rounded(a), _rounded(b)),
        "np_bincount": lambda a, p: n.bincount(_nonneg(a)),
        "np_bincount_length": lambda a, b, p: n.bincount(_nonneg(a), weights=b, length=p["length"]),
        "np_histogram": lambda a, p: tuple(n.histogram(a, bins=p["bins"], density=p.get("density", False))),
        "np_histogram_range": lambda a, p: tuple(n.histogram(a, bins=p["bins"], range=tuple(p["range"]))),
        "np_histogram_bin_edges": lambda a, p: n.histogram_bin_edges(a, bins=p["bins"]),
        "np_histogram2d": lambda a, b, p: tuple(n.histogram2d(a, b, bins=p["bins"], density=p.get("density", False))),
        "np_histogramdd": lambda a, p: (lambda r: (r[0], *r[1]))(n.histogramdd(a, bins=p["bins"])),
        "np_interp": lambda a, b, c, p: n.interp(a, jnp.sort(b), c),
        "np_interp_fill": lambda a, b, c, p: n.interp(a, jnp.sort(b), c, left=p["left"], right=p["right"]),
        "np_interp_period": lambda a, b, c, p: n.interp(a, jnp.sort(b), c, period=p["period"]),
        "np_gradient": lambda a, p: n.gradient(a, p.get("spacing", 1.0)),
        "np_gradient_axis": lambda a, p: n.gradient(a, axis=tuple(p["axis"])),
        "np_choose": lambda a, b, p: n.choose(_nonneg(a) % 2, [b, -b], mode="clip"),
        "np_compress": lambda a, p: n.compress(jnp.asarray(p["cond"], bool), a, axis=p["axis"]),
        "np_extract": lambda a, p: n.extract(a > 0, a),
        "np_delete": lambda a, p: n.delete(a, jnp.asarray(p["obj"]), axis=p.get("axis")),
        "np_insert": lambda a, p: n.insert(a, p["pos"], jnp.asarray(p["val"], a.dtype), axis=p.get("axis")),
        "np_place": lambda a, p: n.place(a, a > 0, jnp.asarray(p["val"], a.dtype), inplace=False),
        "np_piecewise": lambda a, p: n.piecewise(a, [a < 0, a >= 0], [lambda x: -x, lambda x: x * 2.0]),
        "np_fill_diagonal": lambda a, p: n.fill_diagonal(a, jnp.asarray(p["val"], a.dtype), inplace=False),
        "np_tril_indices": lambda a, p: n.tril_indices(p["n"], k=p.get("k", 0), m=p.get("m")),
        "np_triu_indices": lambda a, p: n.triu_indices(p["n"], k=p.get("k", 0), m=p.get("m")),
        "np_tril_indices_from": lambda a, p: n.tril_indices_from(a, k=p.get("k", 0)),
        "np_triu_indices_from": lambda a, p: n.triu_indices_from(a, k=p.get("k", 0)),
        "np_ravel_multi_index": lambda a, p: n.ravel_multi_index(tuple(n.asarray(c) for c in p["coords"]), tuple(p["dims"]), mode=p["mode"]),
        "np_trim_zeros": lambda a, p: n.trim_zeros(jnp.concatenate([jnp.zeros(2, a.dtype), a, jnp.zeros(1, a.dtype)]), trim=p["trim"]),
        "np_frexp": lambda a, p: tuple(n.frexp(a)),
        "np_ldexp": lambda a, p: n.ldexp(a, jnp.asarray(p["e"], jnp.int32)),
        "np_apply_along_axis": lambda a, p: n.apply_along_axis(jnp.sum, p["axis"], a),
        "np_apply_along_axis_keep": lambda a, p: n.apply_along_axis(lambda v: v * 2.0, p["axis"], a),
        "np_apply_over_axes": lambda a, p: n.apply_over_axes(jnp.sum, a, p["axes"]),
        "np_polydiv": lambda a, b, p: tuple(n.polydiv(a, b)),
        "np_polyfit": lambda a, b, p: n.polyfit(a, b, p["deg"]),
        "np_roots": lambda a, p: _sc(n.roots(a)),
        "np_block": lambda a, b, p: n.block([[a, b], [b, a]]),
    }


def _binarize(a):
    return a > 0


def _rounded(a):
    return jnp.round(a * 2.0) / 2.0


def _nonneg(a):
    return jnp.abs(a).astype(jnp.int32)


def _newton(f, x0):
    x = x0
    for _ in range(40):
        fx = f(x)
        slope = jax.jvp(f, (x,), (jnp.ones_like(x),))[1]
        x = x - fx / slope
    return x


OPS = {}
OPS.update(_fft_ops())
OPS.update(_lax_ops())
OPS.update(_numpy_ops())


def _complex_input(shape):
    return {"a": {"kind": "complex", "shape": shape}}


def _real_input(shape):
    return {"a": {"kind": "real", "shape": shape}}


def _fft_cases():
    out = []
    for op in ["fft", "ifft"]:
        for shape, p in [
            ([8], {}),
            ([7], {}),
            ([2, 6], {"axis": 0}),
            ([8], {"n": 5}),
            ([5], {"n": 8}),
            ([6], {"norm": "ortho"}),
            ([6], {"norm": "forward"}),
        ]:
            out.append((op, _complex_input(shape), p))
    for shape, p in [
        ([8], {}),
        ([9], {}),
        ([3, 8], {}),
        ([8], {"n": 5}),
        ([8], {"norm": "ortho"}),
    ]:
        out.append(("rfft", _real_input(shape), p))
        out.append(("ihfft", _real_input(shape), p))
    for shape, p in [
        ([5], {}),
        ([5], {"n": 9}),
        ([3, 5], {}),
        ([5], {"norm": "forward"}),
    ]:
        out.append(("irfft", _complex_input(shape), p))
        out.append(("hfft", _complex_input(shape), p))
    for op in ["fftn", "ifftn"]:
        out.append((op, _complex_input([3, 4]), {}))
        out.append((op, _complex_input([2, 3, 4]), {"axes": [0, 2]}))
        out.append((op, _complex_input([3, 4]), {"s": [5, 2]}))
    out.append(("rfftn", _real_input([3, 4]), {}))
    out.append(("rfftn", _real_input([2, 3, 4]), {"axes": [0, 1]}))
    out.append(("irfftn", _complex_input([3, 3]), {}))
    out.append(("irfftn", _complex_input([3, 3]), {"s": [3, 5]}))
    for op in ["fft2", "ifft2"]:
        out.append((op, _complex_input([3, 4]), {}))
    out.append(("rfft2", _real_input([3, 4]), {}))
    out.append(("irfft2", _complex_input([3, 3]), {}))
    for op in ["fftshift", "ifftshift"]:
        out.append((op, _real_input([5]), {}))
        out.append((op, _real_input([2, 4]), {"axes": [1]}))
    return out


def _two(a_kind, a_shape, b_kind, b_shape):
    return {"a": {"kind": a_kind, "shape": a_shape}, "b": {"kind": b_kind, "shape": b_shape}}


def _lax_cases():
    r = _real_input
    pos = lambda shape: {"a": {"kind": "positive", "shape": shape}}
    return [
        ("lax_clamp", r([5]), {}),
        ("lax_add_scalar", r([2, 3]), {}),
        ("lax_expand_dims", r([2, 3]), {"dims": [0, 2]}),
        ("lax_broadcast", r([2, 3]), {"sizes": [4]}),
        ("lax_collapse", r([2, 3, 4]), {"start": 1, "stop": 3}),
        ("lax_slice_in_dim", r([6, 2]), {"start": 1, "stop": 5, "stride": 2}),
        ("lax_slice_in_dim", r([2, 6]), {"start": 1, "stop": -1, "axis": 1}),
        ("lax_index_in_dim", r([4, 2]), {"index": 2, "keepdims": False}),
        ("lax_index_in_dim", r([2, 4]), {"index": -1, "axis": 1}),
        ("lax_dynamic_slice_in_dim", r([6, 2]), {"start": 2, "size": 3}),
        ("lax_dynamic_index_in_dim", r([4, 2]), {"index": 1, "keepdims": False}),
        ("lax_reduce_custom", r([2, 3]), {"dims": [1]}),
        ("lax_reduce_window", r([2, 6]), {"win": [1, 2], "str": [1, 2], "pad": [[0, 0], [0, 0]]}),
        ("lax_sort", r([5]), {}),
        ("lax_sort", r([3, 4]), {"dimension": 0}),
        ("lax_sort_key_val", _two("real", [5], "real", [5]), {}),
        ("lax_top_k", r([6]), {"k": 2}),
        ("lax_dot", _two("real", [2, 3], "real", [3, 2]), {}),
        ("lax_dot", _two("real", [3], "real", [3]), {}),
        ("lax_batch_matmul", _two("real", [2, 2, 3], "real", [2, 3, 2]), {}),
        ("lax_conv_same", _two("real", [1, 1, 7], "real", [1, 1, 3]), {"str": [2]}),
        ("lax_conv_valid", _two("real", [1, 2, 4, 4], "real", [3, 2, 2, 2]), {"str": [1, 1]}),
        ("lax_conv_general", _two("real", [2, 5, 1], "real", [3, 1, 1]), {"str": [1], "padding": "SAME", "dn": ["NHC", "HIO", "NHC"]}),
        ("lax_conv_general", _two("real", [1, 1, 5], "real", [1, 1, 3]), {"str": [1], "padding": "VALID", "rd": [2], "dn": ["NCH", "OIH", "NCH"]}),
        ("lax_conv_transpose", _two("real", [1, 1, 4], "real", [1, 1, 3]), {"str": [2], "padding": "SAME"}),
        ("lax_conv_transpose", _two("real", [1, 1, 4], "real", [1, 1, 3]), {"str": [2], "padding": "VALID", "tk": True}),
        ("lax_map", r([4]), {}),
        ("lax_switch", r([3]), {"i": 0}),
        ("lax_switch", r([3]), {"i": 1}),
        ("lax_switch", r([3]), {"i": 7}),
        ("lax_fori_loop", r([2]), {"n": 3}),
        ("lax_associative_scan", r([5]), {}),
        ("lax_custom_root", pos([3]), {}),
        ("lax_reciprocal", pos([3]), {}),
        ("lax_full_like", r([2, 2]), {"v": 3.5}),
        ("lax_stop_gradient", r([3]), {}),
    ]


def _three(sa, sb, sc):
    return {
        "a": {"kind": "real", "shape": sa},
        "b": {"kind": "real", "shape": sb},
        "c": {"kind": "real", "shape": sc},
    }


def _numpy_cases():
    r = _real_input
    rr = lambda sa, sb: _two("real", sa, "real", sb)
    return [
        ("np_nonzero", r([5]), {}),
        ("np_nonzero", r([3, 4]), {}),
        ("np_nonzero_size", r([3, 4]), {"size": 3}),
        ("np_nonzero_size", r([3, 2]), {"size": 8, "fill": 1}),
        ("np_flatnonzero", r([3, 2]), {}),
        ("np_argwhere", r([3, 2]), {}),
        ("np_unique_values", r([9]), {}),
        ("np_unique_counts", r([9]), {}),
        ("np_unique_inverse", r([2, 5]), {}),
        ("np_unique_all", r([9]), {}),
        ("np_unique_size", r([9]), {"size": 12, "fill": -9.0}),
        ("np_unique_size", r([9]), {"size": 2, "fill": 0.0}),
        ("np_unique_axis", r([4, 2]), {"axis": 0}),
        ("np_union1d", rr([6], [5]), {}),
        ("np_intersect1d", rr([12], [12]), {}),
        ("np_setdiff1d", rr([8], [8]), {}),
        ("np_setxor1d", rr([8], [8]), {}),
        ("np_bincount", r([8]), {}),
        ("np_bincount_length", rr([8], [8]), {"length": 3}),
        ("np_histogram", r([20]), {"bins": 5}),
        ("np_histogram", r([20]), {"bins": 4, "density": True}),
        ("np_histogram_range", r([20]), {"bins": 3, "range": [-1.0, 1.0]}),
        ("np_histogram_bin_edges", r([20]), {"bins": 6}),
        ("np_histogram2d", rr([15], [15]), {"bins": [3, 2]}),
        ("np_histogram2d", rr([15], [15]), {"bins": [2, 2], "density": True}),
        ("np_histogramdd", r([15, 3]), {"bins": [2, 3, 2]}),
        ("np_interp", _three([6], [5], [5]), {}),
        ("np_interp_fill", _three([6], [5], [5]), {"left": -7.0, "right": 9.0}),
        ("np_interp_period", _three([6], [5], [5]), {"period": 2.0}),
        ("np_gradient", r([3, 4]), {}),
        ("np_gradient", r([5]), {"spacing": 0.5}),
        ("np_gradient_axis", r([3, 4]), {"axis": [1]}),
        ("np_choose", rr([5], [5]), {}),
        ("np_compress", r([3, 2]), {"cond": [True, False, True], "axis": 0}),
        ("np_extract", r([3, 2]), {}),
        ("np_delete", r([5]), {"obj": [1, 3]}),
        ("np_delete", r([3, 2]), {"obj": [0], "axis": 0}),
        ("np_insert", r([4]), {"pos": 1, "val": 9.0}),
        ("np_insert", r([3, 2]), {"pos": 1, "val": 9.0, "axis": 1}),
        ("np_place", r([6]), {"val": 5.0}),
        ("np_piecewise", r([6]), {}),
        ("np_fill_diagonal", r([3, 3]), {"val": 7.0}),
        ("np_fill_diagonal", r([5, 2]), {"val": 7.0}),
        ("np_tril_indices", r([1]), {"n": 3}),
        ("np_tril_indices", r([1]), {"n": 3, "k": -1, "m": 4}),
        ("np_triu_indices", r([1]), {"n": 3, "k": 1}),
        ("np_tril_indices_from", r([3, 4]), {}),
        ("np_triu_indices_from", r([3, 4]), {"k": 1}),
        ("np_ravel_multi_index", r([1]), {"coords": [[0, 1, 2], [3, 1, 0]], "dims": [3, 4], "mode": "clip"}),
        ("np_ravel_multi_index", r([1]), {"coords": [[5, 1], [7, 1]], "dims": [3, 4], "mode": "wrap"}),
        ("np_trim_zeros", r([3]), {"trim": "fb"}),
        ("np_trim_zeros", r([3]), {"trim": "f"}),
        ("np_frexp", r([5]), {}),
        ("np_ldexp", r([3]), {"e": [1, 2, 3]}),
        ("np_apply_along_axis", r([3, 4]), {"axis": 0}),
        ("np_apply_along_axis", r([3, 4]), {"axis": 1}),
        ("np_apply_along_axis_keep", r([3, 4]), {"axis": 0}),
        ("np_apply_over_axes", r([2, 3, 4]), {"axes": [0, 2]}),
        ("np_polydiv", rr([5], [2]), {}),
        ("np_polyfit", rr([6], [6]), {"deg": 2}),
        ("np_roots", r([4]), {}),
        ("np_block", rr([2, 2], [2, 2]), {}),
    ]


def all_cases():
    return _fft_cases() + _lax_cases() + _numpy_cases()
