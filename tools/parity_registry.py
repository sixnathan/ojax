import numpy as np
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


def _scipy_ops():
    import jax.scipy as js
    from jax.scipy import linalg as sl
    from jax import random as jr
    from jax import nn as jn
    return {
        "sp_dct": lambda a, p: js.fft.dct(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "sp_idct": lambda a, p: js.fft.idct(a, n=p.get("n"), axis=p.get("axis", -1), norm=p.get("norm")),
        "sp_dctn": lambda a, p: js.fft.dctn(a, norm=p.get("norm")),
        "sp_idctn": lambda a, p: js.fft.idctn(a, norm=p.get("norm")),
        "sp_convolve": lambda a, b, p: js.signal.convolve(a, b, mode=p["mode"], method=p.get("method", "auto")),
        "sp_correlate": lambda a, b, p: js.signal.correlate(a, b, mode=p["mode"]),
        "sp_convolve2d": lambda a, b, p: js.signal.convolve2d(a, b, mode=p["mode"]),
        "sp_correlate2d": lambda a, b, p: js.signal.correlate2d(a, b, mode=p["mode"]),
        "sp_fftconvolve": lambda a, b, p: js.signal.fftconvolve(a, b, mode=p["mode"]),
        "sp_detrend": lambda a, p: js.signal.detrend(a, axis=p.get("axis", -1), type=p["type"]),
        "sl_hilbert": lambda a, p: sl.hilbert(p["n"]),
        "sl_circulant": lambda a, p: sl.circulant(a),
        "sl_hankel": lambda a, p: sl.hankel(a),
        "sl_hankel_r": lambda a, b, p: sl.hankel(a, b),
        "sl_fiedler": lambda a, p: sl.fiedler(a),
        "sl_companion": lambda a, p: sl.companion(a),
        "sl_fiedler_companion": lambda a, p: sl.fiedler_companion(a),
        "sl_leslie": lambda a, b, p: sl.leslie(a, b),
        "sl_dft": lambda a, p: sl.dft(p["n"], scale=p.get("scale")),
        "sl_pascal": lambda a, p: sl.pascal(p["n"], kind=p["kind"]),
        "sl_invpascal": lambda a, p: sl.invpascal(p["n"], kind=p["kind"]),
        "sl_helmert": lambda a, p: sl.helmert(p["n"], full=p.get("full", False)),
        "sl_hadamard": lambda a, p: sl.hadamard(p["n"]),
        "sl_convolution_matrix": lambda a, p: sl.convolution_matrix(a, p["n"], mode=p["mode"]),
        "sl_invhilbert": lambda a, p: sl.invhilbert(p["n"]),
        "sl_solve_sylvester": lambda a, b, c, p: sl.solve_sylvester(a + 4.0 * jnp.eye(a.shape[0], dtype=a.dtype), b + 4.0 * jnp.eye(b.shape[0], dtype=b.dtype), c),
        "rnd_maxwell": lambda a, p: jr.maxwell(jr.key(p["seed"]), tuple(p["shape"])),
        "rnd_double_sided_maxwell": lambda a, p: jr.double_sided_maxwell(jr.key(p["seed"]), p["loc"], p["scale"], tuple(p["shape"])),
        "rnd_ball": lambda a, p: jr.ball(jr.key(p["seed"]), p["d"], p=p.get("p", 2), shape=tuple(p["shape"])),
        "rnd_multivariate_normal": lambda a, b, p: jr.multivariate_normal(jr.key(p["seed"]), a, (b @ b.T + 2.0 * jnp.eye(b.shape[0], dtype=b.dtype)), shape=tuple(p["shape"]), method=p["method"]),
        "nn_swish": lambda a, p: jn.swish(a),
        "nn_hard_swish": lambda a, p: jn.hard_swish(a),
        "nn_logsumexp": lambda a, p: jn.logsumexp(a, axis=p["axis"]),
        "nn_attention": lambda a, b, c, p: jn.dot_product_attention(a, b, c),
    }


def _opt_run(name, a, b, p):
    from jax.example_libraries import optimizers as opt
    make = {
        "sgd": lambda: opt.sgd(0.1),
        "momentum": lambda: opt.momentum(0.1, 0.9),
        "nesterov": lambda: opt.nesterov(0.1, 0.9),
        "adagrad": lambda: opt.adagrad(0.1),
        "rmsprop": lambda: opt.rmsprop(0.1),
        "rmsprop_momentum": lambda: opt.rmsprop_momentum(0.1),
        "adam": lambda: opt.adam(0.1),
        "adamax": lambda: opt.adamax(0.1),
        "sm3": lambda: opt.sm3(0.1),
    }[name]
    init, update, get = make()
    state = init(a)
    for i in range(3):
        state = update(i, b, state)
    return get(state)


def _spd(a):
    return a @ jnp.swapaxes(a, -1, -2) + 3.0 * jnp.eye(a.shape[-1], dtype=a.dtype)


def _batched_ops():
    la = jnp.linalg
    return {
        "bl_inv": lambda a, p: la.inv(_spd(a)),
        "bl_det": lambda a, p: la.det(_spd(a)),
        "bl_slogdet": lambda a, p: tuple(la.slogdet(_spd(a))),
        "bl_solve": lambda a, b, p: la.solve(_spd(a), b),
        "bl_cholesky": lambda a, p: la.cholesky(_spd(a)),
        "bl_eigvalsh": lambda a, p: la.eigvalsh(_spd(a)),
        "bl_svdvals": lambda a, p: la.svd(a, compute_uv=False),
        "bl_matrix_power": lambda a, p: la.matrix_power(a, p["n"]),
        "bl_norm": lambda a, p: la.norm(a, axis=(-2, -1)),
        "bl_grad_det": lambda a, p: jax.grad(lambda x: jnp.sum(jnp.linalg.det(_spd(x))))(a),
        "bl_vmap_inv": lambda a, p: jax.vmap(lambda m: la.inv(_spd(m)))(a),
        "np_packbits": lambda a, p: jnp.packbits(a > 0, axis=p.get("axis"), bitorder=p.get("order", "big")),
        "np_unpackbits": lambda a, p: jnp.unpackbits(jnp.packbits(a > 0), count=p.get("count"), bitorder=p.get("order", "big")),
    }


def _misc_scipy_ops():
    import jax.scipy as js
    from jax.scipy.interpolate import RegularGridInterpolator as RGI
    from jax.scipy.stats import gaussian_kde
    def kde(a, p):
        return gaussian_kde(a, bw_method=p.get("bw", "scott"))
    return {
        "sg_stft": lambda a, p: js.signal.stft(a, nperseg=p["nperseg"], noverlap=p.get("noverlap"), boundary=p.get("boundary", "zeros"), padded=p.get("padded", True)),
        "sg_csd": lambda a, b, p: js.signal.csd(a, b, nperseg=p["nperseg"], average=p.get("average", "mean")),
        "sg_welch": lambda a, p: js.signal.welch(a, nperseg=p["nperseg"], detrend=p.get("detrend", "constant"), scaling=p.get("scaling", "density"), average=p.get("average", "mean")),
        "sg_istft": lambda a, p: js.signal.istft(a, nperseg=p["nperseg"]),
        "ip_linear": lambda a, b, p: RGI((jnp.arange(a.shape[0], dtype=a.dtype), jnp.arange(a.shape[1], dtype=a.dtype)), a)(b),
        "ip_nearest": lambda a, b, p: RGI((jnp.arange(a.shape[0], dtype=a.dtype), jnp.arange(a.shape[1], dtype=a.dtype)), a, method="nearest")(b),
        "kde_evaluate": lambda a, b, p: kde(a, p).evaluate(b),
        "kde_logpdf": lambda a, b, p: kde(a, p).logpdf(b),
        "kde_box": lambda a, p: kde(a, p).integrate_box_1d(p["lo"], p["hi"]),
        "sp_fresnel": lambda a, p: tuple(js.special.fresnel(a * p.get("scale", 1.0))),
    }


def _lax_extra_ops():
    return {
        "lax_patches": lambda a, p: lax.conv_general_dilated_patches(a, tuple(p["filter"]), tuple(p["str"]), p["padding"]),
    }


def _array_random_ops():
    from jax import random as jr
    k = lambda p: jr.key(p["seed"])
    return {
        "ra_uniform": lambda a, b, p: jr.uniform(k(p), a.shape, dtype=a.dtype, minval=a, maxval=a + jnp.abs(b) + 1.0),
        "ra_bernoulli": lambda a, p: jr.bernoulli(k(p), jnp.abs(jnp.sin(a))),
        "ra_gamma": lambda a, p: jr.gamma(k(p), a, dtype=a.dtype),
        "ra_loggamma": lambda a, p: jr.loggamma(k(p), a, dtype=a.dtype),
        "ra_beta": lambda a, b, p: jr.beta(k(p), a, b, dtype=a.dtype),
        "ra_poisson": lambda a, p: jr.poisson(k(p), a * 8.0),
        "ra_bits": lambda a, p: jr.bits(k(p), tuple(p["shape"]), dtype=jnp.dtype(p["dtype"])),
        "ra_gamma_grad": lambda a, p: jax.lax.random_gamma_grad(a, a * 0.7 + 0.2),
        "ra_grad_gamma": lambda a, p: jax.grad(lambda x: jnp.sum(jr.gamma(k(p), x, dtype=x.dtype)))(a),
        "sp_fresnel_jit": lambda a, p: jax.jit(lambda x: jax.scipy.special.fresnel(x))(a),
        "sp_fresnel_grad": lambda a, p: jax.grad(lambda x: jnp.sum(jax.scipy.special.fresnel(x)[0]))(a),
    }


def _gap_ops():
    import jax.scipy as js
    from jax.scipy import stats as st
    from jax import nn as jn
    from jax import random as jr
    k = lambda p: jr.key(p["seed"])
    return {
        "gp_qr_full": lambda a, p: jnp.linalg.qr(a, mode="complete"),
        "gp_svd_hermitian": lambda a, p: jnp.linalg.svd(a + a.T, hermitian=True, compute_uv=False),
        "gp_pinv_hermitian": lambda a, p: jnp.linalg.pinv(a + a.T, hermitian=True),
        "gp_init_orthogonal": lambda a, p: jn.initializers.orthogonal()(k(p), tuple(p["shape"]), jnp.float32),
        "gp_init_delta": lambda a, p: jn.initializers.delta_orthogonal()(k(p), tuple(p["shape"]), jnp.float32),
        "gp_mode": lambda a, p: tuple(st.mode(jnp.round(a), axis=p.get("axis", 0), keepdims=p.get("keepdims", False))),
        "gp_rankdata": lambda a, p: st.rankdata(jnp.round(a * 2.0), method=p["method"]),
        "gp_rankdata_axis": lambda a, p: st.rankdata(jnp.round(a * 2.0), method=p["method"], axis=p["axis"]),
        "gp_wofz": lambda a, b, p: js.special.wofz(a + 1j * b),
        "gp_bernoulli": lambda a, p: js.special.bernoulli(p["n"]),
        "gp_expm_batched": lambda a, p: js.linalg.expm(a),
        "gp_reduce_precision": lambda a, p: jax.lax.reduce_precision(a * p["scale"], p["e"], p["m"]),
        "gp_clamp_jit": lambda a, p: jax.jit(lambda x: jax.lax.clamp(jnp.asarray(-0.5, x.dtype), x, jnp.asarray(0.5, x.dtype)))(a),
        "gp_int_pow_neg": lambda a, p: jax.jit(lambda x: jax.lax.integer_pow(x, -2))(a),
    }


def _herm(a):
    return a @ jnp.conj(jnp.swapaxes(a, -1, -2)) + 3.0 * jnp.eye(a.shape[-1], dtype=a.dtype)


def _complex_linalg_ops():
    la = jnp.linalg
    return {
        "cl_inv": lambda a, p: la.inv(_herm(a)),
        "cl_det": lambda a, p: la.det(_herm(a)),
        "cl_solve": lambda a, b, p: la.solve(_herm(a), b),
        "cl_cholesky": lambda a, p: la.cholesky(_herm(a)),
        "cl_eigvalsh": lambda a, p: la.eigvalsh(_herm(a)),
        "cl_svdvals": lambda a, p: la.svd(a, compute_uv=False),
        "cl_qr_r": lambda a, p: jnp.abs(la.qr(a)[1]),
        "cl_slogdet": lambda a, p: tuple(la.slogdet(_herm(a))),
        "cl_grad_eigvalsh": lambda a, p: jax.grad(lambda x: jnp.sum(jnp.linalg.eigvalsh(_herm(x))))(a),
        "cl_grad_logdet": lambda a, p: jax.grad(lambda x: jnp.linalg.slogdet(_herm(x))[1].sum())(a),
        "cl_grad_eigvals": lambda a, p: jax.grad(lambda x: jnp.sum(jnp.abs(jnp.linalg.eigvals(x))))(a),
        "cl_jvp_eigvals": lambda a, p: jnp.sum(jax.jvp(jnp.linalg.eigvals, (a,), (a * 0.5 + 0.1,))[1]),
        "cl_grad_svdvals": lambda a, p: jax.grad(lambda x: jnp.sum(jnp.linalg.svd(x, compute_uv=False)))(a),
    }


def _complex_grad_ops():
    n = jnp
    fns = {
        "abs": lambda z: n.abs(z),
        "sin": lambda z: n.abs(n.sin(z)),
        "cos": lambda z: n.abs(n.cos(z)),
        "tan": lambda z: n.abs(n.tan(z)),
        "exp": lambda z: n.abs(n.exp(z)),
        "log": lambda z: n.abs(n.log(z)),
        "sqrt": lambda z: n.abs(n.sqrt(z)),
        "tanh": lambda z: n.abs(n.tanh(z)),
        "sinh": lambda z: n.abs(n.sinh(z)),
        "cosh": lambda z: n.abs(n.cosh(z)),
        "square": lambda z: n.abs(z * z),
        "reciprocal": lambda z: n.abs(1.0 / z),
        "conj": lambda z: n.abs(n.conj(z) * z),
        "real": lambda z: n.real(z) ** 2,
        "imag": lambda z: n.imag(z) ** 2,
        "matmul": lambda z: n.abs(z @ n.conj(z)),
        "fft": lambda z: n.abs(n.fft.fft(z)),
        "angle": lambda z: n.angle(z),
    }
    return {"cg_" + k: (lambda f: lambda a, p: jax.grad(lambda x: n.sum(f(x)))(a))(v) for k, v in fns.items()}


def _linalg_batch_ops():
    ll = jax.lax.linalg
    sl = jax.scipy.linalg
    recon = lambda z, t: z @ t @ jnp.conj(jnp.swapaxes(z, -1, -2))
    diag_mag = lambda t: jnp.sort(jnp.abs(jnp.diagonal(t, axis1=-2, axis2=-1)), axis=-1)

    def rsf(a):
        t, z = sl.schur(a)
        tc, zc = sl.rsf2csf(t, z)
        return recon(zc, tc), diag_mag(tc)

    def schur_c(a):
        t, z = sl.schur(a, output="complex")
        return recon(z, t), diag_mag(t)

    def hess(a):
        h, q = sl.hessenberg(a, calc_q=True)
        return recon(q, h), jnp.abs(h)

    def tri(a):
        _, d, e, _ = ll.tridiagonal(_herm(a))
        return d, jnp.abs(e)

    def ormqr(a, b, p):
        from jax._src.lax.linalg import geqrf

        h, t = geqrf(a)
        return ll.ormqr(h, t, b, left=p["left"], transpose=p["transpose"])

    def qdwh(a):
        u, h, _, ok = ll.qdwh(a)
        return u, h, ok

    def qrp(a, p):
        out = sl.qr(a, pivoting=True, mode=p["mode"])
        if p["mode"] == "r":
            return jnp.abs(out[0]), out[1]
        return jnp.abs(out[0]), jnp.abs(out[1]), out[2]

    def qrm(a, b, p):
        out = sl.qr_multiply(a, b, mode=p["mode"], conjugate=p["conjugate"], pivoting=p["pivoting"])
        return tuple(out)

    return {
        "lx_cholesky_update": lambda a, b, p: ll.cholesky_update(jnp.swapaxes(jnp.linalg.cholesky(_spd(a)), -1, -2), b),
        "lx_symmetric_product": lambda a, b, p: ll.symmetric_product(a, _spd(b), alpha=2.0, beta=0.5, symmetrize_output=p["sym"]),
        "lx_ormqr": ormqr,
        "lx_qdwh": lambda a, p: qdwh(a),
        "lx_qr_pivot": lambda a, p: qrp(a, p),
        "lx_qr_multiply": qrm,
        "lx_rsf2csf": lambda a, p: rsf(a),
        "lx_schur_complex": lambda a, p: schur_c(a),
        "lx_hessenberg": lambda a, p: hess(a),
        "lx_tridiagonal": lambda a, p: tri(a),
        "lx_sqrtm": lambda a, p: (lambda r: (r @ r, sl.sqrtm(_herm(a))))(sl.sqrtm(a)),
        "lx_expm_frechet": lambda a, b, p: tuple(sl.expm_frechet(a, b)),
        "lx_polar": lambda a, p: tuple(sl.polar(a, side=p["side"], method=p["method"])),
        "lx_eigh_tri": lambda a, b, p: (lambda o: (o[0], jnp.abs(o[1])))(sl.eigh_tridiagonal(a, b)),
    }


def _surface_batch_ops():
    lx = jax.lax
    over = lambda f, a: jax.vmap(f, axis_name="i")(a)
    key = jnp.array([1, 2, 3, 4], dtype=jnp.uint32)

    def rng(p):
        algo = getattr(lx.RandomAlgorithm, p["algorithm"])
        k, v = lx.rng_bit_generator(key, tuple(p["shape"]), dtype=jnp.dtype(p["dtype"]), algorithm=algo)
        return k, v

    def uf_add():
        return jnp.frompyfunc(lambda x, y: x * y + 1.0, 2, 1)

    return {
        "sb_approx_max": lambda a, p: tuple(lx.approx_max_k(a, p["k"])),
        "sb_approx_min": lambda a, p: tuple(lx.approx_min_k(a, p["k"])),
        "sb_broadcast_like": lambda a, b, p: lx.broadcast_like(a, b),
        "sb_ragged_dot": lambda a, b, p: lx.ragged_dot(a, b, jnp.array(p["groups"])),
        "sb_conv_local": lambda a, b, p: lx.conv_general_dilated_local(a, b, (1, 1), "VALID", tuple(p["filter"])),
        "sb_rng": lambda a, p: rng(p),
        "sb_all_to_all": lambda a, p: over(lambda x: lx.all_to_all(x, "i", p["split"], p["concat"], tiled=p["tiled"]), a),
        "sb_pswapaxes": lambda a, p: over(lambda x: lx.pswapaxes(x, "i", p["axis"]), a),
        "sb_psum_scatter": lambda a, p: over(lambda x: lx.psum_scatter(x, "i", scatter_dimension=p["dim"], tiled=p["tiled"]), a),
        "sb_pshuffle": lambda a, p: over(lambda x: lx.pshuffle(x, "i", p["perm"]), a),
        "sb_mgrid_int": lambda a, p: jnp.mgrid[0:4, 1:7:2],
        "sb_mgrid_count": lambda a, p: jnp.mgrid[0:1:5j, 0:2:3j],
        "sb_mgrid_one": lambda a, p: jnp.mgrid[1:5],
        "sb_ogrid": lambda a, p: tuple(jnp.ogrid[0:3, 0:1:4j]),
        "sb_r": lambda a, b, p: jnp.r_[a, b],
        "sb_c": lambda a, b, p: jnp.c_[a, b],
        "sb_ufunc_call": lambda a, b, p: uf_add()(a, b),
        "sb_ufunc_reduce": lambda a, p: uf_add().reduce(a, axis=p["axis"]),
        "sb_ufunc_accumulate": lambda a, p: uf_add().accumulate(a, axis=p["axis"]),
        "sb_ufunc_outer": lambda a, b, p: uf_add().outer(a, b),
    }


def _tracer_batch_ops():
    key = jax.random.key if False else None

    def uq(a, p):
        out = jnp.unique(a, size=p["size"], fill_value=p.get("fill"), axis=p.get("axis"), return_index=True, return_inverse=True, return_counts=True)
        return tuple(out)

    def pad_batched(a, b, p):
        return jax.vmap(lambda x, v: jax.lax.pad(x, v, [(1, 1, 0)]))(a, b)

    return {
        "tb_unique": uq,
        "tb_lexsort": lambda a, p: jnp.lexsort((a[:, 0], a[:, 1])),
        "tb_poly_matrix": lambda a, p: jnp.poly(a),
        "tb_qr_raw": lambda a, p: tuple(jnp.linalg.qr(a, mode="raw")),
        "tb_vectorize": lambda a, b, p: jnp.vectorize(lambda x, y: x * y + 1.0)(a, b),
        "tb_vectorize_core": lambda a, p: jnp.vectorize(lambda x: jnp.sum(x, keepdims=False), signature="(n)->()")(a),
        "tb_pad_vmap": pad_batched,
        "tb_convolve_c": lambda a, b, p: jnp.convolve(a, b, mode=p["mode"]),
        "tb_correlate_c": lambda a, b, p: jnp.correlate(a, b, mode=p["mode"]),
        "tb_lax_conv_c": lambda a, b, p: jax.lax.conv_general_dilated(a, b, (1, 1), "SAME"),
        "tb_grad_conv_c": lambda a, b, p: jax.grad(lambda x: jnp.sum(jnp.abs(jnp.convolve(x, b))))(a),
        "tb_seed": lambda a, p: jax.random.key_data(jax.random.PRNGKey(jnp.sum(a).astype(jnp.int32))),
    }


def _dtype_sweep_ops():
    n = jnp
    un = {
        "mean": lambda a: n.mean(a),
        "mean_axis": lambda a: n.mean(a, axis=0),
        "var": lambda a: n.var(a, axis=0),
        "std": lambda a: n.std(a, axis=0),
        "sum": lambda a: n.sum(a, axis=0),
        "prod": lambda a: n.prod(a, axis=0),
        "cumsum": lambda a: n.cumsum(a, axis=0),
        "cumprod": lambda a: n.cumprod(a, axis=0),
        "diff": lambda a: n.diff(a, axis=0),
        "gradient": lambda a: n.gradient(a, axis=0),
        "trapezoid": lambda a: n.trapezoid(a, axis=0),
        "average": lambda a: n.average(a, axis=0),
        "cov": lambda a: n.cov(a),
        "corrcoef": lambda a: n.corrcoef(a),
        "sort": lambda a: n.sort(a, axis=0),
        "argsort": lambda a: n.argsort(a, axis=0),
        "flip": lambda a: n.flip(a, axis=0),
        "roll": lambda a: n.roll(a, 1, axis=0),
        "nanmean": lambda a: n.nanmean(a, axis=0),
        "nansum": lambda a: n.nansum(a, axis=0),
        "nanvar": lambda a: n.nanvar(a, axis=0),
        "sinc": lambda a: n.sinc(a),
        "sign": lambda a: n.sign(a),
        "square": lambda a: n.square(a),
        "sqrt": lambda a: n.sqrt(a),
        "exp": lambda a: n.exp(a),
        "log": lambda a: n.log(a),
        "log1p": lambda a: n.log1p(a),
        "expm1": lambda a: n.expm1(a),
        "sin": lambda a: n.sin(a),
        "cos": lambda a: n.cos(a),
        "tan": lambda a: n.tan(a),
        "arcsin": lambda a: n.arcsin(a),
        "arccos": lambda a: n.arccos(a),
        "arctan": lambda a: n.arctan(a),
        "sinh": lambda a: n.sinh(a),
        "cosh": lambda a: n.cosh(a),
        "tanh": lambda a: n.tanh(a),
        "arcsinh": lambda a: n.arcsinh(a),
        "arccosh": lambda a: n.arccosh(a),
        "arctanh": lambda a: n.arctanh(a),
        "real": lambda a: n.real(a),
        "imag": lambda a: n.imag(a),
        "conj": lambda a: n.conj(a),
        "angle": lambda a: n.angle(a),
        "abs": lambda a: n.abs(a),
        "isnan": lambda a: n.isnan(a),
        "isfinite": lambda a: n.isfinite(a),
        "isinf": lambda a: n.isinf(a),
        "round": lambda a: n.round(a, 1),
        "floor_div_self": lambda a: (lambda x: x / (x + 2.0))(n.asarray(a)),
        "linspace_end": lambda a: n.linspace(a[0], a[1], 4),
        "geomspace": lambda a: (lambda x: n.geomspace(x[0] + 2.0, x[1] + 3.0, 3))(n.asarray(a)),
        "logspace": lambda a: n.logspace(a[0], a[1], 3),
        "polyval": lambda a: n.polyval(a[:2], a[0]),
        "polyder": lambda a: n.polyder(a[0]),
        "polyint": lambda a: n.polyint(a[0]),
        "polyadd": lambda a: n.polyadd(a[0], a[1]),
        "polymul": lambda a: n.polymul(a[0], a[1]),
        "roots": lambda a: n.sort(n.abs(n.roots(a[0]))),
        "kron": lambda a: n.kron(a, a),
        "dot": lambda a: n.dot(a, a.T),
        "vdot": lambda a: n.vdot(a, a),
        "inner": lambda a: n.inner(a, a),
        "outer": lambda a: n.outer(a[0], a[1]),
        "matmul": lambda a: a @ a.T,
        "trace": lambda a: n.trace(a @ a.T),
        "cross": lambda a: n.cross(a[0], a[1]),
        "tensordot": lambda a: n.tensordot(a, a, axes=1) if False else n.tensordot(a, a.T, axes=1),
        "einsum": lambda a: n.einsum("ij,kj->ik", a, a),
        "clip": lambda a: n.clip(a, -0.5, 0.5),
        "minimum": lambda a: n.minimum(a, a[::-1]),
        "maximum": lambda a: n.maximum(a, a[::-1]),
        "max": lambda a: n.max(a, axis=0),
        "min": lambda a: n.min(a, axis=0),
        "argmax": lambda a: n.argmax(a, axis=0),
        "take": lambda a: n.take(a, jnp.array([0, 1]), axis=0),
        "where": lambda a: (lambda x: n.where(n.real(x) > 0, x, x * 2.0))(n.asarray(a)),
        "concatenate": lambda a: n.concatenate([a, a], axis=0),
        "pad": lambda a: n.pad(a, 1),
        "repeat": lambda a: n.repeat(a, 2, axis=0),
        "tile": lambda a: n.tile(a, (2, 1)),
        "interp": lambda a: n.interp(n.array([0.5]), n.array([0.0, 1.0, 2.0]), a[0]),
        "fft": lambda a: n.fft.fft(a, axis=0),
        "ifft": lambda a: n.fft.ifft(a, axis=0),
        "tril": lambda a: n.tril(a),
        "diag": lambda a: n.diag(a),
    }
    return {"sw_" + k: (lambda f: lambda a, p: f(a))(v) for k, v in un.items()}


def _special_batch_ops():
    sp = jax.scipy.special

    def h2(a, p):
        args = [jnp.full(a.shape, v) for v in p["abc"]]
        return sp.hyp2f1(args[0], args[1], args[2], p["scale"] * jnp.tanh(a))

    def sph(a, p):
        n = jnp.array(p["n"])
        m = jnp.array(p["m"])
        out = sp.sph_harm_y(n, m, jnp.abs(a), a * 0.5 + 0.2, n_max=p["n_max"])
        return out.astype(jnp.complex64) if a.dtype == np.float32 else out

    return {
        "sx_expi": lambda a, p: sp.expi(a * p["scale"]),
        "sx_expi_grad": lambda a, p: jax.vmap(jax.grad(sp.expi))(a * p["scale"] + 0.5),
        "sx_hyp2f1": h2,
        "sx_hyp2f1_grad": lambda a, p: jax.vmap(jax.grad(lambda x: sp.hyp2f1(p["abc"][0], p["abc"][1], p["abc"][2], x)))(0.5 * jnp.tanh(a)),
        "sx_lpmn": lambda a, p: tuple(sp.lpmn(p["l"], p["l"], jnp.tanh(a))),
        "sx_lpmn_values": lambda a, p: sp.lpmn_values(p["l"], p["l"], jnp.tanh(a), p["norm"]),
        "sx_sph": sph,
        "sx_expn": lambda a, p: sp.expn(p["n"], jnp.abs(a) * p["scale"]),
        "sx_exp1": lambda a, p: sp.exp1(jnp.abs(a) * p["scale"]),
        "sx_expn_grad": lambda a, p: jax.vmap(jax.grad(lambda x: sp.expn(p["n"], x)))(jnp.abs(a) * p["scale"] + 0.1),
    }


def _set_output_tol(entry, value):
    for out in entry["outputs"]:
        if out["compare"] == "allclose":
            out["tol"] = {"atol": value, "rtol": value}
    entry["tol"] = {"atol": value, "rtol": value}


def _lpmn_tol(entry, dtype):
    if dtype == "float64":
        _set_output_tol(entry, 1e-13)
    return "derivative recurrence cancels terms, so jit-compiled jax differs from eager evaluation in the last bits"


LOOSE_TOLERANCE = {"sx_lpmn": _lpmn_tol}


def _default_random_ops():
    from jax import random as jr
    k = lambda p: jr.key(p["seed"])
    sh = lambda p: tuple(p["shape"])
    return {
        "rd_uniform": lambda a, p: jr.uniform(k(p), sh(p), minval=-1.0, maxval=3.0),
        "rd_normal": lambda a, p: jr.normal(k(p), sh(p)),
        "rd_randint": lambda a, p: jr.randint(k(p), sh(p), -3, 50),
        "rd_truncated_normal": lambda a, p: jr.truncated_normal(k(p), -1.0, 2.0, sh(p)),
        "rd_exponential": lambda a, p: jr.exponential(k(p), sh(p)),
        "rd_cauchy": lambda a, p: jr.cauchy(k(p), sh(p)),
        "rd_laplace": lambda a, p: jr.laplace(k(p), sh(p)),
        "rd_logistic": lambda a, p: jr.logistic(k(p), sh(p)),
        "rd_gumbel": lambda a, p: jr.gumbel(k(p), sh(p)),
        "rd_bernoulli": lambda a, p: jr.bernoulli(k(p), 0.4, sh(p)),
        "rd_rademacher": lambda a, p: jr.rademacher(k(p), sh(p)),
        "rd_gamma": lambda a, p: jr.gamma(k(p), 2.5, sh(p)),
        "rd_beta": lambda a, p: jr.beta(k(p), 2.0, 3.0, sh(p)),
        "rd_chisquare": lambda a, p: jr.chisquare(k(p), 3.0, sh(p)),
        "rd_t": lambda a, p: jr.t(k(p), 4.0, sh(p)),
        "rd_f": lambda a, p: jr.f(k(p), 3.0, 5.0, sh(p)),
        "rd_lognormal": lambda a, p: jr.lognormal(k(p), 0.5, sh(p)),
        "rd_pareto": lambda a, p: jr.pareto(k(p), 2.0, sh(p)),
        "rd_rayleigh": lambda a, p: jr.rayleigh(k(p), 1.5, sh(p)),
        "rd_wald": lambda a, p: jr.wald(k(p), 1.5, sh(p)),
        "rd_geometric": lambda a, p: jr.geometric(k(p), 0.3, sh(p)),
        "rd_poisson": lambda a, p: jr.poisson(k(p), 3.0, sh(p)),
        "rd_binomial": lambda a, p: jr.binomial(k(p), 10.0, 0.3, sh(p)),
        "rd_poisson_large": lambda a, p: jr.poisson(k(p), 30.0, sh(p)),
        "rd_binomial_large": lambda a, p: jr.binomial(k(p), 100.0, 0.3, sh(p)),
        "rd_binomial_flip": lambda a, p: jr.binomial(k(p), 100.0, 0.7, sh(p)),
        "rd_permutation": lambda a, p: jr.permutation(k(p), 6),
        "rd_choice": lambda a, p: jr.choice(k(p), 6, sh(p)),
    }


def _in_range(a, name):
    return jnp.abs(a) if name.startswith("uint") else a


def _narrow_ops():
    def typed(name):
        return jnp.dtype(jnp.bfloat16) if name == "bfloat16" else jnp.dtype(name)
    return {
        "dt_cast": lambda a, p: (_in_range(a, p["to"]) * 40.0).astype(typed(p["to"])),
        "dt_wrap_add": lambda a, p: jnp.asarray(p["x"], typed(p["to"])) + jnp.asarray(p["y"], typed(p["to"])),
        "dt_binary": lambda a, b, p: jnp.add((_in_range(a, p["x"]) * 5.0).astype(typed(p["x"])), (_in_range(b, p["y"]) * 5.0).astype(typed(p["y"]))),
        "dt_float_ops": lambda a, p: jnp.sin(a.astype(typed(p["to"]))) * 2.0,
        "dt_sum": lambda a, p: jnp.sum((_in_range(a, p["to"]) * 5.0).astype(typed(p["to"]))),
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


SINGLE_PRECISION_ONLY = {
    "ra_gamma_grad",
    "ra_grad_gamma",
    "nn_attention",
}

OPS = {}
OPS.update(_fft_ops())
OPS.update(_lax_ops())
OPS.update(_numpy_ops())
OPS.update(_scipy_ops())
OPS.update(_narrow_ops())
OPS.update(_batched_ops())
OPS.update(_misc_scipy_ops())
OPS.update(_lax_extra_ops())
OPS.update(_default_random_ops())
OPS.update(_array_random_ops())
OPS.update(_gap_ops())
OPS.update(_complex_linalg_ops())
OPS.update(_complex_grad_ops())
OPS.update(_linalg_batch_ops())
OPS.update(_surface_batch_ops())
OPS.update(_tracer_batch_ops())
OPS.update(_dtype_sweep_ops())
OPS.update(_special_batch_ops())


def _transform_sweep_ops():
    base = _dtype_sweep_ops()
    out = {}
    for k, fn in base.items():
        name = k[len("sw_"):]
        f = (lambda g: lambda x: g(x, {}))(fn)
        out["swv_" + name] = (lambda h: lambda a, p: jax.vmap(h)(a))(f)
        out["swg_" + name] = (lambda h: lambda a, p: jax.grad(lambda x: jnp.sum(jnp.abs(h(x)) ** 2))(a))(f)
        out["swj_" + name] = (lambda h: lambda a, p: jax.jvp(h, (a,), (a * 0.5 + 0.25,))[1])(f)
    return out


OPS.update(_transform_sweep_ops())
for _name in ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "rmsprop_momentum", "adam", "adamax", "sm3"]:
    OPS["opt_" + _name] = (lambda n: lambda a, b, p: _opt_run(n, a, b, p))(_name)


def _complex_input(shape):
    return {"a": {"kind": "complex", "shape": shape}}


def _positive_input(shape):
    return {"a": {"kind": "positive", "shape": shape}}


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


def _scipy_cases():
    r = _real_input
    c1 = _complex_input
    rr = lambda sa, sb: _two("real", sa, "real", sb)
    out = []
    for op in ["sp_dct", "sp_idct"]:
        out += [
            (op, r([8]), {}),
            (op, r([7]), {"norm": "ortho"}),
            (op, r([3, 4]), {"axis": 0}),
            (op, r([4]), {"n": 6}),
        ]
    for op in ["sp_dctn", "sp_idctn"]:
        out += [(op, r([3, 4]), {}), (op, r([3, 4]), {"norm": "ortho"})]
    for mode in ["full", "same", "valid"]:
        out += [
            ("sp_convolve", rr([6], [3]), {"mode": mode}),
            ("sp_convolve", rr([3], [6]), {"mode": mode}),
            ("sp_correlate", rr([6], [3]), {"mode": mode}),
            ("sp_convolve2d", rr([4, 5], [2, 3]), {"mode": mode}),
            ("sp_correlate2d", rr([4, 5], [2, 3]), {"mode": mode}),
            ("sp_fftconvolve", rr([6], [3]), {"mode": mode}),
        ]
    out += [
        ("sp_convolve", rr([6], [3]), {"mode": "full", "method": "fft"}),
        ("sp_detrend", r([6]), {"type": "constant"}),
        ("sp_detrend", r([8]), {"type": "linear"}),
        ("sp_detrend", r([3, 5]), {"type": "linear", "axis": 0}),
        ("sl_hilbert", r([1]), {"n": 4}),
        ("sl_circulant", r([4]), {}),
        ("sl_hankel", r([3]), {}),
        ("sl_hankel_r", rr([3], [4]), {}),
        ("sl_fiedler", r([4]), {}),
        ("sl_companion", r([4]), {}),
        ("sl_fiedler_companion", r([2]), {}),
        ("sl_fiedler_companion", r([3]), {}),
        ("sl_fiedler_companion", r([5]), {}),
        ("sl_fiedler_companion", r([6]), {}),
        ("sl_leslie", rr([4], [3]), {}),
        ("sl_dft", r([1]), {"n": 4}),
        ("sl_dft", r([1]), {"n": 3, "scale": "sqrtn"}),
        ("sl_dft", r([1]), {"n": 3, "scale": "n"}),
        ("sl_pascal", r([1]), {"n": 4, "kind": "symmetric"}),
        ("sl_pascal", r([1]), {"n": 4, "kind": "lower"}),
        ("sl_pascal", r([1]), {"n": 4, "kind": "upper"}),
        ("sl_invpascal", r([1]), {"n": 4, "kind": "symmetric"}),
        ("sl_invpascal", r([1]), {"n": 4, "kind": "lower"}),
        ("sl_invpascal", r([1]), {"n": 4, "kind": "upper"}),
        ("sl_helmert", r([1]), {"n": 4}),
        ("sl_helmert", r([1]), {"n": 4, "full": True}),
        ("sl_hadamard", r([1]), {"n": 4}),
        ("sl_convolution_matrix", r([3]), {"n": 4, "mode": "full"}),
        ("sl_convolution_matrix", r([3]), {"n": 5, "mode": "same"}),
        ("sl_convolution_matrix", r([3]), {"n": 5, "mode": "valid"}),
        ("sl_invhilbert", r([1]), {"n": 3}),
        ("sl_solve_sylvester", {"a": {"kind": "real", "shape": [2, 2]}, "b": {"kind": "real", "shape": [3, 3]}, "c": {"kind": "real", "shape": [2, 3]}}, {}),
        ("rnd_maxwell", r([1]), {"seed": 1, "shape": [4]}),
        ("rnd_double_sided_maxwell", r([1]), {"seed": 2, "loc": 1.0, "scale": 2.0, "shape": [4]}),
        ("rnd_ball", r([1]), {"seed": 3, "d": 3, "shape": [4]}),
        ("rnd_multivariate_normal", rr([3], [3, 3]), {"seed": 4, "shape": [2], "method": "cholesky"}),
        ("nn_swish", r([4]), {}),
        ("nn_hard_swish", r([4]), {}),
        ("nn_logsumexp", r([2, 3]), {"axis": 1}),
        ("nn_attention", {"a": {"kind": "real", "shape": [3, 2, 4]}, "b": {"kind": "real", "shape": [5, 2, 4]}, "c": {"kind": "real", "shape": [5, 2, 4]}}, {}),
    ]
    _ = c1
    return out


def _optimizer_cases():
    out = []
    for name in ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "rmsprop_momentum", "adam", "adamax", "sm3"]:
        out.append(("opt_" + name, _two("real", [3], "positive", [3]), {}))
        out.append(("opt_" + name, _two("real", [2, 2], "positive", [2, 2]), {}))
    return out


def _narrow_cases():
    r = _real_input
    rr = lambda: _two("real", [4], "real", [4])
    names = ["int8", "int16", "uint8", "uint16", "float16", "bfloat16"]
    out = []
    for n in names:
        out.append(("dt_cast", r([5]), {"to": n}))
        out.append(("dt_sum", r([4]), {"to": n}))
    out += [
        ("dt_wrap_add", r([1]), {"to": "int8", "x": 120, "y": 100}),
        ("dt_wrap_add", r([1]), {"to": "uint8", "x": 200, "y": 100}),
        ("dt_wrap_add", r([1]), {"to": "int16", "x": 30000, "y": 10000}),
        ("dt_wrap_add", r([1]), {"to": "uint16", "x": 60000, "y": 10000}),
        ("dt_float_ops", r([4]), {"to": "float16"}),
        ("dt_float_ops", r([4]), {"to": "bfloat16"}),
    ]
    for x, y in [
        ("int8", "int16"),
        ("int8", "uint8"),
        ("uint8", "int16"),
        ("uint16", "int16"),
        ("uint8", "uint16"),
        ("int8", "float16"),
        ("float16", "bfloat16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("uint32", "int32"),
        ("uint8", "float32"),
        ("int32", "float16"),
    ]:
        out.append(("dt_binary", rr(), {"x": x, "y": y}))
    return out


def _batched_cases():
    r = _real_input
    out = []
    for shape in [[3, 3], [2, 3, 3], [2, 2, 3, 3]]:
        for op in ["bl_inv", "bl_det", "bl_slogdet", "bl_cholesky", "bl_eigvalsh", "bl_svdvals", "bl_norm", "bl_grad_det"]:
            out.append((op, r(shape), {}))
        out.append(("bl_matrix_power", r(shape), {"n": 3}))
    out += [
        ("bl_solve", _two("real", [2, 3, 3], "real", [2, 3, 2]), {}),
        ("bl_solve", _two("real", [2, 3, 3], "real", [3, 2]), {}),
        ("bl_solve", _two("real", [2, 3, 3], "real", [3]), {}),
        ("bl_vmap_inv", r([2, 3, 3]), {}),
        ("np_packbits", r([12]), {}),
        ("np_packbits", r([2, 10]), {"axis": 1, "order": "little"}),
        ("np_unpackbits", r([12]), {}),
        ("np_unpackbits", r([12]), {"count": 5, "order": "little"}),
    ]
    return out


def _misc_scipy_cases():
    r = _real_input
    rr = lambda sa, sb: _two("real", sa, "real", sb)
    return [
        ("sg_stft", r([32]), {"nperseg": 8}),
        ("sg_stft", r([32]), {"nperseg": 8, "noverlap": 6}),
        ("sg_stft", r([2, 32]), {"nperseg": 8}),
        ("sg_stft", r([30]), {"nperseg": 8, "boundary": None, "padded": False}),
        ("sg_csd", rr([32], [32]), {"nperseg": 8}),
        ("sg_csd", rr([32], [32]), {"nperseg": 8, "average": "median"}),
        ("sg_welch", r([32]), {"nperseg": 8}),
        ("sg_welch", r([32]), {"nperseg": 8, "scaling": "spectrum"}),
        ("sg_welch", r([32]), {"nperseg": 8, "average": "median"}),
        ("sg_welch", r([2, 32]), {"nperseg": 8, "detrend": "linear"}),
        ("sg_istft", _complex_input([5, 9]), {"nperseg": 8}),
        ("ip_linear", rr([3, 4], [5, 2]), {}),
        ("ip_nearest", rr([3, 4], [5, 2]), {}),
        ("kde_evaluate", rr([2, 6], [2, 3]), {}),
        ("kde_evaluate", rr([2, 6], [2, 3]), {"bw": "silverman"}),
        ("kde_logpdf", rr([2, 6], [2, 3]), {}),
        ("kde_box", r([1, 8]), {"lo": -1.0, "hi": 1.0}),
        ("sp_fresnel", r([8]), {}),
        ("sp_fresnel", r([8]), {"scale": 6.0}),
    ]


def _lax_extra_cases():
    return [
        ("lax_patches", _real_input([1, 1, 6]), {"filter": [2], "str": [1], "padding": "VALID"}),
        ("lax_patches", _real_input([1, 2, 6]), {"filter": [3], "str": [2], "padding": "VALID"}),
    ]


def _default_random_cases():
    ops = "uniform normal randint truncated_normal exponential cauchy laplace logistic gumbel bernoulli rademacher gamma beta chisquare t f lognormal pareto rayleigh wald geometric poisson poisson_large binomial binomial_large binomial_flip permutation choice"
    return [("rd_" + o, _real_input([1]), {"seed": 11, "shape": [4]}) for o in ops.split()]


def _array_random_cases():
    pos = lambda shape: {"a": {"kind": "positive", "shape": shape}}
    r = _real_input
    two_pos = {"a": {"kind": "positive", "shape": [3]}, "b": {"kind": "positive", "shape": [3]}}
    out = [
        ("ra_uniform", _two("real", [3], "real", [3]), {"seed": 3}),
        ("ra_bernoulli", r([6]), {"seed": 4}),
        ("ra_gamma", pos([3]), {"seed": 5}),
        ("ra_loggamma", pos([3]), {"seed": 6}),
        ("ra_beta", two_pos, {"seed": 7}),
        ("ra_poisson", pos([6]), {"seed": 8}),
        ("ra_gamma_grad", pos([3]), {}),
        ("ra_grad_gamma", pos([3]), {"seed": 9}),
        ("sp_fresnel_jit", r([4]), {}),
        ("sp_fresnel_grad", r([4]), {}),
    ]
    for dt in ["uint8", "uint16", "uint32", "uint64"]:
        out.append(("ra_bits", r([1]), {"seed": 10, "shape": [3], "dtype": dt}))
    return out


def _gap_cases():
    r = _real_input
    pos = lambda shape: {"a": {"kind": "positive", "shape": shape}}
    out = [
        ("gp_qr_full", r([4, 2]), {}),
        ("gp_svd_hermitian", r([3, 3]), {}),
        ("gp_pinv_hermitian", r([3, 3]), {}),
        ("gp_init_orthogonal", r([1]), {"seed": 1, "shape": [4, 3]}),
        ("gp_init_orthogonal", r([1]), {"seed": 2, "shape": [3, 4]}),
        ("gp_init_delta", r([1]), {"seed": 3, "shape": [3, 2, 2]}),
        ("gp_mode", r([7]), {}),
        ("gp_mode", r([6, 2]), {"keepdims": True}),
        ("gp_wofz", _two("real", [5], "real", [5]), {}),
        ("gp_bernoulli", r([1]), {"n": 6}),
        ("gp_bernoulli", r([1]), {"n": 7}),
        ("gp_expm_batched", r([2, 3, 3]), {}),
        ("gp_reduce_precision", r([4]), {"scale": 1.0, "e": 5, "m": 10}),
        ("gp_reduce_precision", r([4]), {"scale": 1000.0, "e": 3, "m": 5}),
        ("gp_clamp_jit", r([4]), {}),
        ("gp_int_pow_neg", pos([4]), {}),
    ]
    for m in ["average", "min", "max", "dense", "ordinal"]:
        out.append(("gp_rankdata", r([8]), {"method": m}))
    out.append(("gp_rankdata_axis", r([2, 5]), {"method": "average", "axis": 1}))
    return out


def _complex_linalg_cases():
    c = _complex_input
    out = []
    for op in ["cl_inv", "cl_det", "cl_cholesky", "cl_eigvalsh", "cl_slogdet", "cl_grad_eigvalsh", "cl_grad_logdet"]:
        out.append((op, c([3, 3]), {}))
        out.append((op, c([2, 3, 3]), {}))
    for op in ["cl_grad_eigvals", "cl_jvp_eigvals"]:
        out.append((op, c([3, 3]), {}))
        out.append((op, _real_input([3, 3]), {}))
    for shape in [[3, 3], [4, 2], [2, 4]]:
        out.append(("cl_svdvals", c(shape), {}))
        out.append(("cl_qr_r", c(shape), {}))
        out.append(("cl_grad_svdvals", c(shape), {}))
    out.append(("cl_solve", {"a": {"kind": "complex", "shape": [3, 3]}, "b": {"kind": "complex", "shape": [3, 2]}}, {}))
    return out


def _complex_grad_cases():
    names = "abs sin cos tan exp log sqrt tanh sinh cosh square reciprocal conj real imag fft angle".split()
    return [("cg_" + n, _complex_input([4]), {}) for n in names] + [("cg_matmul", _complex_input([2, 2]), {})]


def _linalg_batch_cases():
    def two(kind, sa, sb):
        return {"a": {"kind": kind, "shape": sa}, "b": {"kind": kind, "shape": sb}}

    out = [
        ("lx_cholesky_update", two("real", [3, 3], [3]), {}),
        ("lx_symmetric_product", two("real", [3, 2], [3, 3]), {"sym": True}),
        ("lx_symmetric_product", two("real", [3, 2], [3, 3]), {"sym": False}),
    ]
    for kind in ["real", "complex"]:
        for left in [True, False]:
            for transpose in [True, False]:
                sb = [4, 2] if left else [2, 4]
                out.append(("lx_ormqr", two(kind, [4, 3], sb), {"left": left, "transpose": transpose}))
        one = _real_input if kind == "real" else _complex_input
        out.append(("lx_qdwh", one([3, 3]), {}))
        out.append(("lx_qdwh", one([4, 2]), {}))
        for mode in ["full", "economic", "r"]:
            out.append(("lx_qr_pivot", one([4, 3]), {"mode": mode}))
            out.append(("lx_qr_pivot", one([3, 4]), {"mode": mode}))
        out.append(("lx_qr_pivot", one([2, 4, 3]), {"mode": "economic"}))
        for pivoting in [False, True]:
            out.append(("lx_qr_multiply", two(kind, [4, 3], [2, 4]), {"mode": "right", "conjugate": False, "pivoting": pivoting}))
            out.append(("lx_qr_multiply", two(kind, [4, 3], [3, 2]), {"mode": "left", "conjugate": False, "pivoting": pivoting}))
            out.append(("lx_qr_multiply", two(kind, [4, 3], [2, 4]), {"mode": "right", "conjugate": True, "pivoting": pivoting}))
        out.append(("lx_schur_complex", one([3, 3]), {}))
        out.append(("lx_hessenberg", one([4, 4]), {}))
    for kind in ["real", "complex"]:
        one = _real_input if kind == "real" else _complex_input
        out.append(("lx_sqrtm", one([3, 3]), {}))
        out.append(("lx_expm_frechet", two(kind, [3, 3], [3, 3]), {}))
        out.append(("lx_polar", one([3, 3]), {"side": "right", "method": "qdwh"}))
        out.append(("lx_polar", one([4, 2]), {"side": "right", "method": "qdwh"}))
        out.append(("lx_polar", one([2, 4]), {"side": "left", "method": "qdwh"}))
        out.append(("lx_polar", one([2, 3, 3]), {"side": "right", "method": "qdwh"}))
    out.append(("lx_eigh_tri", two("real", [4], [3]), {}))
    out.append(("lx_rsf2csf", _real_input([4, 4]), {}))
    out.append(("lx_tridiagonal", _complex_input([4, 4]), {}))
    return out


def _surface_batch_cases():
    def two(sa, sb):
        return {"a": {"kind": "real", "shape": sa}, "b": {"kind": "real", "shape": sb}}

    r = _real_input
    out = [
        ("sb_approx_max", r([5]), {"k": 2}),
        ("sb_approx_max", r([3, 4]), {"k": 2}),
        ("sb_approx_min", r([5]), {"k": 2}),
        ("sb_broadcast_like", two([3], [2, 3]), {}),
        ("sb_ragged_dot", two([3, 2], [2, 2, 4]), {"groups": [1, 2]}),
        ("sb_conv_local", two([1, 2, 4, 4], [3, 8, 3, 3]), {"filter": [2, 2]}),
        ("sb_all_to_all", r([3, 3]), {"split": 0, "concat": 0, "tiled": False}),
        ("sb_all_to_all", r([3, 3, 2]), {"split": 0, "concat": 1, "tiled": False}),
        ("sb_all_to_all", r([3, 2, 3]), {"split": 1, "concat": 0, "tiled": False}),
        ("sb_all_to_all", r([2, 4, 3]), {"split": 0, "concat": 1, "tiled": True}),
        ("sb_pswapaxes", r([3, 3, 2]), {"axis": 0}),
        ("sb_psum_scatter", r([3, 3]), {"dim": 0, "tiled": False}),
        ("sb_psum_scatter", r([2, 4]), {"dim": 0, "tiled": True}),
        ("sb_pshuffle", r([3, 2]), {"perm": [1, 2, 0]}),
        ("sb_mgrid_int", r([1]), {}),
        ("sb_mgrid_count", r([1]), {}),
        ("sb_mgrid_one", r([1]), {}),
        ("sb_ogrid", r([1]), {}),
        ("sb_r", two([2], [3]), {}),
        ("sb_c", two([3], [3]), {}),
        ("sb_c", two([2, 2], [2, 1]), {}),
        ("sb_ufunc_call", two([2, 1], [3]), {}),
        ("sb_ufunc_reduce", r([4]), {"axis": 0}),
        ("sb_ufunc_reduce", r([2, 3]), {"axis": 1}),
        ("sb_ufunc_accumulate", r([2, 3]), {"axis": 1}),
        ("sb_ufunc_outer", two([2], [3]), {}),
    ]
    for dt in ["uint8", "uint16", "uint32"]:
        for algo in ["RNG_DEFAULT", "RNG_THREE_FRY", "RNG_PHILOX"]:
            out.append(("sb_rng", r([1]), {"shape": [2, 3], "dtype": dt, "algorithm": algo}))
    return out


def _tracer_batch_cases():
    def two(sa, sb):
        return {"a": {"kind": "real", "shape": sa}, "b": {"kind": "real", "shape": sb}}

    ints = lambda shape: {"a": {"kind": "int", "shape": shape}}
    r = _real_input
    return [
        ("tb_unique", ints([8]), {"size": 8}),
        ("tb_unique", ints([8]), {"size": 3, "fill": -1}),
        ("tb_unique", ints([8]), {"size": 10, "fill": 7}),
        ("tb_unique", ints([6, 2]), {"size": 6, "axis": 0}),
        ("tb_lexsort", ints([6, 2]), {}),
        ("tb_poly_matrix", r([3, 3]), {}),
        ("tb_qr_raw", r([3, 2]), {}),
        ("tb_qr_raw", r([2, 3, 3]), {}),
        ("tb_vectorize", two([2, 1], [3]), {}),
        ("tb_vectorize_core", r([2, 3]), {}),
        ("tb_pad_vmap", two([2, 3], [2]), {}),
        ("tb_seed", r([3]), {}),
        ("tb_lax_conv_c", {"a": {"kind": "complex", "shape": [1, 1, 4, 4]}, "b": {"kind": "complex", "shape": [1, 1, 2, 2]}}, {}),
        ("tb_grad_conv_c", {"a": {"kind": "complex", "shape": [4]}, "b": {"kind": "complex", "shape": [3]}}, {}),
    ] + [
        (op, {"a": {"kind": "complex", "shape": sa}, "b": {"kind": "complex", "shape": sb}}, {"mode": m})
        for op in ["tb_convolve_c", "tb_correlate_c"]
        for m in ["full", "same", "valid"]
        for sa, sb in [([5], [3]), ([3], [5])]
    ]


def _dtype_sweep_cases():
    out = []
    sample = jnp.ones((3, 3), dtype=jnp.complex64) * (1.0 + 0.5j)
    for op, fn in _dtype_sweep_ops().items():
        try:
            fn(sample, {})
        except (TypeError, NotImplementedError, ValueError):
            continue
        out.append((op, {"a": {"kind": "complex", "shape": [3, 3]}}, {}))
    return out


def _extended_sweep_cases():
    out = []
    base = _dtype_sweep_ops()
    for dt in ["int8", "uint8", "int16", "uint16", "float16", "bool"]:
        spec = {"a": {"kind": "typed", "dtype": dt, "shape": [3, 3]}}
        sample = jnp.ones((3, 3), dtype=dt)
        for op, fn in base.items():
            try:
                fn(sample, {})
            except (TypeError, NotImplementedError, ValueError):
                continue
            out.append((op, spec, {}))
    transforms = _transform_sweep_ops()
    real3 = jnp.ones((3, 3), dtype=jnp.float32) * jnp.arange(1.0, 4.0) + jnp.eye(3)
    complex3 = real3 * (1.0 + 0.5j)
    batch = jnp.stack([real3, real3 + 1.0])
    for op, fn in transforms.items():
        kinds = [("real", real3), ("complex", complex3)]
        shape = [3, 3]
        if op.startswith("swv_"):
            kinds = [("real", batch), ("complex", batch * (1.0 + 0.5j))]
            shape = [2, 3, 3]
        for kind, sample in kinds:
            try:
                result = fn(sample, {})
            except Exception:
                continue
            leaves = jax.tree_util.tree_leaves(result)
            if any(np.asarray(x).dtype.kind == "V" for x in leaves):
                continue
            out.append((op, {"a": {"kind": kind, "shape": shape}}, {}))
    return out


def _special_batch_cases():
    r = _real_input
    out = []
    for scale in [0.5, 2.0, 6.0, 12.0, 30.0, 60.0, 100.0]:
        out.append(("sx_expi", r([6]), {"scale": scale}))
    out.append(("sx_expi_grad", _positive_input([4]), {"scale": 1.5}))
    for scale in [0.2, 1.0, 5.0]:
        out.append(("sx_exp1", r([5]), {"scale": scale}))
        for n in [0.0, 1.0, 2.0, 3.0, 6000.0]:
            out.append(("sx_expn", r([5]), {"n": n, "scale": scale}))
    for n in [1.0, 3.0]:
        out.append(("sx_expn_grad", r([3]), {"n": n, "scale": 0.3}))
    for abc in [[0.5, 1.5, 2.5], [1.0, 2.0, 3.5], [-2.0, 1.5, 3.0], [-3.0, -2.0, 1.5], [1.5, 2.5, 5.0], [0.25, 0.75, 1.0], [2.0, 3.0, 5.0], [2.5, 1.5, 1.5]]:
        for scale in [0.5, 0.95]:
            out.append(("sx_hyp2f1", r([4]), {"abc": abc, "scale": scale}))
        out.append(("sx_hyp2f1_grad", r([3]), {"abc": abc}))
    for l in [0, 1, 2, 3, 5]:
        out.append(("sx_lpmn", r([4]), {"l": l}))
        out.append(("sx_lpmn_values", r([4]), {"l": l, "norm": False}))
        out.append(("sx_lpmn_values", r([4]), {"l": l, "norm": True}))
    out.append(("sx_sph", r([4]), {"n": [0, 1, 2, 3], "m": [0, 1, -1, 2], "n_max": 3}))
    out.append(("sx_sph", r([3]), {"n": [4, 4, 2], "m": [-3, 4, 0], "n_max": 5}))
    return out


def all_cases():
    return _special_batch_cases() + _extended_sweep_cases() + _dtype_sweep_cases() + _tracer_batch_cases() + _surface_batch_cases() + _linalg_batch_cases() + _complex_grad_cases() + _complex_linalg_cases() + _gap_cases() + _array_random_cases() + _default_random_cases() + _lax_extra_cases() + _misc_scipy_cases() + _fft_cases() + _lax_cases() + _numpy_cases() + _scipy_cases() + _optimizer_cases() + _narrow_cases() + _batched_cases()
