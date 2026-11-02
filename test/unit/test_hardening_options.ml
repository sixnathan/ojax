open Ojax

module T3 = struct
  module R = Numpy.Reductions
  module NL = Numpy.Lax_numpy
  module X = Numpy.Extras
  module I = Numpy.Indexing
  module P = Numpy.Polynomial
  module N = Nn.Functions
  module D = Dtype
  module UF = Numpy.Ufuncs

  let register reg =
    let u n f =
      reg ("opt." ^ n) (function [ a ] -> [ f a ] | _ -> failwith "arity")
    in
    let b n f =
      reg ("opt." ^ n) (function
        | [ a; c ] -> [ f a c ]
        | _ -> failwith "arity")
    in
    let c n f =
      reg ("opt." ^ n) (function
        | [ a; c; d ] -> [ f a c d ]
        | _ -> failwith "arity")
    in
    let u0 n f =
      reg ("opt." ^ n) (function [] -> f () | _ -> failwith "arity")
    in
    let ax i = [| i |] in
    let sp name
        (f :
          ?dtype:D.t ->
          ?axis:int array ->
          ?keepdims:bool ->
          ?initial:Types.value ->
          ?where:Types.value ->
          Types.value ->
          Types.value) =
      b (name ^ ".where") (fun a w -> f ~where:w a);
      b (name ^ ".where_axis") (fun a w ->
          f ~axis:(ax 1) ~where:w ~keepdims:true a);
      b (name ^ ".initial") (fun a i -> f ~axis:(ax 0) ~initial:i a);
      u (name ^ ".dtype") (fun a -> f ~axis:(ax 1) ~dtype:D.I32 a);
      u (name ^ ".c64") (fun a -> f ~dtype:D.Complex64 a)
    in
    sp "sum" (fun ?dtype ?axis ?keepdims ?initial ?where a ->
        R.sum ?dtype ?axis ?keepdims ?initial ?where a);
    sp "prod" (fun ?dtype ?axis ?keepdims ?initial ?where a ->
        R.prod ?dtype ?axis ?keepdims ?initial ?where a);
    sp "nansum" R.nansum;
    sp "nanprod" R.nanprod;
    u "sum.nopromote" (fun a -> R.sum ~promote_integers:false a);
    u "sum.promote" (fun a -> R.sum a);
    u "prod.nopromote" (fun a -> R.prod ~promote_integers:false a);
    u "sum.bool_dtype" (fun a -> R.sum ~axis:(ax 0) ~dtype:D.Bool a);
    u "prod.bool_dtype" (fun a -> R.prod ~axis:(ax 0) ~dtype:D.Bool a);
    u "sum.f16" (fun a -> R.sum a);
    u "prod.f16" (fun a -> R.prod a);
    u "mean.f16" (fun a -> R.mean a);
    u "var.f16" (fun a -> R.var a);
    u "std.f16" (fun a -> R.std a);
    let mm name
        (f :
          ?axis:int array ->
          ?keepdims:bool ->
          ?initial:Types.value ->
          ?where:Types.value ->
          Types.value ->
          Types.value) =
      c (name ^ ".where_initial") (fun a w i -> f ~where:w ~initial:i a);
      c (name ^ ".where_initial_axis") (fun a w i ->
          f ~axis:(ax 0) ~where:w ~initial:i a);
      b (name ^ ".initial") (fun a i ->
          f ~axis:(ax 1) ~initial:i ~keepdims:true a);
      b (name ^ ".where_err") (fun a w -> f ~where:w a);
      b (name ^ ".int_where") (fun a w ->
          f ~axis:(ax 1) ~where:w
            ~initial:(Types.Concrete (Ndarray.of_floats D.I32 [||] [| 0. |]))
            a)
    in
    mm "max" R.max;
    mm "min" R.min;
    mm "nanmax" R.nanmax;
    mm "nanmin" R.nanmin;
    u "max.empty" (fun a -> R.max a);
    b "all.where" (fun a w -> R.all ~axis:(ax 1) ~where:w a);
    b "all.where2" (fun a w -> R.all ~where:w a);
    b "any.where" (fun a w -> R.any ~axis:(ax 1) ~where:w a);
    b "any.where2" (fun a w -> R.any ~where:w a);
    let me name
        (f :
          ?dtype:D.t ->
          ?axis:int array ->
          ?keepdims:bool ->
          ?where:Types.value ->
          Types.value ->
          Types.value) =
      b (name ^ ".where") (fun a w -> f ~where:w a);
      b (name ^ ".where_axis") (fun a w ->
          f ~axis:(ax 0) ~where:w ~keepdims:true a);
      u (name ^ ".dtype") (fun a -> f ~dtype:D.F16 a);
      u (name ^ ".int_dtype") (fun a -> f ~dtype:D.F16 a)
    in
    me "mean" R.mean;
    me "nanmean" R.nanmean;
    let va name
        (f :
          ?dtype:D.t ->
          ?axis:int array ->
          ?ddof:int ->
          ?where:Types.value ->
          ?mean:Types.value ->
          Types.value ->
          Types.value) =
      b (name ^ ".where") (fun a w -> f ~where:w a);
      b (name ^ ".where_axis") (fun a w -> f ~axis:(ax 1) ~where:w ~ddof:1 a);
      u (name ^ ".dtype") (fun a -> f ~dtype:D.F16 a);
      b (name ^ ".mean") (fun a m -> f ~axis:(ax 0) ~mean:m a);
      u (name ^ ".complex") (fun a -> f a);
      u (name ^ ".int") (fun a -> f a);
      u (name ^ ".over") (fun a -> f ~ddof:10 a)
    in
    va "var" (fun ?dtype ?axis ?ddof ?where ?mean a ->
        R.var ?dtype ?axis ?ddof ?where ?mean a);
    va "std" (fun ?dtype ?axis ?ddof ?where ?mean a ->
        R.std ?dtype ?axis ?ddof ?where ?mean a);
    va "nanvar" (fun ?dtype ?axis ?ddof ?where ?mean a ->
        R.nanvar ?dtype ?axis ?ddof ?where ?mean a);
    va "nanstd" (fun ?dtype ?axis ?ddof ?where ?mean a ->
        R.nanstd ?dtype ?axis ?ddof ?where ?mean a);
    u "var.correction" (fun a -> R.var ~correction:1.5 a);
    u "std.correction" (fun a -> R.std ~correction:1.5 a);
    u "cumsum.dtype" (fun a -> R.cumsum ~axis:1 ~dtype:D.I32 a);
    u "cumprod.dtype" (fun a -> R.cumprod ~dtype:D.F16 a);
    u "nancumsum.dtype" (fun a -> R.nancumsum ~dtype:D.F16 a);
    u "nancumprod.dtype" (fun a -> R.nancumprod ~axis:0 ~dtype:D.F16 a);
    u "cumsum.bool" (fun a -> R.cumsum ~axis:0 ~dtype:D.Bool a);
    u "cumprod.bool" (fun a -> R.cumprod ~axis:0 ~dtype:D.Bool a);
    u "cumulative_sum.dtype" (fun a ->
        R.cumulative_sum ~axis:1 ~dtype:D.F32 ~include_initial:true a);
    u "cumulative_prod.dtype" (fun a ->
        R.cumulative_prod ~axis:1 ~dtype:D.F32 a);
    u "cumulative_prod.int8" (fun a -> R.cumulative_prod a);
    u "cumulative_sum.int8" (fun a -> R.cumulative_sum a);
    reg "opt.average.returned" (function
      | [ a ] ->
          let x, y = R.average_returned ~axis:(ax 1) a in
          [ x; y ]
      | _ -> failwith "a");
    reg "opt.average.returned_w" (function
      | [ a; w ] ->
          let x, y = R.average_returned ~axis:(ax 0) ~weights:w a in
          [ x; y ]
      | _ -> failwith "a");
    b "average.w_axis1" (fun a w -> R.average ~axis:(ax 1) ~weights:w a);
    b "diff.prepend" (fun a p -> NL.diff ~prepend:p a);
    b "diff.append" (fun a p -> NL.diff ~append:p ~n:2 a);
    c "diff.both" (fun a p q -> NL.diff ~axis:0 ~prepend:p ~append:q a);
    b "diff.int_prep_float" (fun a p -> NL.diff ~prepend:p a);
    c "ediff1d.ends" (fun a p q -> NL.ediff1d ~to_end:p ~to_begin:q a);
    c "select.default" (fun c x d -> NL.select ~default:d [ c ] [ x ]);
    c "select.default_int" (fun c x d ->
        NL.select ~default:d
          [ c; UF.logical_not c ]
          [
            x;
            UF.multiply x
              ( UF.add x (UF.subtract x x) |> fun _ ->
                Types.Concrete (Ndarray.of_floats D.I32 [||] [| 2. |]) );
          ]);
    b "concatenate.dtype" (fun a c ->
        NL.concatenate ~axis:1 ~dtype:D.F32 [ a; c ]);
    b "stack.dtype" (fun a c -> NL.stack ~dtype:D.I8 [ a; c ]);
    b "vstack.dtype" (fun a c -> NL.vstack ~dtype:D.F16 [ a; c ]);
    b "hstack.dtype" (fun a c -> NL.hstack ~dtype:D.I32 [ a; c ]);
    b "dstack.dtype" (fun a c -> NL.dstack ~dtype:D.I32 [ a; c ]);
    u "reshape.F" (fun a -> NL.reshape ~order:"F" a [| 3; 2 |]);
    u "ravel.F" (fun a -> NL.ravel ~order:"F" a);
    u "reshape.A" (fun a -> NL.reshape ~order:"A" a [| 3; 2 |]);
    u "reshape.neg" (fun a -> NL.reshape ~order:"F" a [| -1; 2 |]);
    u "repeat.total_short" (fun a ->
        NL.repeat ~axis:1 ~total_repeat_length:4 a 2);
    u "repeat.total_long" (fun a ->
        NL.repeat ~axis:0 ~total_repeat_length:6 a 2);
    u "repeat.total_zero" (fun a -> NL.repeat ~total_repeat_length:0 a 2);
    u "repeat.total_flat" (fun a -> NL.repeat ~total_repeat_length:8 a 2);
    c "searchsorted.sorter" (fun a v s -> NL.searchsorted ~sorter:s a v);
    List.iter
      (fun m ->
        b ("searchsorted." ^ m) (fun a v ->
            NL.searchsorted ~side:"right" ~method_:m a v);
        b ("digitize." ^ m) (fun v a -> NL.digitize ~method_:m v a))
      [ "scan"; "scan_unrolled"; "sort"; "compare_all" ];
    b "searchsorted.bad" (fun a v -> NL.searchsorted ~method_:"bad" a v);
    b "cov.f" (fun m f -> NL.cov ~fweights:f m);
    b "cov.a" (fun m f -> NL.cov ~aweights:f m);
    c "cov.fa" (fun m f a -> NL.cov ~fweights:f ~aweights:a m);
    c "cov.ddof0" (fun m f a -> NL.cov ~ddof:0 ~fweights:f ~aweights:a m);
    b "cov.bias" (fun m f -> NL.cov ~bias:true ~fweights:f m);
    b "cov.float_f" (fun m f -> NL.cov ~fweights:f m);
    b "cov.complex" (fun m f -> NL.cov ~fweights:f m);
    List.iter
      (fun m ->
        u ("pad." ^ m) (fun a -> NL.pad ~mode:m a [| (1, 2); (3, 1) |] 0.0);
        u ("pad1." ^ m) (fun a -> NL.pad ~mode:m a [| (5, 7) |] 0.0))
      [
        "edge";
        "wrap";
        "symmetric";
        "reflect";
        "maximum";
        "minimum";
        "mean";
        "median";
        "linear_ramp";
      ];
    List.iter
      (fun m ->
        u ("pad.stat." ^ m) (fun a ->
            NL.pad ~mode:m
              ~stat_length:[| (1, 1); (2, 1) |]
              a
              [| (1, 2); (3, 1) |]
              0.0);
        u ("pad1.stat." ^ m) (fun a ->
            NL.pad ~mode:m ~stat_length:[| (2, 3) |] a [| (2, 3) |] 0.0))
      [ "maximum"; "minimum"; "mean"; "median" ];
    List.iter
      (fun m ->
        u ("pad.odd." ^ m) (fun a ->
            NL.pad ~mode:m ~reflect_type:"odd" a [| (1, 2); (5, 1) |] 0.0);
        u ("pad1.odd." ^ m) (fun a ->
            NL.pad ~mode:m ~reflect_type:"odd" a [| (5, 7) |] 0.0);
        u ("pad1.single." ^ m) (fun a ->
            NL.pad ~mode:m ~reflect_type:"odd" a [| (2, 3) |] 0.0);
        u ("pad1.single_even." ^ m) (fun a -> NL.pad ~mode:m a [| (2, 3) |] 0.0))
      [ "symmetric"; "reflect" ];
    u "pad.ramp_end" (fun a ->
        NL.pad ~mode:"linear_ramp"
          ~end_values:[| (5., -1.); (2., 3.) |]
          a
          [| (1, 2); (3, 1) |]
          0.0);
    u "pad.bad" (fun a -> NL.pad ~mode:"foo" a [| (1, 2); (3, 1) |] 0.0);
    u "pad.complex_edge" (fun a ->
        NL.pad ~mode:"edge" a [| (1, 1); (1, 1) |] 0.0);
    u "pad.complex_reflect" (fun a ->
        NL.pad ~mode:"reflect" a [| (1, 1); (1, 2) |] 0.0);
    u "pad.bool_edge" (fun a -> NL.pad ~mode:"edge" a [| (1, 1); (1, 1) |] 0.0);
    u "pad1.wrap_big" (fun a -> NL.pad ~mode:"wrap" a [| (10, 10) |] 0.0);
    b "union1d.size" (fun a c -> X.union1d ~size:8 a c);
    b "union1d.size_fill" (fun a c -> X.union1d ~size:8 ~fill_value:(-1.) a c);
    b "union1d.cut" (fun a c -> X.union1d ~size:2 a c);
    b "intersect1d.size" (fun a c -> X.intersect1d ~size:4 a c);
    b "intersect1d.size_fill" (fun a c ->
        X.intersect1d ~size:4 ~fill_value:(-1.) a c);
    b "intersect1d.assume" (fun a c -> X.intersect1d ~assume_unique:true a c);
    b "setdiff1d.size" (fun a c -> X.setdiff1d ~size:5 a c);
    b "setdiff1d.size_fill" (fun a c ->
        X.setdiff1d ~size:5 ~fill_value:(-1.) a c);
    b "setdiff1d.assume" (fun a c -> X.setdiff1d ~assume_unique:true a c);
    b "setdiff1d.cut" (fun a c -> X.setdiff1d ~size:1 a c);
    b "setxor1d.size" (fun a c -> X.setxor1d ~size:6 a c);
    b "setxor1d.fill" (fun a c -> X.setxor1d ~size:6 ~fill_value:(-1.) a c);
    b "setxor1d.assume" (fun a c -> X.setxor1d ~assume_unique:true a c);
    u "bincount.min" (fun a -> X.bincount ~minlength:9 a);
    u "bincount.neg" (fun a -> X.bincount a);
    u "bincount.drop" (fun a -> X.bincount ~length:3 a);
    b "bincount.w" (fun a w -> X.bincount ~weights:w ~minlength:4 a);
    u "bincount.float" (fun a -> X.bincount a);
    u "bincount.bool" (fun a -> X.bincount a);
    b "compress.size" (fun c a -> X.compress ~axis:1 ~size:2 c a);
    b "compress.fill" (fun c a -> X.compress ~axis:0 ~size:2 ~fill_value:9. c a);
    b "compress.short" (fun c a -> X.compress ~axis:1 c a);
    b "compress.none" (fun c a -> X.compress c a);
    b "compress.oob" (fun c a -> X.compress ~axis:0 c a);
    b "extract.size" (fun c a -> X.extract ~size:4 ~fill_value:(-1.) c a);
    let triu v k = NL.triu ~k v in
    u0 "mask_indices.size" (fun () ->
        let x, y = X.mask_indices ~size:8 3 triu in
        [ x; y ]);
    u0 "mask_indices.k" (fun () ->
        let x, y = X.mask_indices ~k:1 ~size:2 3 triu in
        [ x; y ]);
    u "trim_zeros.axis" (fun a -> X.trim_zeros ~axis:[ 0 ] a);
    u "trim_zeros.all" (fun a -> X.trim_zeros a);
    u "trim_zeros.b" (fun a -> X.trim_zeros ~trim:"b" a);
    u "trim_zeros.empty" (fun a -> X.trim_zeros a);
    b "polymul.trim" (fun a c -> P.polymul ~trim_leading_zeros:true a c);
    b "polymul.notrim" (fun a c -> P.polymul a c);
    reg "opt.polydiv.trim" (function
      | [ a; c ] ->
          let x, y = X.polydiv ~trim_leading_zeros:true a c in
          [ x; y ]
      | _ -> failwith "a");
    c "polyfit.w" (fun x y w -> X.polyfit ~w x y 1);
    b "take.default" (fun a i -> I.take a i);
    b "take.fill" (fun a i -> I.take ~mode:"fill" a i);
    b "take.clip" (fun a i -> I.take ~mode:"clip" a i);
    b "take.wrap" (fun a i -> I.take ~mode:"wrap" a i);
    b "take.fv" (fun a i -> I.take ~fill_value:9. a i);
    b "take.int" (fun a i -> I.take a i);
    b "take.u8" (fun a i -> I.take a i);
    b "take.bool" (fun a i -> I.take a i);
    b "take.c" (fun a i -> I.take ~axis:1 a i);
    b "take.axis" (fun a i -> I.take ~axis:1 ~mode:"clip" a i);
    b "take_along.default" (fun a i -> I.take_along_axis ~axis:0 a i);
    b "take_along.clip" (fun a i -> I.take_along_axis ~axis:0 ~mode:"clip" a i);
    b "take_along.fv" (fun a i -> I.take_along_axis ~axis:0 ~fill_value:7. a i);
    b "take_along.2d" (fun a i -> I.take_along_axis ~axis:1 a i);
    c "put_along.default" (fun a i v ->
        I.put_along_axis ~axis:0 ~inplace:false a i v);
    c "put_along.clip" (fun a i v ->
        I.put_along_axis ~axis:0 ~mode:"clip" ~inplace:false a i v);
    b "softmax.where" (fun a w -> N.softmax ~where:w a);
    b "log_softmax.where" (fun a w -> N.log_softmax ~where:w a);
    b "softmax.where0" (fun a w -> N.softmax ~axis:0 ~where:w a);
    b "standardize.where" (fun a w -> N.standardize ~where:w a);
    b "standardize.mean" (fun a m -> N.standardize ~mean:m a);
    b "standardize.var" (fun a v -> N.standardize ~variance:v a);
    u "standardize.stable" (fun a -> N.standardize ~algorithm:"stable" a);
    u "standardize.eps" (fun a -> N.standardize ~epsilon:0.5 a);
    u "standardize.bad" (fun a -> N.standardize ~algorithm:"bad" a);
    b "logmeanexp.where" (fun a w -> N.logmeanexp ~where:w a);
    b "logmeanexp.where_axis" (fun a w ->
        N.logmeanexp ~axis:(ax 1) ~where:w ~keepdims:true a);
    u "one_hot.int" (fun a -> N.one_hot ~dtype:D.I8 ~num_classes:4 a);
    u "one_hot.bool" (fun a -> N.one_hot ~dtype:D.Bool ~axis:0 ~num_classes:4 a)
end

module T4 = struct
  module LL = Lax.Linalg
  module SL = Scipy.Linalg
  module LN = Numpy.Linalg

  let tup4 (a, b, c, d) = [ a; b; c; d ]

  let register reg =
    let u n f = reg n (function [ a ] -> f a | _ -> failwith "arity") in
    let b n f = reg n (function [ a; c ] -> f a c | _ -> failwith "arity") in
    let q n f =
      reg n (function [ a; c; d; e ] -> [ f a c d e ] | _ -> failwith "arity")
    in
    b "tri.vec" (fun a c -> SL.eigh_tridiagonal a c);
    b "tri.val" (fun a c -> SL.eigh_tridiagonal ~eigvals_only:true a c);
    b "tri.sel" (fun a c ->
        SL.eigh_tridiagonal ~eigvals_only:true ~select:"i"
          ~select_range:(1., 2.) a c);
    b "tri.selv" (fun a c ->
        SL.eigh_tridiagonal ~select:"i" ~select_range:(1., 2.) a c);
    b "tri.tol" (fun a c -> SL.eigh_tridiagonal ~eigvals_only:true ~tol:0.1 a c);
    b "tri.key" (fun a c ->
        SL.eigh_tridiagonal
          ~key:
            (Random.Core.key
               (Types.Concrete (Ndarray.of_floats Dtype.I32 [||] [| 7. |])))
          a c);
    b "tri.cluster" (fun a c -> SL.eigh_tridiagonal a c);
    b "tri.one" (fun a c -> SL.eigh_tridiagonal a c);
    b "tri.complex" (fun a c -> SL.eigh_tridiagonal a c);
    b "tri.bad_dt" (fun a c -> SL.eigh_tridiagonal a c);
    b "tri.int" (fun a c -> SL.eigh_tridiagonal a c);
    q "ts.real" (fun a c d e -> LL.tridiagonal_solve a c d e);
    q "ts.perturb" (fun a c d e ->
        LL.tridiagonal_solve ~perturb_singular:true a c d e);
    q "ts.singular" (fun a c d e -> LL.tridiagonal_solve a c d e);
    q "ts.complex" (fun a c d e -> LL.tridiagonal_solve a c d e);
    q "ts.complex_perturb" (fun a c d e ->
        LL.tridiagonal_solve ~perturb_singular:true a c d e);
    u "ll.chol" (fun a -> [ LL.cholesky a ]);
    u "ll.chol_ns" (fun a -> [ LL.cholesky ~symmetrize_input:false a ]);
    u "ll.eigh" (fun a ->
        let v, w = LL.eigh a in
        [ v; w ]);
    u "ll.eigh_ns" (fun a ->
        let v, w = LL.eigh ~symmetrize_input:false a in
        [ v; w ]);
    u "ll.eigh_nsu" (fun a ->
        let v, w = LL.eigh ~symmetrize_input:false ~lower:false a in
        [ v; w ]);
    u "np.chol" (fun a -> [ LN.cholesky a ]);
    u "np.chol_ns" (fun a -> [ LN.cholesky ~symmetrize_input:false a ]);
    u "np.chol_up" (fun a -> [ LN.cholesky ~upper:true a ]);
    u "sp.chol" (fun a -> [ SL.cholesky a ]);
    u "sp.chol_l" (fun a -> [ SL.cholesky ~lower:true a ]);
    u "sp.eigh" (fun a -> SL.eigh a);
    b "np.pinv_rcond" (fun a r -> [ LN.pinv ~rcond:r a ]);
    b "np.rank_tol" (fun a r -> [ LN.matrix_rank ~tol:r a ]);
    b "np.lstsq_res" (fun a c -> tup4 (LN.lstsq ~numpy_resid:true a c));
    b "np.lstsq_res_wide" (fun a c -> tup4 (LN.lstsq ~numpy_resid:true a c));
    b "np.lstsq" (fun a c -> tup4 (LN.lstsq a c));
    b "np.lstsq_neg" (fun a c -> tup4 (LN.lstsq ~rcond:(-1.) a c))
end

module T5 = struct
  module SL = Scipy.Linalg
  module SP = Scipy.Linalg_special
  module S = Scipy.Signal
  module L = Lax_api
  module D = Dtype

  let register reg =
    let u n f = reg n (function [ a ] -> f a | _ -> failwith "arity") in
    let b n f = reg n (function [ a; c ] -> f a c | _ -> failwith "arity") in
    let c3 n f =
      reg n (function [ a; c; d ] -> f a c d | _ -> failwith "arity")
    in
    let u0 n f = reg n (function [] -> f () | _ -> failwith "arity") in
    u "e.expm_max" (fun a -> [ SL.expm ~max_squarings:1 a ]);
    u "e.expm_max0" (fun a ->
        [ SL.expm ~max_squarings:2 (L.mul a (L.full_like a 50.)) ]);
    u "e.expm" (fun a ->
        [ SL.expm ~max_squarings:1 (L.mul a (L.full_like a 0.01)) ]);
    u0 "e.hadamard" (fun () -> [ SP.hadamard 4 ]);
    u0 "e.hadamard_i" (fun () -> [ SP.hadamard ~dtype:D.I8 2 ]);
    u0 "e.dft_d" (fun () -> [ SP.dft ~dtype:D.Complex64 2 ]);
    u0 "e.dft" (fun () -> [ SP.dft ~scale:"n" 3 ]);
    b "e.fftc_ax" (fun a c -> [ S.fftconvolve ~axes:[ 1 ] a c ]);
    b "e.fftc_same" (fun a c -> [ S.fftconvolve ~mode:"same" ~axes:[ 1 ] a c ]);
    b "e.fftc_0" (fun a c -> [ S.fftconvolve ~axes:[ 0 ] a c ]);
    b "e.fftc_bad" (fun a c -> [ S.fftconvolve ~mode:"x" a c ]);
    b "e.conv2d_bd" (fun a c -> [ S.convolve2d ~boundary:"wrap" a c ]);
    b "e.corr2d_fv" (fun a c -> [ S.correlate2d ~fillvalue:1. a c ]);
    b "e.conv2d" (fun a c ->
        [ S.convolve2d ~boundary:"fill" ~fillvalue:0. a c ]);
    u "e.detrend_bp" (fun a -> [ S.detrend ~bp:[ 3 ] a ]);
    u "e.detrend_bp2" (fun a -> [ S.detrend ~bp:[ 2; 5 ] a ]);
    u "e.detrend_ax" (fun a -> [ S.detrend ~axis:0 ~bp:[ 1 ] a ]);
    u "e.detrend_bad" (fun a -> [ S.detrend ~bp:[ 20 ] a ]);
    u "e.istft" (fun a ->
        let x, y =
          S.istft ~nperseg:4 ~noverlap:2 ~boundary:false ~time_axis:0
            ~freq_axis:1 a
        in
        [ x; y ]);
    u "e.istft_std" (fun a ->
        let x, y = S.istft ~nperseg:4 ~noverlap:2 ~boundary:false a in
        [ x; y ]);
    u "e.istft_same" (fun a ->
        let x, y = S.istft ~time_axis:0 ~freq_axis:0 a in
        [ x; y ]);
    b "e.dot_dn" (fun a c ->
        [ L.dot ~dimension_numbers:(([| 1 |], [| 1 |]), ([||], [||])) a c ]);
    b "e.dot_dnb" (fun a c ->
        [
          L.dot ~dimension_numbers:(([| 1 |], [| 1 |]), ([| 0 |], [| 0 |])) a c;
        ]);
    let two r =
      L.full (Core.get_aval r).Types.shape 2.0 (Core.get_aval r).Types.dtype
    in
    u "e.map_bs" (fun a ->
        L.map ~batch_size:2
          (function
            | [ r ] ->
                [ Numpy.Ufuncs.add (L.mul r (two r)) (Numpy.Reductions.sum r) ]
            | _ -> failwith "m")
          [ a ]);
    u "e.map_bs1" (fun a ->
        L.map ~batch_size:5
          (function [ r ] -> [ L.mul r (two r) ] | _ -> failwith "m")
          [ a ]);
    u "e.assoc_rev" (fun a -> [ L.associative_scan ~reverse:true L.add a ]);
    u "e.assoc_rev1" (fun a ->
        [ L.associative_scan ~reverse:true ~axis:1 L.add a ]);
    let dn =
      {
        Types.offset_dims = [||];
        collapsed_slice_dims = [| 0 |];
        start_index_map = [| 0 |];
        g_operand_batching_dims = [||];
        g_start_indices_batching_dims = [||];
        g_sorted_unique = false;
      }
    in
    b "e.gather_d" (fun a i -> [ L.gather a i dn [| 1 |] ]);
    b "e.gather_clip" (fun a i ->
        [ L.gather ~mode:Lax_config.Clip a i dn [| 1 |] ]);
    b "e.gather_fv" (fun a i ->
        [ L.gather ~mode:Lax_config.Fill_or_drop ~fill_value:9. a i dn [| 1 |] ]);
    b "e.gather_int" (fun a i -> [ L.gather a i dn [| 1 |] ]);
    b "e.gather_bad" (fun a i ->
        [
          L.gather
            ~mode:(Lax_config.gather_scatter_mode_of_string "x")
            a i dn [| 1 |];
        ]);
    let sd =
      {
        Types.update_window_dims = [||];
        inserted_window_dims = [| 0 |];
        scatter_dims_to_operand_dims = [| 0 |];
        s_operand_batching_dims = [||];
        s_scatter_indices_batching_dims = [||];
        s_sorted_unique = false;
        s_clip = false;
      }
    in
    let fam n
        (f :
          ?mode:Lax_config.gather_scatter_mode ->
          Types.value ->
          Types.value ->
          Types.value ->
          Types.scatter_dims ->
          Types.value) =
      c3 ("e." ^ n) (fun a i v -> [ f a i v sd ]);
      c3
        ("e." ^ n ^ "_clip")
        (fun a i v -> [ f ~mode:Lax_config.Clip a i v sd ]);
      c3
        ("e." ^ n ^ "_drop")
        (fun a i v -> [ f ~mode:Lax_config.Fill_or_drop a i v sd ])
    in
    fam "scatter" (fun ?mode a i v d -> L.scatter ?mode a i v d);
    fam "scatter_add" (fun ?mode x i u d -> L.scatter_add ?mode x i u d);
    fam "scatter_sub" (fun ?mode x i u d -> L.scatter_sub ?mode x i u d);
    fam "scatter_mul" (fun ?mode a i v d -> L.scatter_mul ?mode a i v d);
    fam "scatter_min" (fun ?mode x i u d -> L.scatter_min ?mode x i u d);
    fam "scatter_max" (fun ?mode x i u d -> L.scatter_max ?mode x i u d)
end

module T6 = struct
  module NL = Numpy.Lax_numpy
  module R = Numpy.Reductions
  module X = Numpy.Extras
  module UF = Numpy.Ufuncs
  module I = Numpy.Indexing
  module S = Numpy.Sorting
  module AC = Numpy.Array_creation
  module TC = Numpy.Tensor_contractions
  module P = Numpy.Polynomial
  module LN = Numpy.Linalg
  module SL = Scipy.Linalg
  module D = Dtype

  let f32s v = Types.Concrete (Ndarray.of_floats D.F32 [| Array.length v |] v)
  let row r a = Lax_api.index_in_dim ~keepdims:false a r

  let register reg =
    let u n f =
      reg ("g." ^ n) (function [ a ] -> [ f a ] | _ -> failwith "arity")
    in
    let ul n f =
      reg ("g." ^ n) (function [ a ] -> f a | _ -> failwith "arity")
    in
    let b n f =
      reg ("g." ^ n) (function [ a; c ] -> [ f a c ] | _ -> failwith "arity")
    in
    List.iter
      (fun (n, f) -> u n f)
      [
        ("sort", fun a -> S.sort a);
        ("argsort", fun a -> S.argsort a);
        ("flip", fun a -> NL.flip a);
        ("nan_to_num", fun a -> NL.nan_to_num a);
        ("round", fun a -> NL.round a);
        ("ravel", fun a -> NL.ravel a);
        ("squeeze", fun a -> NL.squeeze a);
        ("transpose", fun a -> NL.transpose a);
        ("tril", fun a -> NL.tril a);
        ("triu", fun a -> NL.triu a);
        ("diagonal", fun a -> NL.diagonal a);
        ("trace", fun a -> NL.trace a);
        ("sum", fun a -> R.sum a);
        ("prod", fun a -> R.prod a);
        ("mean", fun a -> R.mean a);
        ("var", fun a -> R.var a);
        ("std", fun a -> R.std a);
        ("max", fun a -> R.max a);
        ("min", fun a -> R.min a);
        ("ptp", fun a -> R.ptp a);
        ("all", fun a -> R.all a);
        ("any", fun a -> R.any a);
        ("cumsum", fun a -> R.cumsum a);
        ("cumprod", fun a -> R.cumprod a);
        ("argmax", fun a -> NL.argmax a);
        ("argmin", fun a -> NL.argmin a);
        ("nanmax", fun a -> R.nanmax a);
        ("nanmin", fun a -> R.nanmin a);
        ("nansum", fun a -> R.nansum a);
        ("nanprod", fun a -> R.nanprod a);
        ("nanmean", fun a -> R.nanmean a);
        ("nanvar", fun a -> R.nanvar a);
        ("nanstd", fun a -> R.nanstd a);
        ("nanargmax", fun a -> NL.nanargmax a);
        ("nanargmin", fun a -> NL.nanargmin a);
        ("count_nonzero", fun a -> R.count_nonzero a);
        ("median", fun a -> R.median a);
        ("nanmedian", fun a -> R.nanmedian a);
        ("angle", fun a -> NL.angle a);
        ("iscomplex", NL.iscomplex);
        ("isreal", NL.isreal);
        ("unwrap", fun a -> NL.unwrap a);
        ("diff", fun a -> NL.diff a);
        ("flatnonzero", fun a -> X.flatnonzero a);
        ("argwhere", fun a -> X.argwhere a);
        ("sinc", UF.sinc);
        ("i0", NL.i0);
        ("zeros_like", fun a -> AC.zeros_like a);
        ("ones_like", fun a -> AC.ones_like a);
        ("copy", fun a -> NL.copy a);
        ("real", UF.real);
        ("imag", UF.imag);
        ("conj", UF.conj);
        ("cov", fun a -> NL.cov a);
        ("corrcoef", fun a -> NL.corrcoef a);
        ("average", fun a -> R.average a);
        ("sort_complex", fun a -> X.sort_complex (row 0 a));
        ("expand_dims", fun a -> NL.expand_dims a [| 0 |]);
        ("rot90", fun a -> NL.rot90 a);
        ("trapezoid", fun a -> NL.trapezoid a);
        ("histogram_bin_edges", fun a -> X.histogram_bin_edges ~bins:3 a);
        ("bincount", fun a -> X.bincount (row 0 a));
        ("clip", fun a -> NL.clip ~min:0. ~max:2. a);
        ("clip_lo", fun a -> NL.clip ~min:1. a);
        ("ceil", UF.ceil);
        ("trunc", NL.trunc);
      ];
    ul "nonzero" (fun a -> X.nonzero a);
    ul "unique" (fun a -> X.unique a);
    ul "gradient" (fun a -> X.gradient ~axis:[ 1 ] a);
    let scalar v a =
      Types.Concrete
        (Ndarray.of_floats (Core.get_aval a).Types.dtype [||] [| v |])
    in
    u "quantile" (fun a ->
        R.quantile a (Types.Concrete (Ndarray.of_floats D.F32 [||] [| 0.5 |])));
    u "percentile" (fun a ->
        R.percentile a (Types.Concrete (Ndarray.of_floats D.F32 [||] [| 50. |])));
    ignore scalar;
    let tr = fun a -> NL.transpose a in
    List.iter
      (fun (n, f) -> b n f)
      [
        ("where", fun a c -> NL.where_ (UF.not_equal a (AC.zeros_like a)) a c);
        ("dot", fun a c -> TC.dot a (tr c));
        ("matmul", fun a c -> TC.matmul a (tr c));
        ("vdot", fun a c -> TC.vdot a c);
        ("inner", fun a c -> TC.inner a c);
        ("outer", TC.outer);
        ("kron", NL.kron);
        ("cross", fun a c -> NL.cross a c);
        ("isclose", fun a c -> NL.isclose a c);
        ("allclose", fun a c -> NL.allclose a c);
        ("array_equal", fun a c -> NL.array_equal a c);
        ("convolve", fun a c -> NL.convolve (row 0 a) (row 1 c));
        ("correlate", fun a c -> NL.correlate (row 0 a) (row 1 c));
        ( "interp",
          fun a c -> X.interp (row 0 a) (f32s [| 0.; 1.; 2. |]) (row 0 c) );
        ("polyval", fun a c -> P.polyval (row 0 a) (row 0 c));
        ("polyadd", fun a c -> P.polyadd (row 0 a) (row 1 c));
        ("polymul", fun a c -> P.polymul (row 0 a) (row 1 c));
        ("gcd", NL.gcd);
        ("lcm", NL.lcm);
        ( "ldexp",
          fun a _ ->
            X.ldexp a
              (Types.Concrete
                 (Ndarray.of_floats D.I32 [| 2; 3 |]
                    [| 1.; 2.; 3.; 0.; 5.; 7. |])) );
      ];
    reg "g.frexp" (function
      | [ a; _ ] ->
          let x, y = X.frexp a in
          [ x; y ]
      | _ -> failwith "arity");
    let eye a = NL.eye ~dtype:(Core.get_aval a).Types.dtype 2 in
    List.iter
      (fun (n, f) -> ul ("la." ^ n) f)
      [
        ("inv", fun a -> [ LN.inv a ]);
        ("det", fun a -> [ LN.det a ]);
        ( "slogdet",
          fun a ->
            let x, y = LN.slogdet a in
            [ x; y ] );
        ("norm", fun a -> [ LN.norm a ]);
        ("matrix_rank", fun a -> [ LN.matrix_rank a ]);
        ("pinv", fun a -> [ LN.pinv a ]);
        ("qr", fun a -> LN.qr a);
        ("svdvals", fun a -> [ LN.svdvals a ]);
        ("eigvalsh", fun a -> [ LN.eigvalsh a ]);
        ("eigvals", fun a -> [ LN.eigvals a ]);
        ( "cholesky",
          fun a ->
            if (Core.get_aval a).Types.dtype = D.Bool then [ LN.cholesky a ]
            else
              [
                LN.cholesky
                  (UF.add (TC.matmul a (UF.conj (NL.transpose a))) (eye a));
              ] );
        ("trace", fun a -> [ LN.trace a ]);
        ("matrix_power", fun a -> [ LN.matrix_power a 2 ]);
        ("cond", fun a -> [ LN.cond a ]);
        ("expm", fun a -> [ SL.expm a ]);
        ("sqrtm", fun a -> [ SL.sqrtm a ]);
        ("lu", fun a -> SL.lu a);
        ("det2", fun a -> [ SL.det a ]);
        ("inv2", fun a -> [ SL.inv a ]);
      ];
    b "la.solve" LN.solve;
    b "la.lstsq" (fun a c ->
        let x, _, _, _ = LN.lstsq a c in
        x)
end

let options =
  [
    ( "opt.sum.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||7.5" ] );
    ( "opt.sum.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2,1|4.5,3.0" ] );
    ( "opt.sum.initial",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|3|12.0,12.0,12.0" ] );
    ( "opt.sum.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "int32|2|2.0,3.0" ] );
    ( "opt.sum.c64",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "complex64||6.0|0.0" ] );
    ( "opt.prod.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||-18.0" ] );
    ( "opt.prod.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2,1|4.5,-4.0" ] );
    ( "opt.prod.initial",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|3|7.5,-80.0,-30.0" ] );
    ( "opt.prod.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "int32|2|-6.0,0.0" ] );
    ( "opt.prod.c64",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "complex64||18.0|-0.0" ] );
    ( "opt.nansum.where",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||7.5" ] );
    ( "opt.nansum.where_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2,1|4.5,3.0" ] );
    ( "opt.nansum.initial",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|3|11.5,14.0,12.0" ] );
    ( "opt.nansum.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "int32|2|4.0,3.0" ] );
    ( "opt.nansum.c64",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "complex64||7.5|0.0" ] );
    ( "opt.nanprod.where",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||-18.0" ] );
    ( "opt.nanprod.where_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2,1|4.5,-4.0" ] );
    ( "opt.nanprod.initial",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|3|15.0,40.0,-30.0" ] );
    ( "opt.nanprod.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "int32|2|3.0,-4.0" ] );
    ( "opt.nanprod.c64",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "complex64||-18.0|0.0" ] );
    ( "opt.sum.nopromote",
      false,
      [ "int8|3|100.0,100.0,3.0" ],
      Some [ "int8||-53.0" ] );
    ( "opt.sum.promote",
      false,
      [ "int8|3|100.0,100.0,3.0" ],
      Some [ "int32||203.0" ] );
    ( "opt.prod.nopromote",
      false,
      [ "int8|3|100.0,100.0,3.0" ],
      Some [ "int8||48.0" ] );
    ( "opt.sum.bool_dtype",
      false,
      [ "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool|3|1.0,1.0,1.0" ] );
    ( "opt.prod.bool_dtype",
      false,
      [ "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool|3|1.0,0.0,0.0" ] );
    ( "opt.sum.f16",
      false,
      [ "float16|4|1.5,2.5,100.0,1000.0" ],
      Some [ "float16||1104.0" ] );
    ( "opt.prod.f16",
      false,
      [ "float16|4|1.5,2.5,100.0,1000.0" ],
      Some [ "float16||inf" ] );
    ( "opt.mean.f16",
      false,
      [ "float16|4|1.5,2.5,100.0,1000.0" ],
      Some [ "float16||276.0" ] );
    ( "opt.var.f16",
      false,
      [ "float16|4|1.5,2.5,100.0,1000.0" ],
      Some [ "float16||inf" ] );
    ( "opt.std.f16",
      false,
      [ "float16|4|1.5,2.5,100.0,1000.0" ],
      Some [ "float16||inf" ] );
    ( "opt.max.where_initial",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32||10.0" ] );
    ( "opt.max.where_initial_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32|3|10.0,10.0,10.0" ] );
    ( "opt.max.initial",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|2,1|10.0,10.0" ] );
    ( "opt.max.where_err",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      None );
    ( "opt.max.int_where",
      false,
      [
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "int32|2|3.0,6.0" ] );
    ( "opt.min.where_initial",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32||-1.0" ] );
    ( "opt.min.where_initial_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32|3|1.5,4.0,-1.0" ] );
    ( "opt.min.initial",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|2,1|-2.0,-1.0" ] );
    ( "opt.min.where_err",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      None );
    ( "opt.min.int_where",
      false,
      [
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "int32|2|0.0,0.0" ] );
    ( "opt.nanmax.where_initial",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32||10.0" ] );
    ( "opt.nanmax.where_initial_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32|3|10.0,10.0,10.0" ] );
    ( "opt.nanmax.initial",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|2,1|10.0,10.0" ] );
    ( "opt.nanmax.where_err",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      None );
    ( "opt.nanmax.int_where",
      false,
      [
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "int32|2|3.0,6.0" ] );
    ( "opt.nanmin.where_initial",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32||-1.0" ] );
    ( "opt.nanmin.where_initial_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32||10.0";
      ],
      Some [ "float32|3|1.5,4.0,-1.0" ] );
    ( "opt.nanmin.initial",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32||10.0" ],
      Some [ "float32|2,1|1.5,-1.0" ] );
    ( "opt.nanmin.where_err",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      None );
    ( "opt.nanmin.int_where",
      false,
      [
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "int32|2|0.0,0.0" ] );
    ("opt.max.empty", false, [ "float32|0|" ], None);
    ( "opt.all.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "bool|2|1.0,1.0" ] );
    ( "opt.all.where2",
      false,
      [ "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool||0.0" ] );
    ( "opt.any.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "bool|2|1.0,1.0" ] );
    ( "opt.any.where2",
      false,
      [ "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0"; "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool||1.0" ] );
    ( "opt.mean.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||1.875" ] );
    ( "opt.mean.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|1,3|1.5,4.0,1.0" ] );
    ( "opt.mean.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float16||1.0" ] );
    ( "opt.mean.int_dtype",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float16||3.5" ] );
    ( "opt.nanmean.where",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||1.875" ] );
    ( "opt.nanmean.where_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|1,3|1.5,4.0,1.0" ] );
    ( "opt.nanmean.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float16||1.875" ] );
    ( "opt.nanmean.int_dtype",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float16||3.5" ] );
    ( "opt.var.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||3.546875" ] );
    ( "opt.var.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2|1.125,12.5" ] );
    ( "opt.var.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float16||4.41796875" ] );
    ( "opt.var.mean",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32|1,3|1.0,1.0,1.0" ],
      Some [ "float32|3|0.25,9.0,4.0" ] );
    ( "opt.var.complex",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "float32||1.6736111640930176" ] );
    ( "opt.var.int",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32||2.9166667461395264" ] );
    ( "opt.var.over",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32||nan" ] );
    ( "opt.std.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||1.883314847946167" ] );
    ( "opt.std.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2|1.0606601238250732,3.535533905029297" ] );
    ( "opt.std.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float16||2.1015625" ] );
    ( "opt.std.mean",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32|1,3|1.0,1.0,1.0" ],
      Some [ "float32|3|0.5,3.0,2.0" ] );
    ( "opt.std.complex",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "float32||1.293681263923645" ] );
    ( "opt.std.int",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32||1.7078251838684082" ] );
    ( "opt.std.over",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32||nan" ] );
    ( "opt.nanvar.where",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||3.546875" ] );
    ( "opt.nanvar.where_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2|1.125,12.5" ] );
    ( "opt.nanvar.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float16||3.546875" ] );
    ( "opt.nanvar.mean",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32|1,3|1.0,1.0,1.0" ],
      Some [ "float32|3|0.25,9.0,4.0" ] );
    ( "opt.nanvar.complex",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "float32||1.6736111640930176" ] );
    ( "opt.nanvar.int",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32||2.9166667461395264" ] );
    ( "opt.nanvar.over",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float32||nan" ] );
    ( "opt.nanstd.where",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||1.883314847946167" ] );
    ( "opt.nanstd.where_axis",
      false,
      [
        "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2|1.0606601238250732,3.535533905029297" ] );
    ( "opt.nanstd.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float16||1.8837890625" ] );
    ( "opt.nanstd.mean",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0"; "float32|1,3|1.0,1.0,1.0" ],
      Some [ "float32|3|0.5,3.0,2.0" ] );
    ( "opt.nanstd.complex",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "float32||1.293681263923645" ] );
    ( "opt.nanstd.int",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32||1.7078251838684082" ] );
    ( "opt.nanstd.over",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float32||nan" ] );
    ( "opt.var.correction",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32||5.888888835906982" ] );
    ( "opt.std.correction",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32||2.4267032146453857" ] );
    ( "opt.cumsum.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "int32|2,3|1.0,-1.0,2.0,0.0,4.0,3.0" ] );
    ( "opt.cumprod.dtype",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float16|6|1.5,-3.0,-9.0,-4.5,-18.0,18.0" ] );
    ( "opt.nancumsum.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float16|6|1.5,1.5,4.5,4.5,8.5,7.5" ] );
    ( "opt.nancumprod.dtype",
      false,
      [ "float32|2,3|1.5,nan,3.0,nan,4.0,-1.0" ],
      Some [ "float16|2,3|1.5,1.0,3.0,1.5,4.0,-3.0" ] );
    ( "opt.cumsum.bool",
      false,
      [ "int32|2,3|-2.0,-1.0,0.0,1.0,2.0,3.0" ],
      Some [ "bool|2,3|1.0,1.0,0.0,1.0,1.0,1.0" ] );
    ( "opt.cumprod.bool",
      false,
      [ "int32|2,3|-2.0,-1.0,0.0,1.0,2.0,3.0" ],
      Some [ "bool|2,3|1.0,1.0,0.0,1.0,1.0,0.0" ] );
    ( "opt.cumulative_sum.dtype",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32|2,4|0.0,1.0,3.0,6.0,0.0,4.0,9.0,15.0" ] );
    ( "opt.cumulative_prod.dtype",
      false,
      [ "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0" ],
      Some [ "float32|2,3|1.0,2.0,6.0,4.0,20.0,120.0" ] );
    ( "opt.cumulative_prod.int8",
      false,
      [ "int8|3|100.0,100.0,3.0" ],
      Some [ "int8|3|100.0,16.0,48.0" ] );
    ( "opt.cumulative_sum.int8",
      false,
      [ "int8|3|100.0,100.0,3.0" ],
      Some [ "int32|3|100.0,200.0,203.0" ] );
    ( "opt.average.returned",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some
        [
          "float32|2|0.8333333730697632,1.1666667461395264"; "float32|2|3.0,3.0";
        ] );
    ( "opt.average.returned_w",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32|2|1.0,3.0" ],
      Some [ "float32|3|0.75,2.5,0.0"; "float32|3|4.0,4.0,4.0" ] );
    ( "opt.average.w_axis1",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32|3|1.0,3.0,2.0" ],
      Some [ "float32|2|0.25,1.75" ] );
    ( "opt.diff.prepend",
      false,
      [ "float32|5|1.0,2.0,4.0,7.0,0.0"; "float32||9.0" ],
      Some [ "float32|5|-8.0,1.0,2.0,3.0,-7.0" ] );
    ( "opt.diff.append",
      false,
      [ "float32|5|1.0,2.0,4.0,7.0,0.0"; "float32||9.0" ],
      Some [ "float32|4|1.0,1.0,-10.0,16.0" ] );
    ( "opt.diff.both",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32|1,3|1.5,-2.0,3.0";
        "float32|1,3|2.5,-1.0,4.0";
      ],
      Some [ "float32|3,3|0.0,0.0,0.0,-1.0,6.0,-4.0,2.0,-5.0,5.0" ] );
    ( "opt.diff.int_prep_float",
      false,
      [ "int32|3|1.0,2.0,4.0"; "float32||0.5" ],
      Some [ "float32|3|0.5,1.0,2.0" ] );
    ( "opt.ediff1d.ends",
      false,
      [ "float32|5|1.0,2.0,4.0,7.0,0.0"; "float32|2|5.0,6.0"; "float32||9.0" ],
      Some [ "float32|7|9.0,1.0,2.0,3.0,-7.0,5.0,6.0" ] );
    ( "opt.select.default",
      false,
      [
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32||7.0";
      ],
      Some [ "float32|2,3|1.5,7.0,3.0,7.0,4.0,-1.0" ] );
    ( "opt.select.default_int",
      false,
      [
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0";
        "float32||7.5";
      ],
      Some [ "float32|2,3|1.0,4.0,3.0,8.0,5.0,6.0" ] );
    ( "opt.concatenate.dtype",
      false,
      [
        "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0"; "int32|2,3|1.0,2.0,3.0,4.0,5.0,6.0";
      ],
      Some [ "float32|2,6|1.0,2.0,3.0,1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,6.0" ] );
    ( "opt.stack.dtype",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
      ],
      Some [ "int8|2,2,3|1.0,-2.0,3.0,0.0,4.0,-1.0,1.0,-2.0,3.0,0.0,4.0,-1.0" ]
    );
    ( "opt.vstack.dtype",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
      ],
      Some [ "float16|4,3|1.5,-2.0,3.0,0.5,4.0,-1.0,1.5,-2.0,3.0,0.5,4.0,-1.0" ]
    );
    ( "opt.hstack.dtype",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
      ],
      Some [ "int32|2,6|1.0,-2.0,3.0,1.0,-2.0,3.0,0.0,4.0,-1.0,0.0,4.0,-1.0" ]
    );
    ( "opt.dstack.dtype",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
      ],
      Some [ "int32|2,3,2|1.0,1.0,-2.0,-2.0,3.0,3.0,0.0,0.0,4.0,4.0,-1.0,-1.0" ]
    );
    ( "opt.reshape.F",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|3,2|1.5,4.0,0.5,3.0,-2.0,-1.0" ] );
    ( "opt.ravel.F",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|6|1.5,0.5,-2.0,4.0,3.0,-1.0" ] );
    ("opt.reshape.A", false, [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ], None);
    ( "opt.reshape.neg",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|3,2|1.5,4.0,0.5,3.0,-2.0,-1.0" ] );
    ( "opt.repeat.total_short",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|2,4|1.5,1.5,-2.0,-2.0,0.5,0.5,4.0,4.0" ] );
    ( "opt.repeat.total_long",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some
        [
          "float32|6,3|1.5,-2.0,3.0,1.5,-2.0,3.0,0.5,4.0,-1.0,0.5,4.0,-1.0,0.5,4.0,-1.0,0.5,4.0,-1.0";
        ] );
    ( "opt.repeat.total_zero",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|0|" ] );
    ( "opt.repeat.total_flat",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|8|1.5,1.5,-2.0,-2.0,3.0,3.0,0.5,0.5" ] );
    ( "opt.searchsorted.sorter",
      false,
      [ "float32|3|3.0,1.0,2.0"; "float32|2|1.5,2.5"; "int32|3|1.0,2.0,0.0" ],
      Some [ "int32|2|1.0,2.0" ] );
    ( "opt.searchsorted.scan",
      false,
      [ "float32|4|1.0,2.0,2.0,3.0"; "float32|3|0.0,2.0,5.0" ],
      Some [ "int32|3|0.0,3.0,4.0" ] );
    ( "opt.digitize.scan",
      false,
      [ "float32|3|0.0,2.0,5.0"; "float32|3|1.0,2.0,3.0" ],
      Some [ "int32|3|0.0,2.0,3.0" ] );
    ( "opt.searchsorted.scan_unrolled",
      false,
      [ "float32|4|1.0,2.0,2.0,3.0"; "float32|3|0.0,2.0,5.0" ],
      Some [ "int32|3|0.0,3.0,4.0" ] );
    ( "opt.digitize.scan_unrolled",
      false,
      [ "float32|3|0.0,2.0,5.0"; "float32|3|1.0,2.0,3.0" ],
      Some [ "int32|3|0.0,2.0,3.0" ] );
    ( "opt.searchsorted.sort",
      false,
      [ "float32|4|1.0,2.0,2.0,3.0"; "float32|3|0.0,2.0,5.0" ],
      Some [ "int32|3|0.0,3.0,4.0" ] );
    ( "opt.digitize.sort",
      false,
      [ "float32|3|0.0,2.0,5.0"; "float32|3|1.0,2.0,3.0" ],
      Some [ "int32|3|0.0,2.0,3.0" ] );
    ( "opt.searchsorted.compare_all",
      false,
      [ "float32|4|1.0,2.0,2.0,3.0"; "float32|3|0.0,2.0,5.0" ],
      Some [ "int32|3|0.0,3.0,4.0" ] );
    ( "opt.digitize.compare_all",
      false,
      [ "float32|3|0.0,2.0,5.0"; "float32|3|1.0,2.0,3.0" ],
      Some [ "int32|3|0.0,2.0,3.0" ] );
    ( "opt.searchsorted.bad",
      false,
      [ "float32|2|1.0,2.0"; "float32|1|0.0" ],
      None );
    ( "opt.cov.f",
      false,
      [ "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5"; "int32|3|1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|1.7666666507720947,-0.7166666388511658,-0.7166666388511658,0.34166663885116577";
        ] );
    ( "opt.cov.a",
      false,
      [ "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5"; "float32|3|0.5,1.0,2.0" ],
      Some [ "float32|2,2|2.5,-1.0,-1.0,0.4642857313156128" ] );
    ( "opt.cov.fa",
      false,
      [
        "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5";
        "int32|3|1.0,2.0,3.0";
        "float32|3|0.5,1.0,2.0";
      ],
      Some
        [
          "float32|2,2|1.3103448152542114,-0.4568965435028076,-0.4568965435028076,0.18534483015537262";
        ] );
    ( "opt.cov.ddof0",
      false,
      [
        "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5";
        "int32|3|1.0,2.0,3.0";
        "float32|3|0.5,1.0,2.0";
      ],
      Some
        [
          "float32|2,2|1.0519031286239624,-0.36678197979927063,-0.36678197979927063,0.1487889289855957";
        ] );
    ( "opt.cov.bias",
      false,
      [ "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5"; "int32|3|1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|1.4722222089767456,-0.5972222089767456,-0.5972222089767456,0.2847222089767456";
        ] );
    ( "opt.cov.float_f",
      false,
      [ "float32|2,3|1.0,2.0,4.0,2.0,1.0,0.5"; "float32|3|1.0,2.0,3.0" ],
      None );
    ( "opt.cov.complex",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "int32|3|1.0,2.0,3.0";
      ],
      Some
        [
          "complex64|2,2|1.7083336114883423,-1.0333331823349,-1.033333420753479,2.5333333015441895|0.0,1.399999976158142,-1.399999976158142,0.0";
        ] );
    ( "opt.pad.edge",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|1.0,1.0,1.0,1.0,2.0,4.0,4.0,1.0,1.0,1.0,1.0,2.0,4.0,4.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0";
        ] );
    ( "opt.pad1.edge",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|1.0,1.0,1.0,1.0,1.0,1.0,5.0,9.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0";
        ] );
    ( "opt.pad.wrap",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|3.0,0.0,7.0,3.0,0.0,7.0,3.0,1.0,2.0,4.0,1.0,2.0,4.0,1.0,3.0,0.0,7.0,3.0,0.0,7.0,3.0,1.0,2.0,4.0,1.0,2.0,4.0,1.0,3.0,0.0,7.0,3.0,0.0,7.0,3.0";
        ] );
    ( "opt.pad1.wrap",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0";
        ] );
    ( "opt.pad.symmetric",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|4.0,2.0,1.0,1.0,2.0,4.0,4.0,4.0,2.0,1.0,1.0,2.0,4.0,4.0,7.0,0.0,3.0,3.0,0.0,7.0,7.0,7.0,0.0,3.0,3.0,0.0,7.0,7.0,4.0,2.0,1.0,1.0,2.0,4.0,4.0";
        ] );
    ( "opt.pad1.symmetric",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|2.0,2.0,9.0,5.0,1.0,1.0,5.0,9.0,2.0,2.0,9.0,5.0,1.0,1.0,5.0,9.0";
        ] );
    ( "opt.pad.reflect",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|0.0,7.0,0.0,3.0,0.0,7.0,0.0,2.0,4.0,2.0,1.0,2.0,4.0,2.0,0.0,7.0,0.0,3.0,0.0,7.0,0.0,2.0,4.0,2.0,1.0,2.0,4.0,2.0,0.0,7.0,0.0,3.0,0.0,7.0,0.0";
        ] );
    ( "opt.pad1.reflect",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|5.0,9.0,2.0,9.0,5.0,1.0,5.0,9.0,2.0,9.0,5.0,1.0,5.0,9.0,2.0,9.0";
        ] );
    ( "opt.pad.maximum",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|7.0,7.0,7.0,3.0,2.0,7.0,7.0,4.0,4.0,4.0,1.0,2.0,4.0,4.0,7.0,7.0,7.0,3.0,0.0,7.0,7.0,7.0,7.0,7.0,3.0,2.0,7.0,7.0,7.0,7.0,7.0,3.0,2.0,7.0,7.0";
        ] );
    ( "opt.pad1.maximum",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|9.0,9.0,9.0,9.0,9.0,1.0,5.0,9.0,2.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0";
        ] );
    ( "opt.pad.minimum",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|0.0,0.0,0.0,1.0,0.0,4.0,0.0,1.0,1.0,1.0,1.0,2.0,4.0,1.0,0.0,0.0,0.0,3.0,0.0,7.0,0.0,0.0,0.0,0.0,1.0,0.0,4.0,0.0,0.0,0.0,0.0,1.0,0.0,4.0,0.0";
        ] );
    ( "opt.pad1.minimum",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|1.0,1.0,1.0,1.0,1.0,1.0,5.0,9.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0";
        ] );
    ( "opt.pad.mean",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|2.8333334922790527,2.8333334922790527,2.8333334922790527,2.0,1.0,5.5,2.8333334922790527,2.3333334922790527,2.3333334922790527,2.3333334922790527,1.0,2.0,4.0,2.3333334922790527,3.3333334922790527,3.3333334922790527,3.3333334922790527,3.0,0.0,7.0,3.3333334922790527,2.8333334922790527,2.8333334922790527,2.8333334922790527,2.0,1.0,5.5,2.8333334922790527,2.8333334922790527,2.8333334922790527,2.8333334922790527,2.0,1.0,5.5,2.8333334922790527";
        ] );
    ( "opt.pad1.mean",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|4.0,4.0,4.0,4.0,4.0,1.0,5.0,9.0,2.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0";
        ] );
    ( "opt.pad.median",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|2.0,2.0,2.0,2.0,1.0,5.5,2.0,2.0,2.0,2.0,1.0,2.0,4.0,2.0,3.0,3.0,3.0,3.0,0.0,7.0,3.0,2.0,2.0,2.0,2.0,1.0,5.5,2.0,2.0,2.0,2.0,2.0,1.0,5.5,2.0";
        ] );
    ( "opt.pad1.median",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|4.0,4.0,4.0,4.0,4.0,1.0,5.0,9.0,2.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0";
        ] );
    ( "opt.pad.linear_ramp",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3333333432674408,0.6666666865348816,1.0,2.0,4.0,0.0,0.0,1.0,2.0,3.0,0.0,7.0,0.0,0.0,0.5,1.0,1.5,0.0,3.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        ] );
    ( "opt.pad1.linear_ramp",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|0.0,0.0,0.0,0.0,0.0,1.0,5.0,9.0,2.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0";
        ] );
    ( "opt.pad.stat.maximum",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|2.0,2.0,2.0,1.0,2.0,4.0,4.0,2.0,2.0,2.0,1.0,2.0,4.0,4.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0,3.0,3.0,3.0,3.0,0.0,7.0,7.0";
        ] );
    ( "opt.pad1.stat.maximum",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some [ "int32|9|5.0,5.0,1.0,5.0,9.0,2.0,9.0,9.0,9.0" ] );
    ( "opt.pad.stat.minimum",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|1.0,1.0,1.0,1.0,2.0,4.0,4.0,1.0,1.0,1.0,1.0,2.0,4.0,4.0,0.0,0.0,0.0,3.0,0.0,7.0,7.0,0.0,0.0,0.0,3.0,0.0,7.0,7.0,0.0,0.0,0.0,3.0,0.0,7.0,7.0";
        ] );
    ( "opt.pad1.stat.minimum",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some [ "int32|9|1.0,1.0,1.0,5.0,9.0,2.0,2.0,2.0,2.0" ] );
    ( "opt.pad.stat.mean",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|1.5,1.5,1.5,1.0,2.0,4.0,4.0,1.5,1.5,1.5,1.0,2.0,4.0,4.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0";
        ] );
    ( "opt.pad1.stat.mean",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some [ "int32|9|3.0,3.0,1.0,5.0,9.0,2.0,5.0,5.0,5.0" ] );
    ( "opt.pad.stat.median",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|1.5,1.5,1.5,1.0,2.0,4.0,4.0,1.5,1.5,1.5,1.0,2.0,4.0,4.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0,1.5,1.5,1.5,3.0,0.0,7.0,7.0";
        ] );
    ( "opt.pad1.stat.median",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some [ "int32|9|3.0,3.0,1.0,5.0,9.0,2.0,5.0,5.0,5.0" ] );
    ( "opt.pad.odd.symmetric",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,9|-4.0,-2.0,-2.0,0.0,1.0,1.0,2.0,4.0,4.0,-4.0,-2.0,-2.0,0.0,1.0,1.0,2.0,4.0,4.0,-8.0,-1.0,-1.0,6.0,3.0,3.0,0.0,7.0,7.0,-8.0,-1.0,-1.0,6.0,3.0,3.0,0.0,7.0,7.0,-12.0,0.0,0.0,12.0,5.0,5.0,-2.0,10.0,10.0";
        ] );
    ( "opt.pad1.odd.symmetric",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|0.0,0.0,-7.0,-3.0,1.0,1.0,5.0,9.0,2.0,2.0,-5.0,-1.0,3.0,3.0,7.0,11.0";
        ] );
    ( "opt.pad1.single.symmetric",
      false,
      [ "float32|1|3.0" ],
      Some [ "float32|6|3.0,3.0,3.0,3.0,3.0,3.0" ] );
    ( "opt.pad1.single_even.symmetric",
      false,
      [ "float32|1|3.0" ],
      Some [ "float32|6|3.0,3.0,3.0,3.0,3.0,3.0" ] );
    ( "opt.pad.odd.reflect",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,9|-10.0,-5.0,0.0,-3.0,-6.0,-1.0,4.0,1.0,-2.0,-6.0,-5.0,-4.0,-2.0,0.0,1.0,2.0,4.0,6.0,-2.0,-5.0,-8.0,-1.0,6.0,3.0,0.0,7.0,14.0,2.0,-5.0,-12.0,0.0,12.0,5.0,-2.0,10.0,22.0,6.0,-5.0,-16.0,1.0,18.0,7.0,-4.0,13.0,30.0";
        ] );
    ( "opt.pad1.odd.reflect",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|16|3.0,7.0,0.0,-7.0,-3.0,1.0,5.0,9.0,2.0,-5.0,-1.0,3.0,7.0,11.0,4.0,-3.0";
        ] );
    ( "opt.pad1.single.reflect",
      false,
      [ "float32|1|3.0" ],
      Some [ "float32|6|3.0,3.0,3.0,3.0,3.0,3.0" ] );
    ( "opt.pad1.single_even.reflect",
      false,
      [ "float32|1|3.0" ],
      Some [ "float32|6|3.0,3.0,3.0,3.0,3.0,3.0" ] );
    ( "opt.pad.ramp_end",
      false,
      [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ],
      Some
        [
          "float32|5,7|2.0,3.0,4.0,5.0,5.0,5.0,3.0,2.0,1.6666666269302368,1.3333332538604736,1.0,2.0,4.0,3.0,2.0,2.3333332538604736,2.6666667461395264,3.0,0.0,7.0,3.0,2.0,1.6666666269302368,1.3333332538604736,1.0,-0.5,3.0,3.0,2.0,0.9999998807907104,-5.960464477539063e-08,-1.0,-1.0,-1.0,3.0";
        ] );
    ("opt.pad.bad", false, [ "float32|2,3|1.0,2.0,4.0,3.0,0.0,7.0" ], None);
    ( "opt.pad.complex_edge",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|4,5|1.0,1.0,2.0,0.0,0.0,1.0,1.0,2.0,0.0,0.0,1.0,1.0,-0.0,3.0,3.0,1.0,1.0,-0.0,3.0,3.0|1.0,1.0,-1.0,0.5,0.5,1.0,1.0,-1.0,0.5,0.5,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0";
        ] );
    ( "opt.pad.complex_reflect",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|4,6|-0.0,1.0,-0.0,3.0,-0.0,1.0,2.0,1.0,2.0,0.0,2.0,1.0,-0.0,1.0,-0.0,3.0,-0.0,1.0,2.0,1.0,2.0,0.0,2.0,1.0|-1.0,0.0,-1.0,0.0,-1.0,0.0,-1.0,1.0,-1.0,0.5,-1.0,1.0,-1.0,0.0,-1.0,0.0,-1.0,0.0,-1.0,1.0,-1.0,0.5,-1.0,1.0";
        ] );
    ( "opt.pad.bool_edge",
      false,
      [ "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some
        [
          "bool|4,5|1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0";
        ] );
    ( "opt.pad1.wrap_big",
      false,
      [ "int32|4|1.0,5.0,9.0,2.0" ],
      Some
        [
          "int32|24|9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0,9.0,2.0,1.0,5.0";
        ] );
    ( "opt.union1d.size",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|8|1.0,2.0,3.0,5.0,7.0,1.0,1.0,1.0" ] );
    ( "opt.union1d.size_fill",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|8|1.0,2.0,3.0,5.0,7.0,-1.0,-1.0,-1.0" ] );
    ( "opt.union1d.cut",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|2|1.0,2.0" ] );
    ( "opt.intersect1d.size",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|4|2.0,5.0,2.0,2.0" ] );
    ( "opt.intersect1d.size_fill",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|4|2.0,5.0,-1.0,-1.0" ] );
    ( "opt.intersect1d.assume",
      false,
      [ "int32|3|1.0,3.0,2.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|1|2.0" ] );
    ( "opt.setdiff1d.size",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|5|1.0,3.0,0.0,0.0,0.0" ] );
    ( "opt.setdiff1d.size_fill",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|5|1.0,3.0,-1.0,-1.0,-1.0" ] );
    ( "opt.setdiff1d.assume",
      false,
      [ "int32|3|3.0,1.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|2|3.0,1.0" ] );
    ( "opt.setdiff1d.cut",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|1|1.0" ] );
    ( "opt.setxor1d.size",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|6|1.0,3.0,7.0,1.0,1.0,1.0" ] );
    ( "opt.setxor1d.fill",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|6|1.0,3.0,7.0,-1.0,-1.0,-1.0" ] );
    ( "opt.setxor1d.assume",
      false,
      [ "int32|3|3.0,1.0,5.0"; "int32|3|2.0,5.0,7.0" ],
      Some [ "int32|4|1.0,2.0,3.0,7.0" ] );
    ( "opt.bincount.min",
      false,
      [ "int32|5|3.0,1.0,2.0,2.0,5.0" ],
      Some [ "int32|9|0.0,1.0,2.0,1.0,0.0,1.0,0.0,0.0,0.0" ] );
    ( "opt.bincount.neg",
      false,
      [ "int32|4|0.0,1.0,1.0,-1.0" ],
      Some [ "int32|2|2.0,2.0" ] );
    ( "opt.bincount.drop",
      false,
      [ "int32|4|0.0,1.0,1.0,5.0" ],
      Some [ "int32|3|1.0,2.0,0.0" ] );
    ( "opt.bincount.w",
      false,
      [ "int32|3|0.0,1.0,1.0"; "float32|3|0.5,0.5,1.0" ],
      Some [ "float32|4|0.5,1.5,0.0,0.0" ] );
    ("opt.bincount.float", false, [ "float32|5|1.0,2.0,4.0,7.0,0.0" ], None);
    ( "opt.bincount.bool",
      false,
      [ "bool|3|1.0,0.0,1.0" ],
      Some [ "int32|2|1.0,2.0" ] );
    ( "opt.compress.size",
      false,
      [ "bool|3|1.0,0.0,1.0"; "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|2,2|1.5,3.0,0.5,-1.0" ] );
    ( "opt.compress.fill",
      false,
      [ "bool|2|1.0,0.0"; "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|2,3|1.5,-2.0,3.0,9.0,9.0,9.0" ] );
    ( "opt.compress.short",
      false,
      [ "bool|2|1.0,0.0"; "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|2,1|1.5,0.5" ] );
    ( "opt.compress.none",
      false,
      [ "bool|3|0.0,1.0,1.0"; "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some [ "float32|2|-2.0,3.0" ] );
    ( "opt.compress.oob",
      false,
      [ "bool|3|1.0,0.0,1.0"; "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      None );
    ( "opt.extract.size",
      false,
      [
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
      ],
      Some [ "float32|4|1.5,3.0,4.0,-1.0" ] );
    ( "opt.mask_indices.size",
      false,
      [],
      Some
        [
          "int32|8|0.0,0.0,0.0,1.0,1.0,2.0,0.0,0.0";
          "int32|8|0.0,1.0,2.0,1.0,2.0,2.0,0.0,0.0";
        ] );
    ( "opt.mask_indices.k",
      false,
      [],
      Some [ "int32|2|0.0,0.0"; "int32|2|1.0,2.0" ] );
    ( "opt.trim_zeros.axis",
      false,
      [ "int32|3,3|0.0,0.0,0.0,1.0,0.0,2.0,0.0,0.0,0.0" ],
      Some [ "int32|1,3|1.0,0.0,2.0" ] );
    ( "opt.trim_zeros.all",
      false,
      [ "int32|3,3|0.0,0.0,0.0,0.0,1.0,2.0,0.0,0.0,0.0" ],
      Some [ "int32|1,2|1.0,2.0" ] );
    ( "opt.trim_zeros.b",
      false,
      [ "int32|4|0.0,1.0,0.0,0.0" ],
      Some [ "int32|2|0.0,1.0" ] );
    ("opt.trim_zeros.empty", false, [ "int32|2|0.0,0.0" ], Some [ "int32|0|" ]);
    ( "opt.polymul.trim",
      false,
      [ "float32|3|0.0,1.0,2.0"; "float32|2|0.0,3.0" ],
      Some [ "float32|2|3.0,6.0" ] );
    ( "opt.polymul.notrim",
      false,
      [ "float32|3|0.0,1.0,2.0"; "float32|2|0.0,3.0" ],
      Some [ "float32|4|0.0,0.0,3.0,6.0" ] );
    ( "opt.polydiv.trim",
      false,
      [ "float32|4|0.0,1.0,2.0,3.0"; "float32|2|1.0,1.0" ],
      Some [ "float32|3|0.0,1.0,1.0"; "float32|1|2.0" ] );
    ( "opt.polyfit.w",
      false,
      [
        "float32|4|0.0,1.0,2.0,3.0";
        "float32|4|1.0,3.0,2.0,5.0";
        "float32|4|1.0,2.0,1.0,0.5";
      ],
      Some [ "float32|2|0.7432432770729065,1.7972968816757202" ] );
    ( "opt.take.default",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|nan,2.0,nan" ] );
    ( "opt.take.fill",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|nan,2.0,nan" ] );
    ( "opt.take.clip",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|2.0,0.0,0.0" ] );
    ( "opt.take.wrap",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|2.0,2.0,1.0" ] );
    ( "opt.take.fv",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|9.0,2.0,9.0" ] );
    ( "opt.take.int",
      false,
      [ "int32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "int32|3|-2147483648.0,2.0,-2147483648.0" ] );
    ( "opt.take.u8",
      false,
      [ "uint8|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "uint8|3|255.0,2.0,255.0" ] );
    ( "opt.take.bool",
      false,
      [ "bool|3|0.0,0.0,0.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "bool|3|1.0,0.0,1.0" ] );
    ( "opt.take.c",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "int32|3|5.0,-1.0,-5.0";
      ],
      Some [ "complex64|2,3|nan,0.0,nan,nan,3.0,nan|0.0,0.5,0.0,0.0,0.0,0.0" ]
    );
    ( "opt.take.axis",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|2,3|3.0,1.5,1.5,-1.0,0.5,0.5" ] );
    ( "opt.take_along.default",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|nan,2.0,nan" ] );
    ( "opt.take_along.clip",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|2.0,2.0,0.0" ] );
    ( "opt.take_along.fv",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|3|5.0,-1.0,-5.0" ],
      Some [ "float32|3|7.0,2.0,7.0" ] );
    ( "opt.take_along.2d",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "int32|2,3|0.0,5.0,1.0,2.0,-1.0,1.0";
      ],
      Some [ "float32|2,3|1.5,nan,-2.0,-1.0,-1.0,4.0" ] );
    ( "opt.put_along.default",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|2|5.0,1.0"; "float32|2|9.0,8.0" ],
      Some [ "float32|3|0.0,8.0,2.0" ] );
    ( "opt.put_along.clip",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|2|5.0,1.0"; "float32|2|9.0,8.0" ],
      Some [ "float32|3|0.0,8.0,9.0" ] );
    ( "opt.softmax.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some
        [
          "float32|2,3|0.18242551386356354,0.0,0.8175744414329529,0.0,0.9933071732521057,0.006692850962281227";
        ] );
    ( "opt.log_softmax.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some
        [
          "float32|2,3|-1.7014133930206299,-inf,-0.2014133334159851,-inf,-0.006715348921716213,-5.006715297698975";
        ] );
    ( "opt.softmax.where0",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some
        [ "float32|2,3|1.0,0.0,0.9820137619972229,0.0,1.0,0.01798621006309986" ]
    );
    ( "opt.standardize.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some
        [
          "float32|2,3|-0.9999910593032837,-5.666615962982178,0.9999910593032837,-0.3999997079372406,0.9999992847442627,-0.9999992847442627";
        ] );
    ( "opt.standardize.mean",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||1.0" ],
      Some
        [
          "float32|2,3|0.2474355399608612,-1.4846131801605225,0.9897421598434448,-0.22941550612449646,1.376492977142334,-0.9176620244979858";
        ] );
    ( "opt.standardize.var",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0"; "float32||2.0" ],
      Some
        [
          "float32|2,3|0.4714033007621765,-2.0034642219543457,1.5320606231689453,-0.4714033901691437,2.0034639835357666,-1.5320608615875244";
        ] );
    ( "opt.standardize.stable",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some
        [
          "float32|2,3|0.3182225227355957,-1.3524458408355713,1.034223198890686,-0.3182225823402405,1.3524457216262817,-1.0342233180999756";
        ] );
    ( "opt.standardize.eps",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      Some
        [
          "float32|2,3|0.30151131749153137,-1.2814233303070068,0.9799118041992188,-0.30151137709617615,1.2814232110977173,-0.9799119234085083";
        ] );
    ( "opt.standardize.bad",
      false,
      [ "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0" ],
      None );
    ( "opt.logmeanexp.where",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32||2.9898810386657715" ] );
    ( "opt.logmeanexp.where_axis",
      false,
      [
        "float32|2,3|1.5,-2.0,3.0,0.5,4.0,-1.0";
        "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
      ],
      Some [ "float32|2,1|2.5082662105560303,3.313568115234375" ] );
    ( "opt.one_hot.int",
      false,
      [ "int32|3|0.0,3.0,5.0" ],
      Some [ "int8|3,4|1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0" ] );
    ( "opt.one_hot.bool",
      false,
      [ "int32|3|0.0,3.0,5.0" ],
      Some [ "bool|4,3|1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0" ] );
  ]

let linalg =
  [
    ( "tri.vec",
      false,
      [
        "float32|7|-0.13210485875606537,0.6404226422309875,0.10490011423826218,-0.5356693863868713,0.3615950644016266,1.3040000200271606,0.9470809698104858";
        "float32|6|-0.7037352323532104,-1.2654215097427368,-0.6232744455337524,0.04132597893476486,-2.32503080368042,-0.218791663646698";
      ],
      Some
        [
          "float32|7|-1.5491738319396973,-1.3624820709228516,-0.5415053963661194,0.10414812713861465,0.9421420693397522,1.8791544437408447,3.2179415225982666";
          "float32|7,7|-0.016662517562508583,-0.2914756238460541,0.6012115478515625,0.6960412859916687,-0.0007698620902374387,0.26236411929130554,-0.00020108815806452185,-0.033552300184965134,-0.5096021890640259,0.34975698590278625,-0.23366989195346832,0.001175189157947898,-0.7498306632041931,0.0009572557755745947,-0.048790063709020615,-0.6444991230964661,-0.007670332212001085,-0.4861147403717041,0.00014793623995501548,0.5881077647209167,-0.001837989897467196,-0.061360735446214676,-0.48271670937538147,-0.7180595397949219,0.4738287031650543,-0.0025846848729997873,-0.15178047120571136,0.007236629258841276,0.7690024971961975,-0.06251408904790878,-0.01428033597767353,0.004378919489681721,-0.09019684046506882,0.0007125539123080671,0.6295776963233948,0.6308949589729309,-0.054935988038778305,-0.018309904262423515,0.008906882256269455,0.022475695237517357,-0.0031628913711756468,-0.7733199596405029,0.05529665946960449,-0.005204245913773775,-0.0026911802124232054,0.0023118702229112387,0.9956660270690918,0.0007424460491165519,0.07450742274522781";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|-0.13210485875606537,0.6404226422309875,0.10490011423826218,-0.5356693863868713,0.3615950644016266,1.3040000200271606,0.9470809698104858";
        "float32|6|-0.7037352323532104,-1.2654215097427368,-0.6232744455337524,0.04132597893476486,-2.32503080368042,-0.218791663646698";
      ],
      Some
        [
          "float32|7|-1.5491738319396973,-1.3624820709228516,-0.5415053963661194,0.10414812713861465,0.9421420693397522,1.8791544437408447,3.2179415225982666";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|6|-1.2459110021591187,-0.7322673797607422,-0.5442590117454529,-0.3163001537322998,0.4116305410861969,1.042513370513916";
        "float32|5|-0.12853465974330902,1.3664634227752686,-0.6651946902275085,0.35151007771492004,0.9034701585769653";
      ],
      Some
        [
          "float32|6|-2.1392102241516113,-1.2443445920944214,-0.5901490449905396,-0.05636954307556152,0.9368717670440674,1.7086080312728882";
          "float32|6,6|0.09694690257310867,-0.9917988777160645,0.0692562460899353,-0.031131545081734657,0.034190934151411057,-0.0013075944734737277,0.6737684011459351,0.012087345123291016,-0.3533335328102112,0.2881110906600952,-0.5806323289871216,0.030056584626436234,-0.6846086382865906,-0.09782198816537857,-0.03023374453186989,0.1395808607339859,-0.7060280442237854,0.05356624349951744,-0.2574244737625122,-0.07812274992465973,-0.7279143333435059,0.48947057127952576,0.3792979121208191,-0.11967384815216064,0.039440128952264786,0.021139396354556084,0.5098780989646912,0.6260894536972046,0.016157109290361404,-0.5880245566368103,-0.011199269443750381,-0.008351552300155163,-0.28215235471725464,-0.5147528052330017,-0.13817907869815826,-0.7975781559944153";
        ] );
    ( "tri.val",
      false,
      [
        "float32|6|-1.2459110021591187,-0.7322673797607422,-0.5442590117454529,-0.3163001537322998,0.4116305410861969,1.042513370513916";
        "float32|5|-0.12853465974330902,1.3664634227752686,-0.6651946902275085,0.35151007771492004,0.9034701585769653";
      ],
      Some
        [
          "float32|6|-2.1392102241516113,-1.2443445920944214,-0.5901490449905396,-0.05636954307556152,0.9368717670440674,1.7086080312728882";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|7|-0.7434992493538084,-0.9217253762584194,-0.45772582566733916,0.2201951234700494,-1.009618183538736,-0.20917557487171307,-0.15922500991447772";
        "float64|6|0.5408455846858077,0.2146591225063409,0.3553727090399214,-0.6538286094183394,-0.12961363369276946,0.7839754700613295";
      ],
      Some
        [
          "float64|7|-1.4186880852858001,-1.3306038339284771,-0.9519432607178031,-0.564756899617644,-0.22682703700503376,0.5737708116393299,0.6382742087809843";
          "float64|7,7|-0.5829902093626609,0.2127099229854974,0.03327446929481049,-0.4254793987314645,-0.6575202272569853,-0.015166307745383606,0.01319580449762021,0.7278019678167346,-0.23090319031051776,-0.012824111081257345,-0.14061534317188273,-0.6281320215977912,-0.03693868212016193,0.033713157562596616,-0.2160763370838954,-0.09611463095259161,-0.08203160475229206,0.8381819846905559,-0.37674143983439917,-0.21913360665677054,0.21175722083630996,0.14467029137714846,0.37555451066342355,0.1218277617387728,-0.1675062555361748,0.1346336543594444,-0.6137397378828691,0.6327134267553255,0.2451866984185857,0.8385271291369378,0.17381803204833274,0.25447437754825986,-0.11271975250930238,0.2127917694871699,-0.2894818470284278,0.04404569977602184,0.18213278784829331,-0.6918980285828327,-0.02843386216441657,0.0016100875121035569,0.49646361621809537,0.488751002901589,-0.027417038944080475,-0.12189706270909983,0.6842671802283399,0.054968427938735034,-0.01867206012107296,0.5309925178943975,0.4804629123894335";
        ] );
    ( "tri.val",
      true,
      [
        "float64|7|-0.7434992493538084,-0.9217253762584194,-0.45772582566733916,0.2201951234700494,-1.009618183538736,-0.20917557487171307,-0.15922500991447772";
        "float64|6|0.5408455846858077,0.2146591225063409,0.3553727090399214,-0.6538286094183394,-0.12961363369276946,0.7839754700613295";
      ],
      Some
        [
          "float64|7|-1.4186880852858001,-1.3306038339284771,-0.9519432607178031,-0.564756899617644,-0.22682703700503376,0.5737708116393299,0.6382742087809843";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|4|1.4934310913085938,-1.2590655088424683,1.5139237642288208,1.3458753824234009";
        "float32|3|0.7813113927841187,0.2644556164741516,-0.31392282247543335";
      ],
      Some
        [
          "float32|4|-1.4874294996261597,1.1113494634628296,1.6720811128616333,1.798163652420044";
          "float32|4,4|-0.2525956332683563,0.08123084157705307,-0.847524106502533,0.45967358350753784,0.963703453540802,-0.039723992347717285,-0.19378973543643951,0.17928513884544373,-0.08590955287218094,-0.5960506200790405,0.35602980852127075,0.7145529985427856,-0.009518555365502834,-0.7978390455245972,-0.3426239788532257,-0.495954692363739";
        ] );
    ( "tri.val",
      false,
      [
        "float32|4|1.4934310913085938,-1.2590655088424683,1.5139237642288208,1.3458753824234009";
        "float32|3|0.7813113927841187,0.2644556164741516,-0.31392282247543335";
      ],
      Some
        [
          "float32|4|-1.4874294996261597,1.1113494634628296,1.6720811128616333,1.798163652420044";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|3|1.9602583646774292,1.8016349077224731,1.3151037693023682";
        "float32|2|0.3573804199695587,-1.2083185911178589";
      ],
      Some
        [
          "float32|3|0.2944599688053131,1.9062579870224,2.876279354095459";
          "float32|3,3|0.13713185489177704,-0.9456630945205688,0.2948155105113983,-0.6391901969909668,0.14289039373397827,0.755657434463501,-0.7567236423492432,-0.29206788539886475,-0.5848637223243713";
        ] );
    ( "tri.val",
      false,
      [
        "float32|3|1.9602583646774292,1.8016349077224731,1.3151037693023682";
        "float32|2|0.3573804199695587,-1.2083185911178589";
      ],
      Some [ "float32|3|0.2944599688053131,1.9062579870224,2.876279354095459" ]
    );
    ( "tri.vec",
      true,
      [
        "float64|3|-0.004454133120083229,0.6564749350763358,-1.2883614637495544";
        "float64|2|0.39512206018200824,0.42986369482223";
      ],
      Some
        [
          "float64|3|-1.3842255261117449,-0.16314076708460556,0.9110256314030485";
          "float64|3,3|0.06221106138836773,-0.9186948306438175,-0.39003793660265956,-0.21724183861508034,0.36896089843305746,-0.9037000824292373,0.9741333416920253,0.1409526997818722,-0.17662550503290658";
        ] );
    ( "tri.val",
      true,
      [
        "float64|3|-0.004454133120083229,0.6564749350763358,-1.2883614637495544";
        "float64|2|0.39512206018200824,0.42986369482223";
      ],
      Some
        [
          "float64|3|-1.3842255261117449,-0.16314076708460556,0.9110256314030485";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|8|-1.1841179132461548,-0.661702573299408,-0.4364352524280548,-1.169801950454712,1.7393678426742554,-0.49591073393821716,0.3289696276187897,-0.2585725486278534";
        "float32|7|1.5834728479385376,1.3203610181808472,0.6333526372909546,-2.20350980758667,0.052028972655534744,0.6836861968040466,1.0039615631103516";
      ],
      Some
        [
          "float32|8|-2.955469846725464,-2.39867901802063,-1.270559549331665,-0.6509798765182495,-0.4143444895744324,1.2591626644134521,1.33427095413208,2.95839524269104";
          "float32|8,8|0.551705002784729,0.35865548253059387,0.006214615888893604,-0.6073936223983765,-0.017031824216246605,-0.012485797517001629,0.4441276788711548,0.018081555142998695,-0.6171648502349854,-0.27509722113609314,-0.0003392546786926687,-0.20450273156166077,-0.008279680274426937,-0.019265444949269295,0.7063500285148621,0.04730304330587387,0.41051098704338074,-0.06822671741247177,-0.0072965784929692745,0.726769745349884,0.018874675035476685,-0.013053555972874165,0.535150408744812,0.10800834745168686,-0.346111923456192,0.7848787903785706,0.010316829197108746,0.1801409274339676,0.017919128760695457,0.005216304678469896,0.023615531623363495,0.4803215563297272,-0.1624874770641327,0.418109267950058,-0.0016255021328106523,0.16648007929325104,-0.000718316703569144,-0.009501988999545574,0.1269810050725937,-0.8688206672668457,0.003677323227748275,-0.01282927580177784,0.5309710502624512,-0.019278498366475105,0.7886379957199097,0.30861809849739075,0.01148241851478815,-0.013898225501179695,-0.0008637563441880047,0.0038867832627147436,-0.6014915108680725,-0.008296618238091469,0.09414225071668625,0.7929688096046448,0.021074315533041954,-0.0041025858372449875,0.0003215466276742518,-0.0018233583541586995,0.5967215299606323,0.02122662030160427,-0.60675448179245,0.5245382189750671,0.013283041305840015,-0.001280347933061421";
        ] );
    ( "tri.val",
      false,
      [
        "float32|8|-1.1841179132461548,-0.661702573299408,-0.4364352524280548,-1.169801950454712,1.7393678426742554,-0.49591073393821716,0.3289696276187897,-0.2585725486278534";
        "float32|7|1.5834728479385376,1.3203610181808472,0.6333526372909546,-2.20350980758667,0.052028972655534744,0.6836861968040466,1.0039615631103516";
      ],
      Some
        [
          "float32|8|-2.955469846725464,-2.39867901802063,-1.270559549331665,-0.6509798765182495,-0.4143444895744324,1.2591626644134521,1.33427095413208,2.95839524269104";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|7|-0.6179070472717285,1.8220113515853882,-1.3204309940338135,-0.6615280508995056,0.9350500106811523,0.04905461519956589,2.002392530441284";
        "float32|6|0.1885191947221756,-0.633194088935852,-0.3775635063648224,-1.0911461114883423,-1.2776801586151123,0.6304115056991577";
      ],
      Some
        [
          "float32|7|-1.7679235935211182,-1.2726340293884277,-0.6301308274269104,-0.3422088027000427,1.8234995603561401,1.9621987342834473,2.4358408451080322";
          "float32|7,7|0.0192291047424078,0.0414847768843174,-0.9955494999885559,0.04079969972372055,0.011343664489686489,0.070673368871212,0.0015189767582342029,-0.11730259656906128,-0.14407655596733093,0.06455156207084656,0.05966714397072792,0.14690545201301575,0.9672477841377258,0.02460530586540699,-0.6593296527862549,-0.6918023228645325,-0.04641643911600113,0.21608597040176392,0.0030320456717163324,-0.1931048184633255,-0.02340058609843254,-0.5847225785255432,0.3292018175125122,-0.023393109440803528,-0.6599180698394775,-0.27161577343940735,0.05677466094493866,0.191541388630867,-0.3647501468658447,0.423753023147583,0.01673436351120472,0.11835107952356339,0.6175414323806763,-0.06969907879829407,-0.5356195569038391,-0.2722843587398529,0.4510561525821686,0.04047773778438568,0.6818857789039612,-0.1974530965089798,0.007546395994722843,0.4655730128288269,0.045527003705501556,-0.08682402223348618,-0.009693222120404243,-0.1833440363407135,0.6958165764808655,-0.11836044490337372,0.6771337389945984";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|-0.6179070472717285,1.8220113515853882,-1.3204309940338135,-0.6615280508995056,0.9350500106811523,0.04905461519956589,2.002392530441284";
        "float32|6|0.1885191947221756,-0.633194088935852,-0.3775635063648224,-1.0911461114883423,-1.2776801586151123,0.6304115056991577";
      ],
      Some
        [
          "float32|7|-1.7679235935211182,-1.2726340293884277,-0.6301308274269104,-0.3422088027000427,1.8234995603561401,1.9621987342834473,2.4358408451080322";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|3|1.294558819441117,-0.7546057912599311,1.689107452443673";
        "float64|2|-0.2873877078086663,1.5744082788445868";
      ],
      Some
        [
          "float64|3|-1.5491186761221196,1.3042754513487815,2.473903705398197";
          "float64|3,3|0.0905158217244084,-0.9900134520104115,0.108075209257183,0.8956465368431189,0.033472539135966414,-0.44350520872332366,-0.4354585710012869,-0.13694142529643827,-0.8897319141063207";
        ] );
    ( "tri.val",
      true,
      [
        "float64|3|1.294558819441117,-0.7546057912599311,1.689107452443673";
        "float64|2|-0.2873877078086663,1.5744082788445868";
      ],
      Some
        [ "float64|3|-1.5491186761221196,1.3042754513487815,2.473903705398197" ]
    );
    ( "tri.vec",
      false,
      [ "float32|2|-0.0,-1.0"; "float32|1|1.0" ],
      Some
        [
          "float32|2|-1.6180340051651,0.6180340051651001";
          "float32|2,2|-0.5257311463356018,0.8506508469581604,0.8506507873535156,0.5257311463356018";
        ] );
    ( "tri.val",
      false,
      [ "float32|2|-0.0,-1.0"; "float32|1|1.0" ],
      Some [ "float32|2|-1.6180340051651,0.6180340051651001" ] );
    ( "tri.vec",
      false,
      [
        "float32|5|0.16100957989692688,-0.5855288505554199,-1.3412196636199951,-1.4015202522277832,0.5026828646659851";
        "float32|4|0.9897130131721497,-0.16429458558559418,-1.0743649005889893,0.8730421662330627";
      ],
      Some
        [
          "float32|5|-2.587062358856201,-1.279011607170105,-0.5715122222900391,0.8453264236450195,0.9276833534240723";
          "float32|5,5|-0.02309562638401985,-0.5593675374984741,0.12076755613088608,-0.7982349395751953,0.18657641112804413,0.06412812322378159,0.8138734102249146,-0.08938434720039368,-0.5519231557846069,0.1445300430059433,0.6421182751655579,0.06569848209619522,0.7351314425468445,-0.0018350143218412995,-0.2072347104549408,0.7347992062568665,-0.1282636523246765,-0.513001561164856,0.08813608437776566,0.41554784774780273,-0.20762573182582855,0.06285004317760468,0.41693738102912903,0.22456735372543335,0.853624165058136";
        ] );
    ( "tri.val",
      false,
      [
        "float32|5|0.16100957989692688,-0.5855288505554199,-1.3412196636199951,-1.4015202522277832,0.5026828646659851";
        "float32|4|0.9897130131721497,-0.16429458558559418,-1.0743649005889893,0.8730421662330627";
      ],
      Some
        [
          "float32|5|-2.587062358856201,-1.279011607170105,-0.5715122222900391,0.8453264236450195,0.9276833534240723";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|8|-1.2803939447145731,-0.7130680950592722,0.6210178535400985,-2.2501411735745918,0.38636959756630584,-0.5816408364095031,0.10927969747781388,-0.07570152622082311";
        "float64|7|0.20211439504395987,0.6941719367070082,-0.7583697508984092,1.4209820223119163,0.726093788947765,0.843732662303268,1.1648639811110282";
      ],
      Some
        [
          "float64|8|-3.0482455999154006,-1.5940768874740856,-1.37103720531637,-0.8946657499728801,-0.5344466068393676,0.836927780178967,1.253918590537296,1.5673472514072952";
          "float64|8,8|0.006663762938375641,-0.04336167749074012,0.9013919687610649,-0.42736376559686645,0.0423154843112564,-0.029446510604440587,0.015397910736048319,0.005770341610266637,-0.058286518077601726,0.06729762417627835,-0.4042517956774188,-0.81560867431455,0.15617454098712918,-0.30847746698858064,0.19307441306478476,0.08130266780690433,0.1941342141767958,-0.07278568424053464,0.12072067081191863,0.33779670299320946,0.027865661801860092,-0.6802165469252699,0.542607156090532,0.26540626684760926,0.8859381735762104,-0.15099634991564895,-0.05292672720094748,-0.0714423655904407,0.1854105669322806,-0.08870461389719529,-0.2761049336367193,-0.25676620420900514,-0.39398504228680503,-0.10855997564745781,0.03168429187400899,0.1121312080234592,0.2387365722955231,-0.5557374410554489,-0.3912726031006012,-0.548160297447686,0.1298520903528443,0.5916044557748411,0.026891481604251064,-0.05801691812557243,-0.6656131674281317,-0.17125114096887717,0.07284457597273754,-0.3890749462265456,-0.04056224678955361,-0.6164712807621472,-0.05242632820340957,-0.07497283883604808,-0.2426814228194109,0.19032807105546012,0.49519376657163805,-0.519242239820762,0.01589530688416707,0.47294312636664976,0.04714572629443678,0.10663855268504957,0.616226440937403,0.24293139943253617,0.43383322430202026,-0.3681245443679207";
        ] );
    ( "tri.val",
      true,
      [
        "float64|8|-1.2803939447145731,-0.7130680950592722,0.6210178535400985,-2.2501411735745918,0.38636959756630584,-0.5816408364095031,0.10927969747781388,-0.07570152622082311";
        "float64|7|0.20211439504395987,0.6941719367070082,-0.7583697508984092,1.4209820223119163,0.726093788947765,0.843732662303268,1.1648639811110282";
      ],
      Some
        [
          "float64|8|-3.0482455999154006,-1.5940768874740856,-1.37103720531637,-0.8946657499728801,-0.5344466068393676,0.836927780178967,1.253918590537296,1.5673472514072952";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|8|0.8440786600112915,0.07559361308813095,-1.4267739057540894,-0.1350450962781906,-0.7695146203041077,-1.4227417707443237,0.25845280289649963,-0.5685494542121887";
        "float32|7|-1.0298044681549072,-1.0430010557174683,0.268417090177536,0.35867196321487427,1.3224574327468872,-0.013914668932557106,1.0418397188186646";
      ],
      Some
        [
          "float32|8|-2.4807071685791016,-2.0720603466033936,-1.2759333848953247,-0.32001379132270813,-0.0757712870836258,0.42674657702445984,0.9659514427185059,1.6872882843017578";
          "float32|8,8|0.006849352736026049,-0.17400062084197998,0.00019252086349297315,0.35322412848472595,0.5167666673660278,-0.07930688560009003,0.00028856503195129335,-0.7560238242149353,0.02211354859173298,-0.49272459745407104,0.000396334013203159,0.39928510785102844,0.4615902900695801,-0.032139409333467484,-3.4150412830058485e-05,0.6190364956855774,0.047435611486434937,-0.8427751660346985,0.0003234869509469718,-0.19730724394321442,-0.44324031472206116,0.08912400156259537,-0.00025576146435923874,-0.2101062685251236,-0.10032714903354645,0.1114649549126625,0.001721841748803854,0.7379671335220337,-0.4373033046722412,0.4905489683151245,-0.0024126104544848204,-0.032145049422979355,0.6206257939338684,0.028733545914292336,-0.005719037260860205,-0.23291541635990143,0.2594365179538727,0.7016547322273254,-0.007214461453258991,-0.006086026318371296,-0.7758479118347168,-0.05853202939033508,0.0017230425728484988,-0.2793162167072296,0.2547006607055664,0.5016539692878723,-0.008813227526843548,-0.0025880904868245125,-0.004971493501216173,-0.0005063175340183079,-0.5617194771766663,-0.0007858419558033347,0.001397011219523847,0.007568690925836563,0.8272765278816223,3.8000987842679024e-05,0.002708719577640295,0.0003508466179482639,0.8273043036460876,-0.0032941815443336964,0.002953583374619484,0.007922631688416004,0.561674177646637,1.7550437405589037e-05";
        ] );
    ( "tri.val",
      false,
      [
        "float32|8|0.8440786600112915,0.07559361308813095,-1.4267739057540894,-0.1350450962781906,-0.7695146203041077,-1.4227417707443237,0.25845280289649963,-0.5685494542121887";
        "float32|7|-1.0298044681549072,-1.0430010557174683,0.268417090177536,0.35867196321487427,1.3224574327468872,-0.013914668932557106,1.0418397188186646";
      ],
      Some
        [
          "float32|8|-2.4807071685791016,-2.0720603466033936,-1.2759333848953247,-0.32001379132270813,-0.0757712870836258,0.42674657702445984,0.9659514427185059,1.6872882843017578";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|8|1.4022648334503174,1.1501656770706177,-2.3653039932250977,1.2286837100982666,0.3396199941635132,0.42377135157585144,0.3712274134159088,0.38275715708732605";
        "float32|7|0.3194142282009125,-0.3589133024215698,-1.9016352891921997,-0.10891472548246384,-0.8037318587303162,1.0801633596420288,-0.2887665033340454";
      ],
      Some
        [
          "float32|8|-3.210296869277954,-0.9757906198501587,0.19451889395713806,0.5359439849853516,0.9385926723480225,1.6121132373809814,1.7543563842773438,2.0837481021881104";
          "float32|8,8|-0.005249771289527416,-0.00029617457767017186,-0.0021363780833780766,0.00514507619664073,-0.5629051923751831,0.8206103444099426,-0.05023990944027901,0.08457961678504944,0.07581031322479248,0.0022050351835787296,0.008077917620539665,-0.01395456027239561,0.8171312212944031,0.5391232967376709,-0.05537967011332512,0.1804540902376175,0.9163527488708496,0.012797537259757519,0.019607091322541237,-0.01930209808051586,-0.01927252672612667,0.03641001507639885,0.04851463809609413,-0.39411425590515137,0.3928735554218292,-0.009767260402441025,-0.027918048202991486,0.0320822112262249,-0.12074072659015656,-0.17790809273719788,-0.09464871138334274,0.8880081176757812,0.01275580283254385,-0.42113614082336426,-0.6074238419532776,0.5410670042037964,0.014906471595168114,-0.00939610879868269,-0.3902400732040405,-0.09036969393491745,0.003100939793512225,-0.6879200339317322,-0.1058775782585144,-0.1365116685628891,0.0052528358064591885,0.0389847606420517,0.6997302770614624,0.07577051222324371,-0.0009413219522684813,0.5779739022254944,-0.42950278520584106,0.3884226083755493,0.013595239259302616,0.03589759021997452,0.5715823769569397,0.049200236797332764,-7.565214764326811e-05,0.12285139411687851,-0.6588778495788574,-0.7321992516517639,-0.007062967866659164,-0.008432075381278992,-0.12033677846193314,-0.008352413773536682";
        ] );
    ( "tri.val",
      false,
      [
        "float32|8|1.4022648334503174,1.1501656770706177,-2.3653039932250977,1.2286837100982666,0.3396199941635132,0.42377135157585144,0.3712274134159088,0.38275715708732605";
        "float32|7|0.3194142282009125,-0.3589133024215698,-1.9016352891921997,-0.10891472548246384,-0.8037318587303162,1.0801633596420288,-0.2887665033340454";
      ],
      Some
        [
          "float32|8|-3.210296869277954,-0.9757906198501587,0.19451889395713806,0.5359439849853516,0.9385926723480225,1.6121132373809814,1.7543563842773438,2.0837481021881104";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|5|-0.8496059556101431,-0.5106224678981267,-0.011533061686586776,-1.4853751842635452,0.3006851142946058";
        "float64|4|-0.10607225344775092,-1.1857198050052338,-2.3982328653977714,0.513052133880305";
      ],
      Some
        [
          "float64|5|-3.477627808272422,-0.9015918941967511,-0.6785906237289492,0.3430297654763741,2.158329005557952";
          "float64|5,5|-0.009749502358893075,0.8769507309789533,-0.47998627180762576,-0.01819357464888178,0.012022285845251848,-0.2415514370529921,0.429792951147296,0.773859410953604,0.20456157303378728,-0.34092095464816496,-0.6035576440615766,0.06326601193890817,0.15256299409119234,-0.1456453788598517,0.7663077309272601,-0.7528775385216286,-0.18901564899866055,-0.3401721920884891,-0.07960539360123889,-0.5247802944685831,0.10223224905009724,0.0806593500255151,0.17821953742056493,-0.9645071081157466,-0.1449360941360709";
        ] );
    ( "tri.val",
      true,
      [
        "float64|5|-0.8496059556101431,-0.5106224678981267,-0.011533061686586776,-1.4853751842635452,0.3006851142946058";
        "float64|4|-0.10607225344775092,-1.1857198050052338,-2.3982328653977714,0.513052133880305";
      ],
      Some
        [
          "float64|5|-3.477627808272422,-0.9015918941967511,-0.6785906237289492,0.3430297654763741,2.158329005557952";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|3|-0.29758402705192566,-0.5300084352493286,-0.23615463078022003";
        "float32|2|1.8164759874343872,-0.049800969660282135";
      ],
      Some
        [
          "float32|3|-2.2346463203430176,-0.23620113730430603,1.4071000814437866";
          "float32|3,3|-0.6839243769645691,0.02755644917488098,0.7290322780609131,0.7293264865875244,0.0009311973117291927,0.6841651797294617,0.018174290657043457,0.9996197819709778,-0.02073451690375805";
        ] );
    ( "tri.val",
      false,
      [
        "float32|3|-0.29758402705192566,-0.5300084352493286,-0.23615463078022003";
        "float32|2|1.8164759874343872,-0.049800969660282135";
      ],
      Some
        [
          "float32|3|-2.2346463203430176,-0.23620113730430603,1.4071000814437866";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|3|-1.4870728254318237,1.647339105606079,0.9174879789352417";
        "float32|2|1.0669348239898682,0.047672729939222336";
      ],
      Some
        [
          "float32|3|-1.8158514499664307,0.9156026244163513,1.9780030250549316";
          "float32|3,3|0.9556426405906677,-0.017544135451316833,-0.29400551319122314,-0.2944837212562561,-0.039508379995822906,-0.9548394680023193,0.0051361508667469025,0.9990651607513428,-0.042922355234622955";
        ] );
    ( "tri.val",
      false,
      [
        "float32|3|-1.4870728254318237,1.647339105606079,0.9174879789352417";
        "float32|2|1.0669348239898682,0.047672729939222336";
      ],
      Some
        [
          "float32|3|-1.8158514499664307,0.9156026244163513,1.9780030250549316";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|5|0.9166547888245957,0.37094683509441023,0.6131890778590062,-0.1521929584082903,-1.473887948041959";
        "float64|4|1.028854347803183,-1.934959636609707,-0.23993667125803605,-0.20452248839966083";
      ],
      Some
        [
          "float64|5|-1.6908004977865811,-1.5023979841557518,-0.1282777570996234,0.8614375807781992,2.734748453591519";
          "float64|5,5|0.28388515964963396,-0.03831004045126457,0.1082008920367204,0.8711615993404707,0.3838119223374068,-0.7194583586096654,0.09007495549248445,-0.10989177800396518,-0.04675405355046088,0.6782359679517453,-0.6156536026184674,0.06683653606326938,0.0291800816308421,0.4750646020301133,-0.6244723098456434,-0.10976479073555867,-0.1370898913016248,0.9763918803043562,-0.1144754973545013,0.05207983293801144,-0.10349501753741255,-0.9834419567740749,-0.1484041205672084,0.010025503207494976,-0.002530866535247997";
        ] );
    ( "tri.val",
      true,
      [
        "float64|5|0.9166547888245957,0.37094683509441023,0.6131890778590062,-0.1521929584082903,-1.473887948041959";
        "float64|4|1.028854347803183,-1.934959636609707,-0.23993667125803605,-0.20452248839966083";
      ],
      Some
        [
          "float64|5|-1.6908004977865811,-1.5023979841557518,-0.1282777570996234,0.8614375807781992,2.734748453591519";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|2|0.6131231188774109,-0.2003297060728073";
        "float32|1|-0.4368683397769928";
      ],
      Some
        [
          "float32|2|-0.39049553871154785,0.8032888770103455";
          "float32|2,2|0.3991197645664215,0.916898787021637,0.9168988466262817,-0.3991197943687439";
        ] );
    ( "tri.val",
      false,
      [
        "float32|2|0.6131231188774109,-0.2003297060728073";
        "float32|1|-0.4368683397769928";
      ],
      Some [ "float32|2|-0.39049553871154785,0.8032888770103455" ] );
    ( "tri.vec",
      false,
      [
        "float32|7|1.0,-0.0,1.0,0.0,-0.0,-2.0,-1.0";
        "float32|6|2.0,1.0,1.0,3.0,0.0,0.0";
      ],
      Some
        [
          "float32|7|-3.1389944553375244,-2.0,-1.7697582244873047,-1.0000001192092896,0.9104981422424316,2.6859724521636963,3.3122823238372803";
          "float32|7,7|0.04268089681863785,1.0385009606298325e-19,0.5596436262130737,-3.130334973442655e-21,-0.41595399379730225,0.684298574924469,0.20903368294239044,-0.08832799643278122,-1.4271008481314123e-19,-0.7750388383865356,4.03543373787619e-21,0.01861432194709778,0.5768542885780334,0.24167242646217346,0.19189928472042084,4.261823862685291e-20,0.2523442208766937,-1.2713978060751359e-21,0.8488562703132629,0.18081766366958618,0.3824198842048645,-0.7059419751167297,3.247472310791973e-20,0.07610628753900528,-1.11575653397344e-22,-0.0945884957909584,-0.27200067043304443,0.6425902843475342,0.6746829152107239,-4.151813364859713e-20,-0.12901131808757782,4.054021825158094e-22,-0.31165963411331177,-0.3038013279438019,0.5820068120956421,6.702172348177303e-22,1.0,-1.422636089076369e-18,4.419851791172788e-21,-3.5734257623389593e-25,4.993228811519829e-24,-5.251360271356701e-23,-6.721605841914114e-22,3.419622167196183e-20,-1.2033304947951467e-20,-1.0,-2.0681231086294548e-25,-1.480102640003639e-23,-3.518007248367245e-23";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|1.0,-0.0,1.0,0.0,-0.0,-2.0,-1.0";
        "float32|6|2.0,1.0,1.0,3.0,0.0,0.0";
      ],
      Some
        [
          "float32|7|-3.1389944553375244,-2.0,-1.7697582244873047,-1.0000001192092896,0.9104981422424316,2.6859724521636963,3.3122823238372803";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|7|0.5863372815313004,-0.663535198304047,-0.6134178486140281,-1.6051493968851136,0.7293494040178566,0.8061393585150219,-0.476376747401162";
        "float64|6|0.16333994554129863,-1.2926461227593415,-0.4718131547409021,1.377950952722521,0.13573073406713437,2.310363486795888";
      ],
      Some
        [
          "float64|7|-2.454255845743693,-2.2316433217923044,-1.7677374903553305,0.5088370417332959,0.7409818246622044,1.3960955342304144,2.571069110125241";
          "float64|7,7|0.015923097825735824,0.0037225073283413228,-0.04564775926599846,0.800832994236975,0.596516855818434,0.02197754146187721,0.00015396765232184346,-0.296410419713397,-0.06422160490033686,0.6578809496056285,-0.37997287733775553,0.564761279503435,0.10895372539653739,0.0018708497736077971,-0.4086094186689192,-0.07743680535683548,0.5562060159234122,0.44581240053085286,-0.5382616231777947,-0.17082372889028066,-0.004661994929692227,-0.7821527661420807,-0.089642270447933,-0.4416318782470251,-0.019381994624314523,-0.0021515120029618135,0.4290553507978325,0.026340332447955812,0.3420612743244011,0.014241896188886723,0.24255536780767584,0.12291212614880788,-0.18796506760956738,0.8760132658805434,0.07823456495443395,-0.08266307167913456,0.599366838395378,0.021109716162679797,-0.0029190539267150463,0.005733331413447371,-0.05259512085391799,0.7941491993281038,0.09655885573284388,-0.7889145038298867,-0.037767229416707876,-0.0068452915323033514,0.010881000766172922,-0.06489487080611314,0.6020692077808126";
        ] );
    ( "tri.val",
      true,
      [
        "float64|7|0.5863372815313004,-0.663535198304047,-0.6134178486140281,-1.6051493968851136,0.7293494040178566,0.8061393585150219,-0.476376747401162";
        "float64|6|0.16333994554129863,-1.2926461227593415,-0.4718131547409021,1.377950952722521,0.13573073406713437,2.310363486795888";
      ],
      Some
        [
          "float64|7|-2.454255845743693,-2.2316433217923044,-1.7677374903553305,0.5088370417332959,0.7409818246622044,1.3960955342304144,2.571069110125241";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|3|-0.7871927618980408,0.5802844166755676,-0.19550582766532898";
        "float32|2|0.5658178329467773,-0.007211359683424234";
      ],
      Some
        [
          "float32|3|-0.9909571409225464,-0.19554533064365387,0.7840882539749146";
          "float32|3,3|0.9408458471298218,0.005236613564193249,0.33879438042640686,-0.3388209044933319,0.0054756649769842625,0.9408349394798279,-0.0030716643668711185,0.9999712705612183,-0.006926029920578003";
        ] );
    ( "tri.val",
      false,
      [
        "float32|3|-0.7871927618980408,0.5802844166755676,-0.19550582766532898";
        "float32|2|0.5658178329467773,-0.007211359683424234";
      ],
      Some
        [
          "float32|3|-0.9909571409225464,-0.19554533064365387,0.7840882539749146";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|6|-0.8676167726516724,3.0660367012023926,-0.07734505832195282,-2.016660690307617,-0.6486005783081055,0.6780397295951843";
        "float32|5|-0.5000084042549133,1.3604462146759033,1.0023982524871826,-0.1523386389017105,-0.4722159504890442";
      ],
      Some
        [
          "float32|6|-2.510044813156128,-0.9527393579483032,-0.7933416366577148,-0.0837143063545227,0.8299175500869751,3.643775701522827";
          "float32|6,6|0.03269617632031441,0.9660745859146118,-0.1161692813038826,-0.20391419529914856,0.002074398333206773,0.10271387547254562,0.10740042477846146,0.1644672155380249,0.017256638035178185,0.3196924030780792,-0.0070426082238554955,-0.9267497062683105,-0.4281867742538452,-0.1307743638753891,-0.09165046364068985,-0.8151077032089233,0.012338103726506233,-0.3558109998703003,0.8933947086334229,-0.10900820046663284,0.04204387962818146,-0.4287046492099762,0.02072528563439846,-0.0630713552236557,0.07596904784440994,-0.09919929504394531,-0.9406893849372864,0.07615096122026443,-0.3060842752456665,0.002278343541547656,0.01125246100127697,-0.028724605217576027,-0.3018989562988281,0.04720645770430565,0.9516705274581909,-0.00036276670289225876";
        ] );
    ( "tri.val",
      false,
      [
        "float32|6|-0.8676167726516724,3.0660367012023926,-0.07734505832195282,-2.016660690307617,-0.6486005783081055,0.6780397295951843";
        "float32|5|-0.5000084042549133,1.3604462146759033,1.0023982524871826,-0.1523386389017105,-0.4722159504890442";
      ],
      Some
        [
          "float32|6|-2.510044813156128,-0.9527393579483032,-0.7933416366577148,-0.0837143063545227,0.8299175500869751,3.643775701522827";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|2|-1.0048010070276965,-0.6999665423133247";
        "float64|1|-1.473143074665625";
      ],
      Some
        [
          "float64|2|-2.33339070617175,0.6286231568307286";
          "float64|2,2|-0.742601709388103,0.6697333060359677,-0.6697333060359677,-0.7426017093881031";
        ] );
    ( "tri.val",
      true,
      [
        "float64|2|-1.0048010070276965,-0.6999665423133247";
        "float64|1|-1.473143074665625";
      ],
      Some [ "float64|2|-2.33339070617175,0.6286231568307286" ] );
    ( "tri.vec",
      false,
      [
        "float32|7|1.5907007455825806,-1.2561380863189697,-1.1816829442977905,-1.7685118913650513,-0.9638542532920837,-3.106336832046509,-1.1422789096832275";
        "float32|6|1.2969154119491577,-0.34567251801490784,0.8545842170715332,-0.4889690577983856,1.760667324066162,0.1992179900407791";
      ],
      Some
        [
          "float32|7|-4.133729934692383,-2.45770001411438,-1.7689142227172852,-1.1398863792419434,-0.5921649932861328,0.16627231240272522,2.0980212688446045";
          "float32|7,7|-0.0010200988035649061,0.0809023454785347,0.33787721395492554,0.011507122777402401,0.11336954683065414,-0.02317703515291214,0.9304678440093994,0.004502594470977783,-0.2525416314601898,-0.8752593994140625,-0.02422763779759407,-0.19081467390060425,0.025455810129642487,0.3639753758907318,0.03365510702133179,-0.5743036270141602,-0.030705802142620087,0.051321033388376236,0.791866660118103,-0.19170531630516052,-0.0407705195248127,-0.11443578451871872,0.7553669214248657,-0.3329358398914337,-0.007289842702448368,0.46907052397727966,-0.2920844554901123,-0.009243015199899673,-0.4947234094142914,0.060942962765693665,-0.053939443081617355,0.09906705468893051,0.2554907500743866,0.8206900358200073,0.0018336180364713073,0.8589106202125549,0.15807168185710907,-0.06779862195253372,-0.011929278261959553,0.18420524895191193,0.4456624388694763,0.0006217880290932953,-0.05719982460141182,-0.023939648643136024,0.02155432291328907,-0.9932961463928223,0.06670799106359482,0.06784907728433609,3.8228361518122256e-05";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|1.5907007455825806,-1.2561380863189697,-1.1816829442977905,-1.7685118913650513,-0.9638542532920837,-3.106336832046509,-1.1422789096832275";
        "float32|6|1.2969154119491577,-0.34567251801490784,0.8545842170715332,-0.4889690577983856,1.760667324066162,0.1992179900407791";
      ],
      Some
        [
          "float32|7|-4.133729934692383,-2.45770001411438,-1.7689142227172852,-1.1398863792419434,-0.5921649932861328,0.16627231240272522,2.0980212688446045";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|7|-0.3820022940635681,2.5524239540100098,-0.3244718611240387,-1.2212233543395996,0.20191000401973724,-0.038835037499666214,1.0663245916366577";
        "float32|6|-0.9216338992118835,0.804716944694519,0.852748453617096,-0.6676872968673706,0.16324400901794434,-0.8307519555091858";
      ],
      Some
        [
          "float32|7|-1.9344840049743652,-0.7032642364501953,-0.507811427116394,-0.11807797849178314,0.5926902294158936,1.5167958736419678,3.008277654647827";
          "float32|7,7|0.05903836712241173,0.8838956952095032,-0.14851775765419006,-0.346698135137558,0.0909499004483223,-0.0018684696406126022,-0.25438234210014343,0.09944946318864822,0.3081071674823761,-0.02027365192770958,0.09928242117166519,-0.09618590772151947,0.003849518485367298,0.9357591867446899,-0.4868902862071991,-0.2342098206281662,-0.09299775213003159,-0.7265444993972778,0.3384062647819519,-0.007094066124409437,0.23874478042125702,0.82541424036026,-0.1867164820432663,0.039126090705394745,-0.2695384919643402,0.45473650097846985,-0.018950311467051506,0.05002089962363243,0.2599131166934967,-0.15427961945533752,-0.16057923436164856,-0.48259052634239197,-0.803186297416687,0.06865018606185913,-0.011943038552999496,-0.025472916662693024,0.09177488088607788,0.8581663370132446,-0.15647880733013153,-0.06277486681938171,0.47544941306114197,-0.0007243052823469043,-0.007051990367472172,0.043084681034088135,0.4528982639312744,-0.10975582152605057,-0.11010678112506866,-0.8768165111541748,0.0003098519810009748";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|-0.3820022940635681,2.5524239540100098,-0.3244718611240387,-1.2212233543395996,0.20191000401973724,-0.038835037499666214,1.0663245916366577";
        "float32|6|-0.9216338992118835,0.804716944694519,0.852748453617096,-0.6676872968673706,0.16324400901794434,-0.8307519555091858";
      ],
      Some
        [
          "float32|7|-1.9344840049743652,-0.7032642364501953,-0.507811427116394,-0.11807797849178314,0.5926902294158936,1.5167958736419678,3.008277654647827";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|4|-0.7041395622801669,-0.4530744436687142,-1.0658380219633747,-0.3461212751731734";
        "float64|3|-0.005876026696904225,0.7677891353446865,-0.6104866460656994";
      ],
      Some
        [
          "float64|4|-1.7734341202185955,-0.7041909855596682,-0.3894402156081696,0.297892018301004";
          "float64|4,4|-0.002590993055048632,-0.9999168808327585,-0.012137023759701503,-0.003494400511147659,-0.47149798942870985,-0.008750641869650808,0.6500163536923471,0.595895806466247,0.8108107804783826,-0.004790517593315984,0.05378035389856363,0.5828126652156501,0.34679793969934286,-0.00816753535311739,0.7579176116975429,-0.5524720574519939";
        ] );
    ( "tri.val",
      true,
      [
        "float64|4|-0.7041395622801669,-0.4530744436687142,-1.0658380219633747,-0.3461212751731734";
        "float64|3|-0.005876026696904225,0.7677891353446865,-0.6104866460656994";
      ],
      Some
        [
          "float64|4|-1.7734341202185955,-0.7041909855596682,-0.3894402156081696,0.297892018301004";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|4|-0.18577395379543304,-1.4164893627166748,-0.8274022340774536,2.755807638168335";
        "float32|3|1.0412431955337524,-0.7814240455627441,-1.3373972177505493";
      ],
      Some
        [
          "float32|4|-2.402616262435913,-0.9722081422805786,0.4865589141845703,3.214407205581665";
          "float32|4,4|-0.3653896450996399,0.44392701983451843,0.8179831504821777,0.017966724932193756,0.7779271006584167,-0.33529090881347656,0.528173565864563,0.058670371770858765,0.4948340654373169,0.7821609377861023,-0.19633443653583527,-0.3237534165382385,0.1282929927110672,0.2805942893028259,-0.11571104824542999,0.9441496729850769";
        ] );
    ( "tri.val",
      false,
      [
        "float32|4|-0.18577395379543304,-1.4164893627166748,-0.8274022340774536,2.755807638168335";
        "float32|3|1.0412431955337524,-0.7814240455627441,-1.3373972177505493";
      ],
      Some
        [
          "float32|4|-2.402616262435913,-0.9722081422805786,0.4865589141845703,3.214407205581665";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|6|-0.021690860390663147,0.0347277894616127,-0.744360625743866,-1.2865744829177856,1.4223785400390625,0.45168545842170715";
        "float32|5|-0.37456801533699036,-0.2206612080335617,-0.5295304656028748,-2.9360454082489014,0.11566182971000671";
      ],
      Some
        [
          "float32|6|-3.2469122409820557,-0.762093186378479,-0.31544163823127747,0.4059283435344696,0.4498356580734253,3.3248484134674072";
          "float32|6,6|0.0014029970625415444,0.1674240380525589,-0.7399082779884338,-0.6506285071372986,-0.03442680090665817,0.00053745589684695,0.012080519460141659,0.3309441804885864,-0.58026522397995,0.7427793145179749,0.0433383472263813,-0.004801844712346792,0.1772780865430832,0.9108601808547974,0.33515051007270813,-0.14508920907974243,-0.023089347407221794,0.0706845298409462,0.8327789306640625,-0.10740618407726288,-0.02966843545436859,0.005650176201015711,0.03401150554418564,-0.5411785840988159,0.5240564942359924,-0.14509165287017822,-0.05063287541270256,0.022910479456186295,-0.015950504690408707,0.8372396230697632,-0.016388194635510445,0.013825888745486736,0.007634053472429514,-0.057911794632673264,0.9974929690361023,0.03370385244488716";
        ] );
    ( "tri.val",
      false,
      [
        "float32|6|-0.021690860390663147,0.0347277894616127,-0.744360625743866,-1.2865744829177856,1.4223785400390625,0.45168545842170715";
        "float32|5|-0.37456801533699036,-0.2206612080335617,-0.5295304656028748,-2.9360454082489014,0.11566182971000671";
      ],
      Some
        [
          "float32|6|-3.2469122409820557,-0.762093186378479,-0.31544163823127747,0.4059283435344696,0.4498356580734253,3.3248484134674072";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|8|-1.0,-1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0";
        "float64|7|0.0,2.0,1.0,1.0,0.0,2.0,2.0";
      ],
      Some
        [
          "float64|8|-3.140732066483598,-2.6038754716096766,-1.3994474441776075,-1.0000000000000004,0.1099162641747422,0.5804909718503346,1.959688538810871,3.493959207434934";
          "float64|8,8|-1.3605913408560436e-50,-6.215251753042213e-49,-1.7961700868754262e-45,-1.0,1.0058080882822614e-48,-2.4272841921140185e-47,7.988394634830973e-50,2.0648851828679425e-49,-0.6667995944995272,4.744253861165731e-48,-0.18953730993487267,-9.473407254975464e-46,-7.617340546562064e-48,-0.6435793862620168,-0.32443717804302863,4.642517837581594e-48,0.713719636931699,-4.806392295317118e-48,0.03785509701489193,3.534581204236927e-46,-6.027332948260504e-48,-0.5085857048280483,-0.4801164987090468,6.806067411978542e-48,-0.1942833242596649,4.757956703458109e-49,0.36395349811805133,1.9013316794106843e-45,5.1702527077626906e-48,0.4833436576311643,-0.7721209424371128,1.037614533052656e-47,0.09075555381332608,1.3059779825274214e-48,-0.9111423878737497,-5.101521782011325e-45,3.0936852601100366e-48,0.3058186767528938,-0.2608791203236986,3.4905613064373116e-48,4.497451358875643e-49,0.7369762290995782,5.219395796239767e-48,-7.614010990256223e-47,0.5910090485061036,1.6052073657519785e-45,-3.466532165339499e-49,0.3279852776056818,-3.565420636190701e-49,-0.5910090485061036,-2.0634747187415374e-47,-7.028535602376766e-47,0.32798527760568175,8.934035417652628e-46,-4.670748746841212e-50,0.7369762290995783,1.9724424914799136e-49,0.32798527760568175,2.3083611657477033e-47,1.2273805129766402e-46,-0.7369762290995783,-2.014286484238365e-45,6.910330561315141e-49,0.5910090485061034";
        ] );
    ( "tri.val",
      true,
      [
        "float64|8|-1.0,-1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0";
        "float64|7|0.0,2.0,1.0,1.0,0.0,2.0,2.0";
      ],
      Some
        [
          "float64|8|-3.140732066483598,-2.6038754716096766,-1.3994474441776075,-1.0000000000000004,0.1099162641747422,0.5804909718503346,1.959688538810871,3.493959207434934";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|7|0.5583399534225464,-3.77227520942688,0.2606297433376312,-0.02544531598687172,-0.14704550802707672,-0.6305779814720154,0.05536497384309769";
        "float32|6|0.4121173918247223,-0.2637881636619568,-0.4633241295814514,1.229753851890564,-1.1053670644760132,1.0301551818847656";
      ],
      Some
        [
          "float32|7|-3.828282594680786,-2.12030291557312,-0.8971747756004333,0.19572566449642181,0.5972186923027039,0.72051602602005,1.6312899589538574";
          "float32|7,7|0.09333384037017822,-0.0016351882368326187,0.006036902777850628,0.05610136687755585,0.9899981021881104,-0.08940931409597397,-0.0034614792093634605,-0.9934555292129517,0.010628246702253819,-0.02132110856473446,-0.049362506717443466,0.09339553862810135,-0.03518429398536682,-0.009011982940137386,-0.06511452049016953,-0.06911401450634003,0.24181616306304932,0.8301768898963928,-0.0003632081497926265,0.4595677852630615,0.1791979819536209,-0.009034210816025734,-0.3612145185470581,0.61641526222229,0.1443977653980255,-0.05290979892015457,-0.43612614274024963,-0.5249938368797302,0.0034043423365801573,0.5892811417579651,-0.34584811329841614,0.3387487828731537,-0.026926780119538307,-0.091404028236866,-0.6397614479064941,0.0012867533368989825,0.6500994563102722,0.45107942819595337,0.05560176447033882,-0.04073342680931091,-0.4134637415409088,0.4451891779899597,-0.00034131715074181557,-0.3078150749206543,-0.4878346025943756,0.40807992219924927,-0.07744109630584717,-0.6403533816337585,0.2910124957561493";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|0.5583399534225464,-3.77227520942688,0.2606297433376312,-0.02544531598687172,-0.14704550802707672,-0.6305779814720154,0.05536497384309769";
        "float32|6|0.4121173918247223,-0.2637881636619568,-0.4633241295814514,1.229753851890564,-1.1053670644760132,1.0301551818847656";
      ],
      Some
        [
          "float32|7|-3.828282594680786,-2.12030291557312,-0.8971747756004333,0.19572566449642181,0.5972186923027039,0.72051602602005,1.6312899589538574";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|5|0.17681246995925903,-0.8043056726455688,-0.2899819016456604,-0.9199935793876648,0.6750645041465759";
        "float32|4|0.34790176153182983,-0.5567961931228638,-1.1022182703018188,0.3017160892486572";
      ],
      Some
        [
          "float32|5|-1.8906742334365845,-0.9265164136886597,0.24009691178798676,0.5541958808898926,0.860493540763855";
          "float32|5,5|0.055373407900333405,0.271306574344635,-0.916419267654419,0.2721392810344696,0.09717167913913727,-0.3290693163871765,-0.8604163527488708,-0.16669981181621552,0.29520073533058167,0.19095748662948608,-0.6074504852294922,-0.019332079216837883,-0.2599194347858429,-0.5502065420150757,-0.5102400183677673,-0.7159351110458374,0.423483282327652,0.20921051502227783,0.27227407693862915,0.4361153542995453,0.08418983966112137,-0.07977849990129471,-0.145119309425354,-0.6796591281890869,0.709613561630249";
        ] );
    ( "tri.val",
      false,
      [
        "float32|5|0.17681246995925903,-0.8043056726455688,-0.2899819016456604,-0.9199935793876648,0.6750645041465759";
        "float32|4|0.34790176153182983,-0.5567961931228638,-1.1022182703018188,0.3017160892486572";
      ],
      Some
        [
          "float32|5|-1.8906742334365845,-0.9265164136886597,0.24009691178798676,0.5541958808898926,0.860493540763855";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|4|-0.11383634938188146,0.41835250088922316,-0.37602715054974273,0.06756920600942723";
        "float64|3|-0.2912829113936885,0.2940421429749802,-1.509772513035727";
      ],
      Some
        [
          "float64|4|-1.7042742875162336,-0.2404246347318149,0.5289294223019163,1.4118277069131582";
          "float64|4,4|-0.01969024393964986,-0.9143719634689573,0.4025171819682303,0.03893744901602644,-0.10751097900972743,-0.39737579686281393,-0.8882232941365857,-0.20394422083787025,0.7565930864946827,-0.015504030638142482,0.06471650309882156,-0.650490815253437,0.6446864238949168,-0.07600008897060019,-0.2117807172529342,0.7305835538265812";
        ] );
    ( "tri.val",
      true,
      [
        "float64|4|-0.11383634938188146,0.41835250088922316,-0.37602715054974273,0.06756920600942723";
        "float64|3|-0.2912829113936885,0.2940421429749802,-1.509772513035727";
      ],
      Some
        [
          "float64|4|-1.7042742875162336,-0.2404246347318149,0.5289294223019163,1.4118277069131582";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|3|0.6441905498504639,-0.2299324870109558,0.35860806703567505";
        "float32|2|-0.34031012654304504,0.3203611969947815";
      ],
      Some
        [
          "float32|3|-0.4601510763168335,0.4412516951560974,0.7917655110359192";
          "float32|3,3|-0.2758380174636841,0.38635361194610596,0.8801387548446655,-0.8951228857040405,0.23039618134498596,-0.3816708028316498,0.350240558385849,0.8931115865707397,-0.28228193521499634";
        ] );
    ( "tri.val",
      false,
      [
        "float32|3|0.6441905498504639,-0.2299324870109558,0.35860806703567505";
        "float32|2|-0.34031012654304504,0.3203611969947815";
      ],
      Some
        [
          "float32|3|-0.4601510763168335,0.4412516951560974,0.7917655110359192";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|7|1.1891543865203857,-1.7035446166992188,-1.0392388105392456,0.23566606640815735,1.4628422260284424,0.2781262993812561,-0.24790845811367035";
        "float32|6|-1.4250901937484741,-0.1912766695022583,-0.019877010956406593,1.6905686855316162,0.6221252083778381,-1.5290929079055786";
      ],
      Some
        [
          "float32|7|-2.312299966812134,-1.6426677703857422,-1.0184478759765625,-0.896121621131897,1.4707510471343994,1.7751500606536865,2.7987329959869385";
          "float32|7,7|0.37337273359298706,-0.0008489604806527495,0.07555346935987473,0.009205443784594536,-0.0004938510828651488,-0.9245532155036926,0.00014188016939442605,0.917378842830658,-0.0016869845567271113,0.11703962087631226,0.01346994936466217,9.758461965247989e-05,0.3801753520965576,-0.0001602475967956707,0.1378602832555771,0.006862017326056957,-0.9821059703826904,-0.12544412910938263,0.0020599474664777517,-0.02583889849483967,0.002714848145842552,0.0015806397423148155,0.22455184161663055,-0.09901515394449234,0.7735934257507324,-0.26106104254722595,0.0001023630757117644,-0.522657036781311,-0.0007613804773427546,-0.24941127002239227,0.06190517172217369,-0.5193736553192139,-0.19070014357566833,-0.0002105881430907175,-0.7923669815063477,0.0003249174915254116,0.6348060965538025,0.022161465138196945,-0.13281720876693726,0.706985354423523,-0.0003838779521174729,-0.2811783254146576,0.00024066604964900762,0.6959462761878967,0.04397819936275482,-0.31330716609954834,-0.6290054321289062,0.00029014734900556505,0.14112189412117004";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|1.1891543865203857,-1.7035446166992188,-1.0392388105392456,0.23566606640815735,1.4628422260284424,0.2781262993812561,-0.24790845811367035";
        "float32|6|-1.4250901937484741,-0.1912766695022583,-0.019877010956406593,1.6905686855316162,0.6221252083778381,-1.5290929079055786";
      ],
      Some
        [
          "float32|7|-2.312299966812134,-1.6426677703857422,-1.0184478759765625,-0.896121621131897,1.4707510471343994,1.7751500606536865,2.7987329959869385";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|6|2.0267790952431928,-0.39500987549879313,-0.8794597147043167,1.4748226520869099,-0.049755760296968106,-0.3674025993780988";
        "float64|5|0.2187859889352313,0.8448887803757161,0.9933362044496503,-1.3752024000527405,1.9984814702717295";
      ],
      Some
        [
          "float64|6|-2.505875976462279,-1.6679862320526433,0.011320732223131354,1.051058491703536,2.048340927854523,2.873115854185658";
          "float64|6,6|0.0043416119951933275,0.030896002122797746,0.08853770497301404,-0.04601450009282192,-0.9944280338573822,0.013535638735274622,-0.08994648023469211,-0.5217586279218515,-0.8156100799639874,0.20521101931913496,-0.09800303444657255,0.052360339312311835,0.22359758650179157,0.7781222807829926,-0.4151767154967809,0.3631442370562387,-0.025907400645037448,0.1990306510405174,-0.2895977993182571,-0.1739008812602941,0.3214103181869924,0.5312159880946968,0.006996583795075309,0.7073525480958031,-0.6767687321092256,0.16463016527418106,0.04215710718729351,0.4259988313428731,-0.02163130874202393,-0.5754657699735751,0.6324650964903447,-0.2529713018697922,0.22245842947636246,0.6001932489716398,-0.017895016259521546,-0.35489928372513";
        ] );
    ( "tri.val",
      true,
      [
        "float64|6|2.0267790952431928,-0.39500987549879313,-0.8794597147043167,1.4748226520869099,-0.049755760296968106,-0.3674025993780988";
        "float64|5|0.2187859889352313,0.8448887803757161,0.9933362044496503,-1.3752024000527405,1.9984814702717295";
      ],
      Some
        [
          "float64|6|-2.505875976462279,-1.6679862320526433,0.011320732223131354,1.051058491703536,2.048340927854523,2.873115854185658";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|7|-0.3792010545730591,-0.8186598420143127,-0.969012439250946,0.1233784407377243,-0.6480162739753723,-0.7648739814758301,0.8112543821334839";
        "float32|6|0.3645673394203186,-0.3945712447166443,0.734247624874115,1.36737859249115,-1.0944963693618774,-0.6032918691635132";
      ],
      Some
        [
          "float32|7|-2.3614466190338135,-1.457890272140503,-0.8753684163093567,-0.36176353693008423,-0.10505855083465576,0.9634436368942261,1.5529531240463257";
          "float32|7,7|-0.012645255774259567,0.18092408776283264,-0.47214606404304504,-0.4750286042690277,-0.7200123071670532,0.008654188364744186,-0.0063199251890182495,0.06875548511743546,-0.5353218913078308,0.6425793170928955,-0.02272116206586361,-0.5414249897003174,0.03187202662229538,-0.03349468857049942,0.2571525573730469,-0.7000892162322998,-0.3438904285430908,-0.4125964641571045,0.31393197178840637,-0.13595572113990784,0.19548407196998596,-0.45071861147880554,0.17846229672431946,0.3014514744281769,-0.35344198346138,0.0784367099404335,-0.3406924903392792,0.6534417271614075,0.6809695959091187,0.16955216228961945,-0.03552268445491791,0.34695401787757874,-0.18167729675769806,-0.13630369305610657,0.5781939029693604,0.5029629468917847,0.3484172523021698,0.36923110485076904,-0.5323047637939453,0.18811915814876556,-0.2249506711959839,-0.3463555574417114,0.09563884884119034,0.09263283014297485,0.13207107782363892,-0.2737683355808258,0.12385587394237518,0.8917243480682373,0.28172290325164795";
        ] );
    ( "tri.val",
      false,
      [
        "float32|7|-0.3792010545730591,-0.8186598420143127,-0.969012439250946,0.1233784407377243,-0.6480162739753723,-0.7648739814758301,0.8112543821334839";
        "float32|6|0.3645673394203186,-0.3945712447166443,0.734247624874115,1.36737859249115,-1.0944963693618774,-0.6032918691635132";
      ],
      Some
        [
          "float32|7|-2.3614466190338135,-1.457890272140503,-0.8753684163093567,-0.36176353693008423,-0.10505855083465576,0.9634436368942261,1.5529531240463257";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|5|0.9426218271255493,0.7189405560493469,0.2266988605260849,1.1624212265014648,-1.0882117748260498";
        "float32|4|-1.4791377782821655,-0.866507351398468,0.12254409492015839,-0.7960867881774902";
      ],
      Some
        [
          "float32|5|-1.3428916931152344,-1.0096979141235352,0.41107437014579773,1.4253236055374146,2.4786620140075684";
          "float32|5,5|0.021269138902425766,-0.5261260867118835,0.5228323340415955,0.05781271681189537,-0.6678645014762878,0.032864347100257874,-0.694435715675354,0.18788664042949677,-0.018866604194045067,0.6935571432113647,0.041893210262060165,-0.4872610867023468,-0.8257253766059875,-0.08330676704645157,-0.2684382200241089,-0.30420053005218506,0.0058288853615522385,0.08618686348199844,-0.9482432007789612,-0.028892142698168755,-0.9508809447288513,-0.059101514518260956,-0.04576326161623001,0.30032750964164734,0.00644840719178319";
        ] );
    ( "tri.val",
      false,
      [
        "float32|5|0.9426218271255493,0.7189405560493469,0.2266988605260849,1.1624212265014648,-1.0882117748260498";
        "float32|4|-1.4791377782821655,-0.866507351398468,0.12254409492015839,-0.7960867881774902";
      ],
      Some
        [
          "float32|5|-1.3428916931152344,-1.0096979141235352,0.41107437014579773,1.4253236055374146,2.4786620140075684";
        ] );
    ( "tri.vec",
      true,
      [
        "float64|4|-0.9750001191003014,-0.6204291445949027,-1.0047872189929943,0.36746620574561295";
        "float64|3|0.7948910679832192,-0.48044593951821124,-0.2072657954706782";
      ],
      Some
        [
          "float64|4|-1.7516607321010693,-1.0191694282418147,0.1281887069172397,0.40989117648305895";
          "float64|4,4|-0.6487434451280102,-0.523586345171643,0.544100364212264,-0.09457312100868322,0.6338648175425056,0.029093857100240608,0.7551291821079008,-0.16476910781420703,0.41912631017327046,-0.8421201699989579,-0.2764154203450377,0.19687882262178902,0.040993555661665675,-0.12587496140916232,-0.23943522587249041,-0.9618449953583192";
        ] );
    ( "tri.val",
      true,
      [
        "float64|4|-0.9750001191003014,-0.6204291445949027,-1.0047872189929943,0.36746620574561295";
        "float64|3|0.7948910679832192,-0.48044593951821124,-0.2072657954706782";
      ],
      Some
        [
          "float64|4|-1.7516607321010693,-1.0191694282418147,0.1281887069172397,0.40989117648305895";
        ] );
    ( "tri.vec",
      false,
      [
        "float32|8|-1.0,1.0,0.0,2.0,-1.0,0.0,0.0,-0.0";
        "float32|7|2.0,0.0,3.0,0.0,2.0,0.0,2.0";
      ],
      Some
        [
          "float32|8|-2.5615527629852295,-2.2360680103302,-2.162277936935425,-2.000000238418579,1.5615530014038086,1.999999761581421,2.2360680103302,4.1622772216796875";
          "float32|8,8|-2.1172760528151648e-20,-0.8506508469581604,-3.1719471433097325e-17,1.311788923910588e-18,1.3046249000333737e-20,6.925230405302107e-20,-0.525731086730957,1.862486814249073e-21,1.3098448620629808e-20,0.525731086730957,1.9603988611604644e-17,-8.103930854188405e-19,2.045997640937529e-20,1.1195317026824978e-19,-0.8506507873535156,3.018015720823388e-21,6.092274068745413e-21,-5.4560184694397464e-17,-0.8112421631813049,3.73772064975865e-18,-2.8235368952261655e-22,4.470597417758236e-24,-4.239332446205965e-22,0.5847102999687195,-4.387222585623126e-21,3.932484791759568e-17,0.5847102403640747,-2.6941547588405785e-18,-1.5992434505493908e-22,-5.046686838509127e-23,-2.8226061655757496e-22,0.8112421035766602,0.7882053852081299,-2.989487085445872e-19,-2.772693645147322e-19,7.902981153794514e-20,-0.615412175655365,5.083925392406787e-20,-1.0304597311000514e-19,2.409074676768329e-21,-0.615412175655365,2.3452643487288546e-19,2.1537963353533224e-19,-6.077276573381488e-20,-0.7882054448127747,6.530085580450672e-20,-1.3281054233847144e-19,2.9913759006758487e-21,-1.975808619464277e-20,-1.0137538880513539e-18,-1.1209138542776491e-17,0.7071068286895752,-2.0398233838996742e-19,0.7071068286895752,-4.403305739385156e-19,-2.0012009236167025e-21,1.973510931442455e-20,1.0140639773834847e-18,1.1208222026657783e-17,-0.7071068286895752,-2.0381012972619154e-19,0.7071068286895752,-4.399032317545554e-19,-1.6770197336847036e-21";
        ] );
    ( "tri.val",
      false,
      [
        "float32|8|-1.0,1.0,0.0,2.0,-1.0,0.0,0.0,-0.0";
        "float32|7|2.0,0.0,3.0,0.0,2.0,0.0,2.0";
      ],
      Some
        [
          "float32|8|-2.5615527629852295,-2.2360680103302,-2.162277936935425,-2.000000238418579,1.5615530014038086,1.999999761581421,2.2360680103302,4.1622772216796875";
        ] );
    ( "tri.sel",
      false,
      [ "float32|4|1.0,2.0,3.0,4.0"; "float32|3|0.5,0.25,0.125" ],
      Some [ "float32|2|2.1446197032928467,3.05082106590271" ] );
    ( "tri.selv",
      false,
      [ "float32|4|1.0,2.0,3.0,4.0"; "float32|3|0.5,0.25,0.125" ],
      Some
        [
          "float32|2|2.1446197032928467,3.05082106590271";
          "float32|4,2|0.38635411858558655,0.06273072212934494,0.8844571113586426,0.25729891657829285,-0.2610684931278229,0.9560389518737793,0.01758861169219017,-0.1259033977985382";
        ] );
    ( "tri.tol",
      false,
      [ "float32|4|1.0,2.0,3.0,4.0"; "float32|3|0.5,0.25,0.125" ],
      Some
        [
          "float32|4|0.8115200400352478,2.170897960662842,3.077150344848633,4.040042877197266";
        ] );
    ( "tri.key",
      false,
      [ "float32|4|1.0,2.0,3.0,4.0"; "float32|3|0.5,0.25,0.125" ],
      Some
        [
          "float32|4|0.7886732816696167,2.1446197032928467,3.05082106590271,4.015886306762695";
          "float32|4,4|0.9202108383178711,0.38635411858558655,0.06273072212934494,-0.0027029537595808506,-0.38893070816993713,0.8844571113586426,0.25729891657829285,-0.01630360260605812,0.044067252427339554,-0.2610684931278229,0.9560389518737793,-0.1260589212179184,-0.001715305377729237,0.01758861169219017,-0.1259033977985382,-0.9918851256370544";
        ] );
    ( "tri.cluster",
      false,
      [ "float32|4|1.0,1.0,1.0,1.0"; "float32|3|0.0,9.999999717180685e-10,0.0" ],
      Some
        [
          "float32|4|1.0,1.0,1.0,1.0";
          "float32|4,4|-0.18128538131713867,0.9834303855895996,-5.556888283515785e-21,-2.2592356343967147e-21,3.2084211463850875e-21,-2.671038495457968e-21,-0.8137047290802002,0.5812784433364868,5.068449844809943e-21,-4.219528680092905e-21,-0.5812784433364868,-0.8137047290802002,0.9834303855895996,0.18128560483455658,4.626158229203043e-21,1.8808334217415576e-21";
        ] );
    ( "tri.one",
      false,
      [ "float32|1|3.0"; "float32|0|" ],
      Some [ "float32|1|3.0"; "float32|1,1|1.0" ] );
    ( "tri.complex",
      false,
      [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0"; "complex64|2|0.5,0.0|0.5,0.25" ],
      Some
        [
          "float32|3|0.628338098526001,2.2967190742492676,3.0749425888061523";
          "complex64|3,3|-0.7042680382728577,0.1530490517616272,0.05434533953666687,0.4603932201862335,-0.3599337637424469,-0.0549260638654232,0.006652067881077528,0.2690451443195343,-0.9355599284172058|-0.5344743132591248,-0.4306216537952423,-0.08081646263599396,-0.06310582160949707,-0.7568570971488953,-0.28045299649238586,0.04853065311908722,-0.1279481053352356,0.18322722613811493";
        ] );
    ( "tri.bad_dt",
      false,
      [ "float32|4|1.0,2.0,3.0,4.0"; "float16|3|0.5,0.25,0.125" ],
      None );
    ("tri.int", false, [ "int32|2|1.0,2.0"; "int32|1|1.0" ], None);
    ( "ts.real",
      false,
      [
        "float32|4|0.0,1.0,2.0,1.0";
        "float32|4|4.0,4.0,1.0,3.0";
        "float32|4|1.0,2.0,1.0,0.0";
        "float32|4,2|1.0,2.0,2.0,3.0,3.0,1.0,4.0,5.0";
      ],
      Some
        [
          "float32|4,2|-0.11111107468605042,0.9999999403953552,1.4444442987442017,-1.999999761581421,-1.833332896232605,4.999999523162842,1.9444442987442017,1.7384685691013146e-07";
        ] );
    ( "ts.perturb",
      false,
      [
        "float32|4|0.0,1.0,2.0,1.0";
        "float32|4|4.0,4.0,1.0,3.0";
        "float32|4|1.0,2.0,1.0,0.0";
        "float32|4,2|1.0,2.0,2.0,3.0,3.0,1.0,4.0,5.0";
      ],
      Some
        [
          "float32|4,2|-0.11111107468605042,1.0,1.4444442987442017,-1.9999998807907104,-1.8333330154418945,4.999999523162842,1.9444442987442017,1.9868211609264108e-07";
        ] );
    ( "ts.singular",
      false,
      [
        "float32|2|0.0,1.0";
        "float32|2|1.0,1.0";
        "float32|2|1.0,0.0";
        "float32|2,1|1.0,1.0";
      ],
      Some [ "float32|2,1|nan,nan" ] );
    ( "ts.perturb",
      false,
      [
        "float32|3|0.0,1.0,1.0";
        "float32|3|1.0,2.0,1.0";
        "float32|3|1.0,1.0,0.0";
        "float32|3,1|1.0,2.0,3.0";
      ],
      Some [ "float32|3,1|-16777220.0,16777220.0,-16777216.0" ] );
    ( "ts.perturb",
      false,
      [
        "float32|2|0.0,1.0";
        "float32|2|1.0,1.0";
        "float32|2|1.0,0.0";
        "float32|2,1|1.0,0.0";
      ],
      Some [ "float32|2,1|8388609.0,-8388608.0" ] );
    ( "ts.perturb",
      false,
      [
        "float32|2|0.0,2.0";
        "float32|2|1.0,4.0";
        "float32|2|2.0,0.0";
        "float32|2,1|3.0,1.0";
      ],
      Some [ "float32|2,1|41943044.0,-20971520.0" ] );
    ( "ts.perturb",
      false,
      [
        "float32|3|0.0,1.0,0.0";
        "float32|3|0.0,0.0,0.0";
        "float32|3|1.0,0.0,0.0";
        "float32|3,1|1.0,2.0,3.0";
      ],
      Some [ "float32|3,1|2.0,1.0,25165824.0" ] );
    ( "ts.perturb",
      false,
      [
        "float32|2|0.0,1.0";
        "float32|2|9.99999993922529e-09,1.0";
        "float32|2|1.0,0.0";
        "float32|2,1|1.0,1.0";
      ],
      Some [ "float32|2,1|0.0,1.0" ] );
    ( "ts.perturb",
      false,
      [
        "float32|3|0.0,1.0,1.0";
        "float32|3|0.5,0.5,0.25";
        "float32|3|0.5,0.5,0.0";
        "float32|3,1|1.0,1.0,1.0";
      ],
      Some
        [
          "float32|3,1|0.7999999523162842,1.2000000476837158,-0.800000011920929";
        ] );
    ( "ts.complex",
      false,
      [
        "complex64|3|0.0,0.0,1.0|0.0,1.0,0.0";
        "complex64|3|4.0,0.0,4.0|0.0,4.0,0.0";
        "complex64|3|0.0,1.0,0.0|1.0,0.0,0.0";
        "complex64|3,1|1.0,2.0,3.0|1.0,0.0,0.0";
      ],
      Some
        [
          "complex64|3,1|0.15625,-0.0625,0.765625|0.265625,-0.375,0.0937499925494194";
        ] );
    ( "ts.complex_perturb",
      false,
      [
        "complex64|3|0.0,0.0,1.0|0.0,1.0,0.0";
        "complex64|3|4.0,0.0,4.0|0.0,4.0,0.0";
        "complex64|3|0.0,1.0,0.0|1.0,0.0,0.0";
        "complex64|3,1|1.0,2.0,3.0|1.0,0.0,0.0";
      ],
      Some
        [
          "complex64|3,1|0.15625,-0.0625,0.765625|0.265625,-0.375,0.0937499925494194";
        ] );
    ( "ll.chol",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.0,0.5,1.658312439918518" ] );
    ( "ll.chol_ns",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.0,0.0,1.7320507764816284" ] );
    ( "ll.eigh",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some
        [
          "float32|2,2|0.5257311463356018,-0.8506508469581604,-0.8506508469581604,-0.5257311463356018";
          "float32|2|2.3819661140441895,4.6180338859558105";
        ] );
    ( "ll.eigh_ns",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|0.0,1.0,1.0,0.0"; "float32|2|3.0,4.0" ] );
    ( "ll.eigh_nsu",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some
        [
          "float32|2,2|0.6154122352600098,-0.7882054448127747,-0.7882054448127747,-0.6154122352600098";
          "float32|2|1.438447117805481,5.561553001403809";
        ] );
    ( "np.chol",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.0,0.5,1.658312439918518" ] );
    ( "np.chol_ns",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.0,0.0,1.7320507764816284" ] );
    ( "np.chol_up",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.5,0.0,1.658312439918518" ] );
    ( "sp.chol",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,1.0,0.0,1.4142135381698608" ] );
    ( "sp.chol_l",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some [ "float32|2,2|2.0,0.0,0.0,1.7320507764816284" ] );
    ( "sp.eigh",
      false,
      [ "float32|2,2|4.0,2.0,0.0,3.0" ],
      Some
        [
          "float32|2|2.3819661140441895,4.6180338859558105";
          "float32|2,2|0.5257311463356018,-0.8506508469581604,-0.8506508469581604,-0.5257311463356018";
        ] );
    ( "np.pinv_rcond",
      false,
      [
        "float32|4,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0,1.0,1.0,1.0";
        "float32||0.5";
      ],
      Some
        [
          "float32|3,4|0.11896926909685135,0.1275244653224945,-0.08343124389648438,0.04850434511899948,0.12292129546403885,0.15428780019283295,-0.023802686482667923,0.06648530066013336,-0.07839664071798325,-0.017377546057105064,0.23961706459522247,0.016474446281790733";
        ] );
    ( "np.rank_tol",
      false,
      [
        "float32|4,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0,1.0,1.0,1.0";
        "float32||2.5";
      ],
      Some [ "int32||2.0" ] );
    ( "np.lstsq_res",
      false,
      [
        "float32|4,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0,1.0,1.0,1.0";
        "float32|4|2.0,1.0,0.0,1.0";
      ],
      Some
        [
          "float32|3|0.9999998211860657,1.54338053448555e-07,3.29648841557173e-09";
          "float32|1|1.417511399058602e-13";
          "int32||3.0";
          "float32|3|4.99534273147583,3.047250986099243,1.3269563913345337";
        ] );
    ( "np.lstsq_res_wide",
      false,
      [
        "float32|3,4|2.0,1.0,0.0,1.0,1.0,3.0,1.0,1.0,0.0,1.0,4.0,1.0";
        "float32|3|2.0,1.0,0.0";
      ],
      Some
        [
          "float32|4|0.8431369066238403,-0.03921554237604141,-0.07843124121427536,0.35294109582901";
          "float32|0|";
          "int32||3.0";
          "float32|3|4.995343208312988,3.0472512245178223,1.3269565105438232";
        ] );
    ( "np.lstsq",
      false,
      [
        "float32|3,4|2.0,1.0,0.0,1.0,1.0,3.0,1.0,1.0,0.0,1.0,4.0,1.0";
        "float32|3|2.0,1.0,0.0";
      ],
      Some
        [
          "float32|4|0.8431369066238403,-0.03921554237604141,-0.07843124121427536,0.35294109582901";
          "float32|1|7.247535904753022e-13";
          "int32||3.0";
          "float32|3|4.995343208312988,3.0472512245178223,1.3269565105438232";
        ] );
    ( "np.lstsq_neg",
      false,
      [
        "float32|4,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0,1.0,1.0,1.0";
        "float32|4|2.0,1.0,0.0,1.0";
      ],
      Some
        [
          "float32|3|0.9999998211860657,1.8414037583625031e-07,-4.140699516597124e-08";
          "float32|1|1.4600396358501605e-13";
          "int32||3.0";
          "float32|3|4.99534273147583,3.047250986099243,1.3269563913345337";
        ] );
  ]

let lax_scipy =
  [
    ( "e.expm_max",
      false,
      [ "float32|3,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0" ],
      Some
        [
          "float32|3,3|13.975503921508789,18.92854881286621,12.818523406982422,18.92854881286621,45.72257614135742,44.56560134887695,12.818522453308105,44.56559371948242,77.46964263916016";
        ] );
    ( "e.expm_max0",
      false,
      [ "float32|3,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0" ],
      Some [ "float32|3,3|nan,nan,nan,nan,nan,nan,nan,nan,nan" ] );
    ( "e.expm",
      false,
      [ "float32|3,3|2.0,1.0,0.0,1.0,3.0,1.0,0.0,1.0,4.0" ],
      Some
        [
          "float32|3,3|1.0202524662017822,0.010253535583615303,5.15240099048242e-05,0.010253536514937878,1.0305575132369995,0.010356583632528782,5.15240099048242e-05,0.010356584563851357,1.0408626794815063";
        ] );
    ( "e.hadamard",
      false,
      [],
      Some
        [
          "int32|4,4|1.0,1.0,1.0,1.0,1.0,-1.0,1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,-1.0,1.0";
        ] );
    ("e.hadamard_i", false, [], Some [ "int8|2,2|1.0,1.0,1.0,-1.0" ]);
    ( "e.dft_d",
      false,
      [],
      Some
        [ "complex64|2,2|1.0,1.0,1.0,-1.0|0.0,0.0,0.0,8.742277657347586e-08" ]
    );
    ( "e.dft",
      false,
      [],
      Some
        [
          "complex64|3,3|0.3333333432674408,0.3333333432674408,0.3333333432674408,0.3333333432674408,-0.1666666865348816,-0.166666641831398,0.3333333432674408,-0.166666641831398,-0.16666673123836517|0.0,0.0,0.0,0.0,-0.28867512941360474,0.2886751592159271,0.0,0.2886751592159271,-0.28867509961128235";
        ] );
    ( "e.fftc_ax",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      Some
        [
          "float32|3,5|1.9073486612342094e-07,1.0,4.828427791595459,10.853007316589355,10.392305374145508,-3.814697322468419e-07,7.999998569488525,11.180340766906738,14.696939468383789,18.520259857177734,67.88225555419922,103.6274185180664,121.86833953857422,141.0714111328125,36.48287582397461";
        ] );
    ( "e.fftc_same",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      Some
        [
          "float32|3,4|1.9073486612342094e-07,1.0,4.828427791595459,10.853007316589355,-3.814697322468419e-07,7.999998569488525,11.180340766906738,14.696939468383789,67.88225555419922,103.6274185180664,121.86833953857422,141.0714111328125";
        ] );
    ( "e.fftc_0",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|1,4|0.0,1.0,2.8284270763397217,5.196152210235596";
      ],
      Some
        [
          "float32|3,4|0.0,1.0,7.999999523162842,26.999998092651367,0.0,11.180339813232422,41.569217681884766,96.23409271240234,0.0,27.0,89.44271850585938,189.570556640625";
        ] );
    ( "e.fftc_bad",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      None );
    ( "e.conv2d_bd",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      None );
    ( "e.corr2d_fv",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      None );
    ( "e.conv2d",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
        "float32|3,2|1.0,2.0,0.0,1.0,3.0,1.0";
      ],
      Some
        [
          "float32|5,5|0.0,1.0,4.828427314758301,10.853006362915039,10.392304420471191,8.0,27.180339813232422,38.0576171875,50.7425651550293,42.236671447753906,22.627416610717773,83.25483703613281,106.28839111328125,132.84225463867188,96.68215942382812,24.0,64.1684341430664,82.27115631103516,101.8804931640625,55.00313186645508,67.88224792480469,103.6274185180664,121.86833190917969,141.07139587402344,36.482872009277344";
        ] );
    ( "e.detrend_bp",
      false,
      [ "float32|8|1.0,3.0,2.0,5.0,4.0,7.0,9.0,8.0" ],
      Some
        [
          "float32|8|-0.5,1.0,-0.5,0.599998950958252,-1.5000009536743164,0.39999866485595703,1.2999987602233887,-0.8000011444091797";
        ] );
    ( "e.detrend_bp2",
      false,
      [ "float32|8|1.0,3.0,2.0,5.0,4.0,7.0,9.0,8.0" ],
      Some
        [
          "float32|8|-4.76837158203125e-07,-4.76837158203125e-07,-0.6666667461395264,1.3333330154418945,-0.6666669845581055,-0.4999990463256836,1.0000004768371582,-0.5";
        ] );
    ( "e.detrend_ax",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
      ],
      Some
        [
          "float32|3,4|0.0,0.0,2.384185791015625e-07,-4.76837158203125e-07,-1.9073486328125e-06,-3.814697265625e-06,-2.86102294921875e-06,-3.814697265625e-06,-1.9073486328125e-06,-3.814697265625e-06,-5.7220458984375e-06,-7.62939453125e-06";
        ] );
    ( "e.detrend_bad",
      false,
      [ "float32|8|1.0,3.0,2.0,5.0,4.0,7.0,9.0,8.0" ],
      None );
    ( "e.istft",
      false,
      [
        "complex64|3,3|1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
      ],
      Some
        [
          "float32|8|0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0";
          "float32|8|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        ] );
    ( "e.istft_std",
      false,
      [
        "complex64|3,3|1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
      ],
      Some
        [
          "float32|8|0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0";
          "float32|8|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        ] );
    ( "e.istft_same",
      false,
      [
        "complex64|3,3|1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0|0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
      ],
      None );
    ( "e.dot_dn",
      false,
      [ "float32|2,2|1.0,2.0,3.0,4.0"; "float32|3,2|1.0,0.0,2.0,1.0,1.0,1.0" ],
      Some [ "float32|2,3|1.0,4.0,3.0,3.0,10.0,7.0" ] );
    ( "e.dot_dnb",
      false,
      [ "float32|2,2|1.0,2.0,3.0,4.0"; "float32|2,2|1.0,2.0,3.0,4.0" ],
      Some [ "float32|2|5.0,25.0" ] );
    ( "e.map_bs",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
      ],
      Some
        [
          "float32|3,4|9.024579048156738,11.024579048156738,14.681432723999023,19.41688346862793,68.39753723144531,74.75821685791016,81.79141235351562,89.43805694580078,162.98789978027344,171.73306274414062,180.97862243652344,190.69882202148438";
        ] );
    ( "e.map_bs1",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
      ],
      Some
        [
          "float32|3,4|0.0,2.0,5.656854152679443,10.392304420471191,16.0,22.360679626464844,29.393877029418945,37.04051971435547,45.25483322143555,54.0,63.24555206298828,72.96574401855469";
        ] );
    ( "e.assoc_rev",
      false,
      [ "float32|8|1.0,3.0,2.0,5.0,4.0,7.0,9.0,8.0" ],
      Some [ "float32|8|39.0,38.0,35.0,33.0,28.0,24.0,17.0,8.0" ] );
    ( "e.assoc_rev1",
      false,
      [
        "float32|3,4|0.0,1.0,2.8284270763397217,5.196152210235596,8.0,11.180339813232422,14.696938514709473,18.520259857177734,22.627416610717773,27.0,31.62277603149414,36.482872009277344";
      ],
      Some
        [
          "float32|3,4|9.024579048156738,9.024579048156738,8.024579048156738,5.196152210235596,52.39753723144531,44.39753723144531,33.21719741821289,18.520259857177734,117.73307037353516,95.10565185546875,68.10565185546875,36.482872009277344";
        ] );
    ( "e.gather_d",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|4,1|0.0,1.0,5.0,-1.0" ],
      Some [ "float32|4|0.0,1.0,2.0,0.0" ] );
    ( "e.gather_clip",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|4,1|0.0,1.0,5.0,-1.0" ],
      Some [ "float32|4|0.0,1.0,2.0,0.0" ] );
    ( "e.gather_fv",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|4,1|0.0,1.0,5.0,-1.0" ],
      Some [ "float32|4|0.0,1.0,9.0,9.0" ] );
    ( "e.gather_int",
      false,
      [ "int32|3|0.0,1.0,2.0"; "int32|4,1|0.0,1.0,5.0,-1.0" ],
      Some [ "int32|4|0.0,1.0,2.0,0.0" ] );
    ( "e.gather_bad",
      false,
      [ "float32|3|0.0,1.0,2.0"; "int32|4,1|0.0,1.0,5.0,-1.0" ],
      None );
    ( "e.scatter",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,2.0,2.0" ] );
    ( "e.scatter_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|4.0,2.0,3.0" ] );
    ( "e.scatter_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,2.0,2.0" ] );
    ( "e.scatter_add",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,3.0,2.0" ] );
    ( "e.scatter_add_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|5.0,3.0,5.0" ] );
    ( "e.scatter_add_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,3.0,2.0" ] );
    ( "e.scatter_sub",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|-1.0,-1.0,2.0" ] );
    ( "e.scatter_sub_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|-5.0,-1.0,-1.0" ] );
    ( "e.scatter_sub_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|-1.0,-1.0,2.0" ] );
    ( "e.scatter_mul",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,2.0,2.0" ] );
    ( "e.scatter_mul_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,2.0,6.0" ] );
    ( "e.scatter_mul_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,2.0,2.0" ] );
    ( "e.scatter_min",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,1.0,2.0" ] );
    ( "e.scatter_min_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,1.0,2.0" ] );
    ( "e.scatter_min_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|0.0,1.0,2.0" ] );
    ( "e.scatter_max",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,2.0,2.0" ] );
    ( "e.scatter_max_clip",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|4.0,2.0,3.0" ] );
    ( "e.scatter_max_drop",
      false,
      [
        "float32|3|0.0,1.0,2.0";
        "int32|4,1|0.0,1.0,5.0,-1.0";
        "float32|4|1.0,2.0,3.0,4.0";
      ],
      Some [ "float32|3|1.0,2.0,2.0" ] );
  ]

let numpy_dtypes =
  [
    ( "g.nan_to_num",
      false,
      [ "float32|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float32|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ] );
    ( "g.nanargmax",
      false,
      [ "float32|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||4.0" ] );
    ( "g.nanargmin",
      false,
      [ "float32|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||0.0" ] );
    ( "g.sort_complex",
      false,
      [ "float32|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "complex64|3|-1.5,0.5,2.5|0.0,0.0,0.0" ] );
    ( "g.sort",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|-1.5,2.5,nan,0.0,3.5,inf" ] );
    ( "g.argsort",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|2,3|0.0,2.0,1.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|-1.5,0.0,2.5,0.0,3.5,3.4028234663852886e+38" ] );
    ( "g.round",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|-2.0,nan,2.0,0.0,4.0,inf" ] );
    ( "g.all",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||1.0" ] );
    ( "g.argmin",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||1.0" ] );
    ( "g.nanargmax",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||0.0" ] );
    ( "g.unwrap",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|-1.5,nan,nan,0.0,-2.7831850051879883,nan" ] );
    ( "g.nonzero",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5|0.0,0.0,0.0,1.0,1.0"; "int32|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|6|-1.5,0.0,2.5,3.5,inf,nan" ] );
    ( "g.i0",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some
        [
          "float32|2,3|1.6467232704162598,nan,3.289839506149292,1.0000001192092896,7.378204345703125,nan";
        ] );
    ( "g.sort_complex",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "complex64|3|-1.5,2.5,nan|0.0,0.0,0.0" ] );
    ( "g.clip",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|0.0,nan,2.0,0.0,2.0,2.0" ] );
    ( "g.clip_lo",
      false,
      [ "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float32|2,3|1.0,nan,2.5,1.0,3.5,inf" ] );
    ( "g.isclose",
      false,
      [
        "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float32|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,0.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [
        "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float32|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [
        "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float32|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "float32|2,3|-3.0,nan,20.0,0.0,112.0,inf" ] );
    ( "g.frexp",
      false,
      [
        "float32|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float32|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some
        [
          "float32|2,3|-0.75,nan,0.625,0.0,0.875,inf";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,0.0";
        ] );
    ( "g.sort",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,-0.25,0.0,3.5" ] );
    ( "g.argsort",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|2,3|0.0,1.0,2.0,2.0,0.0,1.0" ] );
    ( "g.nan_to_num",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ] );
    ( "g.round",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-2.0,0.0,2.0,0.0,4.0,-0.0" ] );
    ( "g.all",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||4.0" ] );
    ( "g.argmin",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||0.0" ] );
    ( "g.nanargmax",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||4.0" ] );
    ( "g.nanargmin",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||0.0" ] );
    ( "g.unwrap",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,0.0,-2.78125,-0.25" ] );
    ( "g.nonzero",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5|0.0,0.0,0.0,1.0,1.0"; "int32|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|6|-1.5,-0.25,0.0,0.5,2.5,3.5" ] );
    ( "g.i0",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some
        [
          "float16|2,3|1.646484375,1.0634765625,3.2890625,1.0,7.37890625,1.015625";
        ] );
    ( "g.sort_complex",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "complex64|3|-1.5,0.5,2.5|0.0,0.0,0.0" ] );
    ( "g.clip",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|0.0,0.5,2.0,0.0,2.0,0.0" ] );
    ( "g.clip_lo",
      false,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|1.0,1.0,2.5,1.0,3.5,1.0" ] );
    ( "g.isclose",
      false,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,1.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "float16|2,3|-3.0,2.0,20.0,0.0,112.0,-32.0" ] );
    ( "g.frexp",
      false,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some
        [
          "float16|2,3|-0.75,0.5,0.625,0.0,0.875,-0.5";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,-1.0";
        ] );
    ( "g.sort",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,2.5,nan,0.0,3.5,inf" ] );
    ( "g.argsort",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|2,3|0.0,2.0,1.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,0.0,2.5,0.0,3.5,65504.0" ] );
    ( "g.round",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-2.0,nan,2.0,0.0,4.0,inf" ] );
    ( "g.all",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||1.0" ] );
    ( "g.argmin",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||1.0" ] );
    ( "g.nanargmax",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32||0.0" ] );
    ( "g.unwrap",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,nan,nan,0.0,-2.78125,nan" ] );
    ( "g.nonzero",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5|0.0,0.0,0.0,1.0,1.0"; "int32|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int32|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|6|-1.5,0.0,2.5,3.5,inf,nan" ] );
    ( "g.i0",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|1.646484375,nan,3.2890625,1.0,7.37890625,nan" ] );
    ( "g.sort_complex",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "complex64|3|-1.5,2.5,nan|0.0,0.0,0.0" ] );
    ( "g.clip",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|0.0,nan,2.0,0.0,2.0,2.0" ] );
    ( "g.clip_lo",
      false,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|1.0,nan,2.5,1.0,3.5,inf" ] );
    ( "g.isclose",
      false,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,0.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "float16|2,3|-3.0,nan,20.0,0.0,112.0,inf" ] );
    ( "g.frexp",
      false,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some
        [
          "float16|2,3|-0.75,nan,0.625,0.0,0.875,inf";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,0.0";
        ] );
    ( "g.sort",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|-1.5,0.5,2.5,-0.25,0.0,3.5" ] );
    ( "g.argsort",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|2,3|0.0,1.0,2.0,2.0,0.0,1.0" ] );
    ( "g.nan_to_num",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ] );
    ( "g.round",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|-2.0,0.0,2.0,0.0,4.0,-0.0" ] );
    ( "g.all",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||4.0" ] );
    ( "g.argmin",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||0.0" ] );
    ( "g.nanargmax",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||4.0" ] );
    ( "g.nanargmin",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32||0.0" ] );
    ( "g.unwrap",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,-2.8125,-0.25" ] );
    ( "g.nonzero",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5|0.0,0.0,0.0,1.0,1.0"; "int32|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int32|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|6|-1.5,-0.25,0.0,0.5,2.5,3.5" ] );
    ( "g.i0",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|1.640625,1.0625,3.28125,1.0,7.34375,1.015625" ] );
    ( "g.sort_complex",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "complex64|3|-1.5,0.5,2.5|0.0,0.0,0.0" ] );
    ( "g.clip",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|0.0,0.5,2.0,0.0,2.0,0.0" ] );
    ( "g.clip_lo",
      false,
      [ "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bfloat16|2,3|1.0,1.0,2.5,1.0,3.5,1.0" ] );
    ( "g.isclose",
      false,
      [
        "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "bfloat16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,1.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [
        "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "bfloat16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [
        "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "bfloat16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bfloat16|2,3|-3.0,2.0,20.0,0.0,112.0,-32.0" ] );
    ( "g.frexp",
      false,
      [
        "bfloat16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "bfloat16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some
        [
          "bfloat16|2,3|-0.75,0.5,0.625,0.0,0.875,-0.5";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,-1.0";
        ] );
    ( "g.sort",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2,3|0.0,1.0,1.0,0.0,1.0,1.0" ] );
    ( "g.argsort",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|2,3|1.0,0.0,2.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ("g.round", false, [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ], None);
    ( "g.all",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32||0.0" ] );
    ( "g.argmin",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32||1.0" ] );
    ( "g.nanargmax",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32||0.0" ] );
    ( "g.nanargmin",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32||1.0" ] );
    ( "g.unwrap",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "float32|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ( "g.nonzero",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|4|0.0,0.0,1.0,1.0"; "int32|4|0.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|4|0.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|4,2|0.0,0.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2|0.0,1.0" ] );
    ( "g.i0",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,3|1.2660659551620483,1.0000001192092896,1.2660659551620483,1.0000001192092896,1.2660659551620483,1.2660659551620483";
        ] );
    ( "g.sort_complex",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "complex64|3|0.0,1.0,1.0|0.0,0.0,0.0" ] );
    ( "g.clip",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ( "g.clip_lo",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int32|2,3|1.0,1.0,1.0,1.0,1.0,1.0" ] );
    ( "g.isclose",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool|2,3|1.0,1.0,1.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "float32|2,3|2.0,0.0,8.0,0.0,32.0,128.0" ] );
    ( "g.frexp",
      false,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some
        [
          "float32|2,3|0.5,0.0,0.5,0.0,0.5,0.5";
          "int32|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        ] );
    ( "g.nan_to_num",
      false,
      [ "int8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "int8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "int8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "int8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      false,
      [ "int16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "int16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "int16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "int16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      false,
      [ "int32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "int32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "int32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "int32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      false,
      [ "uint8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "uint8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "uint8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "uint8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "uint8|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      false,
      [ "uint16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "uint16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "uint16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "uint16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "uint16|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      false,
      [ "uint32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "uint32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      false,
      [ "uint32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||5.0" ] );
    ( "g.nanargmin",
      false,
      [ "uint32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int32||3.0" ] );
    ( "g.sort_complex",
      false,
      [ "uint32|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex64|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.sort",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex64|2,3|0.0,1.0,2.0,-0.0,1.0,3.0|0.5,1.0,-1.0,-1.0,0.0,0.0" ]
    );
    ( "g.argsort",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int32|2,3|2.0,0.0,1.0,1.0,0.0,2.0" ] );
    ( "g.nan_to_num",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ]
    );
    ( "g.round",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.0,0.0,-1.0,0.0" ]
    );
    ( "g.all",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.argmin",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nanargmax",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nanargmin",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.unwrap",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nonzero",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "int32|6|0.0,0.0,0.0,1.0,1.0,1.0"; "int32|6|0.0,1.0,2.0,0.0,1.0,2.0" ]
    );
    ( "g.flatnonzero",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int32|6|0.0,1.0,2.0,3.0,4.0,5.0" ] );
    ( "g.argwhere",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int32|6,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,0.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "complex64|6|-0.0,0.0,1.0,1.0,2.0,3.0|-1.0,0.5,0.0,1.0,-1.0,0.0" ]
    );
    ( "g.i0",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.sort_complex",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "complex64|3|0.0,1.0,2.0|0.5,1.0,-1.0" ] );
    ( "g.clip",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.clip_lo",
      false,
      [ "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.isclose",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex64|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      Some [ "bool|2,3|0.0,1.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex64|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex64|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      None );
    ( "g.frexp",
      false,
      [
        "complex64|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex64|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      None );
    ( "g.nan_to_num",
      true,
      [ "float64|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float64|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ] );
    ( "g.nanargmax",
      true,
      [ "float64|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||4.0" ] );
    ( "g.nanargmin",
      true,
      [ "float64|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||0.0" ] );
    ( "g.sort_complex",
      true,
      [ "float64|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "complex128|3|-1.5,0.5,2.5|0.0,0.0,0.0" ] );
    ( "g.sort",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|-1.5,2.5,nan,0.0,3.5,inf" ] );
    ( "g.argsort",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|2,3|0.0,2.0,1.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|-1.5,0.0,2.5,0.0,3.5,1.7976931348623157e+308" ] );
    ( "g.round",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|-2.0,nan,2.0,0.0,4.0,inf" ] );
    ( "g.all",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||1.0" ] );
    ( "g.argmin",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||1.0" ] );
    ( "g.nanargmax",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||5.0" ] );
    ( "g.nanargmin",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||0.0" ] );
    ( "g.unwrap",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|-1.5,nan,nan,0.0,-2.7831853071795862,nan" ] );
    ( "g.nonzero",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5|0.0,0.0,0.0,1.0,1.0"; "int64|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|6|-1.5,0.0,2.5,3.5,inf,nan" ] );
    ( "g.i0",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some
        [
          "float64|2,3|1.6467231897728907,nan,3.289839144050123,1.0,7.3782034322254795,nan";
        ] );
    ( "g.sort_complex",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "complex128|3|-1.5,2.5,nan|0.0,0.0,0.0" ] );
    ( "g.clip",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|0.0,nan,2.0,0.0,2.0,2.0" ] );
    ( "g.clip_lo",
      true,
      [ "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float64|2,3|1.0,nan,2.5,1.0,3.5,inf" ] );
    ( "g.isclose",
      true,
      [
        "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float64|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,0.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      true,
      [
        "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float64|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      true,
      [
        "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float64|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "float64|2,3|-3.0,nan,20.0,0.0,112.0,inf" ] );
    ( "g.frexp",
      true,
      [
        "float64|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float64|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some
        [
          "float64|2,3|-0.75,nan,0.625,0.0,0.875,inf";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,0.0";
        ] );
    ( "g.sort",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,-0.25,0.0,3.5" ] );
    ( "g.argsort",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64|2,3|0.0,1.0,2.0,2.0,0.0,1.0" ] );
    ( "g.nan_to_num",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ] );
    ( "g.round",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-2.0,0.0,2.0,0.0,4.0,-0.0" ] );
    ( "g.all",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||4.0" ] );
    ( "g.argmin",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||0.0" ] );
    ( "g.nanargmax",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||4.0" ] );
    ( "g.nanargmin",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64||0.0" ] );
    ( "g.unwrap",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|-1.5,0.5,2.5,0.0,-2.78125,-0.25" ] );
    ( "g.nonzero",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64|5|0.0,0.0,0.0,1.0,1.0"; "int64|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "int64|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|6|-1.5,-0.25,0.0,0.5,2.5,3.5" ] );
    ( "g.i0",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some
        [
          "float16|2,3|1.646484375,1.0634765625,3.2890625,1.0,7.37890625,1.015625";
        ] );
    ( "g.sort_complex",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "complex64|3|-1.5,0.5,2.5|0.0,0.0,0.0" ] );
    ( "g.clip",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|0.0,0.5,2.0,0.0,2.0,0.0" ] );
    ( "g.clip_lo",
      true,
      [ "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25" ],
      Some [ "float16|2,3|1.0,1.0,2.5,1.0,3.5,1.0" ] );
    ( "g.isclose",
      true,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,1.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      true,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      true,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some [ "float16|2,3|-3.0,2.0,20.0,0.0,112.0,-32.0" ] );
    ( "g.frexp",
      true,
      [
        "float16|2,3|-1.5,0.5,2.5,0.0,3.5,-0.25";
        "float16|2,3|2.5,0.5,-1.5,-0.25,3.5,0.0";
      ],
      Some
        [
          "float16|2,3|-0.75,0.5,0.625,0.0,0.875,-0.5";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,-1.0";
        ] );
    ( "g.sort",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,2.5,nan,0.0,3.5,inf" ] );
    ( "g.argsort",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|2,3|0.0,2.0,1.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,0.0,2.5,0.0,3.5,65504.0" ] );
    ( "g.round",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-2.0,nan,2.0,0.0,4.0,inf" ] );
    ( "g.all",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||1.0" ] );
    ( "g.argmin",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||1.0" ] );
    ( "g.nanargmax",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||5.0" ] );
    ( "g.nanargmin",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64||0.0" ] );
    ( "g.unwrap",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|-1.5,nan,nan,0.0,-2.78125,nan" ] );
    ( "g.nonzero",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5|0.0,0.0,0.0,1.0,1.0"; "int64|5|0.0,1.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5|0.0,1.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "int64|5,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|6|-1.5,0.0,2.5,3.5,inf,nan" ] );
    ( "g.i0",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|1.646484375,nan,3.2890625,1.0,7.37890625,nan" ] );
    ( "g.sort_complex",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "complex64|3|-1.5,2.5,nan|0.0,0.0,0.0" ] );
    ( "g.clip",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|0.0,nan,2.0,0.0,2.0,2.0" ] );
    ( "g.clip_lo",
      true,
      [ "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf" ],
      Some [ "float16|2,3|1.0,nan,2.5,1.0,3.5,inf" ] );
    ( "g.isclose",
      true,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool|2,3|0.0,0.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      true,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      true,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some [ "float16|2,3|-3.0,nan,20.0,0.0,112.0,inf" ] );
    ( "g.frexp",
      true,
      [
        "float16|2,3|-1.5,nan,2.5,0.0,3.5,inf";
        "float16|2,3|2.5,nan,-1.5,inf,3.5,0.0";
      ],
      Some
        [
          "float16|2,3|-0.75,nan,0.625,0.0,0.875,inf";
          "int32|2,3|1.0,0.0,2.0,0.0,2.0,0.0";
        ] );
    ( "g.nan_to_num",
      true,
      [ "int64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      true,
      [ "int64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int64||5.0" ] );
    ( "g.nanargmin",
      true,
      [ "int64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int64||3.0" ] );
    ( "g.sort_complex",
      true,
      [ "int64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex128|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.nan_to_num",
      true,
      [ "uint64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "uint64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ] );
    ( "g.nanargmax",
      true,
      [ "uint64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int64||5.0" ] );
    ( "g.nanargmin",
      true,
      [ "uint64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "int64||3.0" ] );
    ( "g.sort_complex",
      true,
      [ "uint64|2,3|1.0,2.0,3.0,0.0,5.0,7.0" ],
      Some [ "complex128|3|1.0,2.0,3.0|0.0,0.0,0.0" ] );
    ( "g.sort",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2,3|0.0,1.0,1.0,0.0,1.0,1.0" ] );
    ( "g.argsort",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|2,3|1.0,0.0,2.0,0.0,1.0,2.0" ] );
    ( "g.nan_to_num",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ("g.round", true, [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ], None);
    ("g.all", true, [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ], Some [ "bool||0.0" ]);
    ("g.any", true, [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ], Some [ "bool||1.0" ]);
    ( "g.argmax",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64||0.0" ] );
    ( "g.argmin",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64||1.0" ] );
    ( "g.nanargmax",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64||0.0" ] );
    ( "g.nanargmin",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64||1.0" ] );
    ( "g.unwrap",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "float64|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ( "g.nonzero",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|4|0.0,0.0,1.0,1.0"; "int64|4|0.0,2.0,1.0,2.0" ] );
    ( "g.flatnonzero",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|4|0.0,2.0,4.0,5.0" ] );
    ( "g.argwhere",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|4,2|0.0,0.0,0.0,2.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "bool|2|0.0,1.0" ] );
    ( "g.i0",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,3|1.2660659551620483,1.0000001192092896,1.2660659551620483,1.0000001192092896,1.2660659551620483,1.2660659551620483";
        ] );
    ( "g.sort_complex",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "complex64|3|0.0,1.0,1.0|0.0,0.0,0.0" ] );
    ( "g.clip",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ] );
    ( "g.clip_lo",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0" ],
      Some [ "int64|2,3|1.0,1.0,1.0,1.0,1.0,1.0" ] );
    ( "g.isclose",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool|2,3|1.0,1.0,1.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some [ "float32|2,3|2.0,0.0,8.0,0.0,32.0,128.0" ] );
    ( "g.frexp",
      true,
      [ "bool|2,3|1.0,0.0,1.0,0.0,1.0,1.0"; "bool|2,3|1.0,0.0,1.0,1.0,1.0,0.0" ],
      Some
        [
          "float32|2,3|0.5,0.0,0.5,0.0,0.5,0.5";
          "int32|2,3|1.0,0.0,1.0,0.0,1.0,1.0";
        ] );
    ( "g.sort",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex128|2,3|0.0,1.0,2.0,-0.0,1.0,3.0|0.5,1.0,-1.0,-1.0,0.0,0.0" ]
    );
    ( "g.argsort",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int64|2,3|2.0,0.0,1.0,1.0,0.0,2.0" ] );
    ( "g.nan_to_num",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ]
    );
    ( "g.round",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.0,0.0,-1.0,0.0" ]
    );
    ( "g.all",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "bool||0.0" ] );
    ( "g.any",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "bool||1.0" ] );
    ( "g.argmax",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.argmin",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nanargmax",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nanargmin",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.unwrap",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.nonzero",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some
        [ "int64|6|0.0,0.0,0.0,1.0,1.0,1.0"; "int64|6|0.0,1.0,2.0,0.0,1.0,2.0" ]
    );
    ( "g.flatnonzero",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int64|6|0.0,1.0,2.0,3.0,4.0,5.0" ] );
    ( "g.argwhere",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "int64|6,2|0.0,0.0,0.0,1.0,0.0,2.0,1.0,0.0,1.0,1.0,1.0,2.0" ] );
    ( "g.unique",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "complex128|6|-0.0,0.0,1.0,1.0,2.0,3.0|-1.0,0.5,0.0,1.0,-1.0,0.0" ]
    );
    ( "g.i0",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.sort_complex",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      Some [ "complex128|3|0.0,1.0,2.0|0.5,1.0,-1.0" ] );
    ( "g.clip",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.clip_lo",
      true,
      [ "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0" ],
      None );
    ( "g.isclose",
      true,
      [
        "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex128|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      Some [ "bool|2,3|0.0,1.0,0.0,0.0,1.0,0.0" ] );
    ( "g.allclose",
      true,
      [
        "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex128|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      Some [ "bool||0.0" ] );
    ( "g.ldexp",
      true,
      [
        "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex128|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      None );
    ( "g.frexp",
      true,
      [
        "complex128|2,3|1.0,2.0,0.0,1.0,-0.0,3.0|1.0,-1.0,0.5,0.0,-1.0,0.0";
        "complex128|2,3|0.0,2.0,1.0,3.0,-0.0,1.0|0.5,-1.0,1.0,0.0,-1.0,0.0";
      ],
      None );
    ( "g.la.inv",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.30000001192092896,-0.10000000149011612,-0.20000000298023224,0.4000000059604645";
        ] );
    ( "g.la.det",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||10.0" ] );
    ( "g.la.slogdet",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||1.0"; "float32||2.3025851249694824" ] );
    ( "g.la.norm",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||5.4772257804870605" ] );
    ( "g.la.matrix_rank",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "int32||2.0" ] );
    ( "g.la.pinv",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.2999999523162842,-0.09999997913837433,-0.20000000298023224,0.3999999761581421";
        ] );
    ( "g.la.qr",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|-0.8944270610809326,-0.4472135901451111,-0.4472135901451111,0.8944271802902222";
          "float32|2,2|-4.4721360206604,-2.2360680103302,0.0,2.2360680103302";
        ] );
    ( "g.la.svdvals",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2|5.116672992706299,1.954395055770874" ] );
    ( "g.la.eigvalsh",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2|1.9188611507415771,5.081138610839844" ] );
    ( "g.la.eigvals",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "complex64|2|5.0,1.9999998807907104|0.0,0.0" ] );
    ( "g.la.cholesky",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|4.242640495300293,0.0,2.5927250385284424,2.697735548019409";
        ] );
    ( "g.la.trace",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||7.0" ] );
    ( "g.la.matrix_power",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2,2|18.0,7.0,14.0,11.0" ] );
    ( "g.la.cond",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||2.6180341243743896" ] );
    ( "g.la.expm",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|101.40640258789062,47.0086784362793,94.0173568725586,54.397735595703125";
        ] );
    ( "g.la.sqrtm",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "complex64|2,2|1.9621165990829468,0.27395161986351013,0.5479031205177307,1.6881650686264038|0.0,0.0,0.0,0.0";
        ] );
    ( "g.la.det2",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||10.0" ] );
    ( "g.la.inv2",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.30000001192092896,-0.10000000149011612,-0.20000000298023224,0.4000000059604645";
        ] );
    ( "g.la.solve",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0"; "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2,2|1.0,0.0,0.0,1.0" ] );
    ( "g.la.lstsq",
      false,
      [ "float32|2,2|4.0,1.0,2.0,3.0"; "float32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.9999997615814209,2.7281203074380755e-08,-2.1858618026726617e-08,1.0";
        ] );
    ( "g.la.inv",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|0.2586206793785095,-0.08620688319206238,-0.1379310041666031,0.37931028008461|-0.10344827175140381,0.03448275476694107,0.15517239272594452,-0.051724132150411606";
        ] );
    ( "g.la.det",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "complex64||10.0|4.0" ] );
    ( "g.la.slogdet",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64||0.9284766912460327|0.3713906705379486";
          "float32||2.3767950534820557";
        ] );
    ( "g.la.norm",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "float32||5.656854152679443" ] );
    ( "g.la.matrix_rank",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "int32||2.0" ] );
    ( "g.la.pinv",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|0.2586207091808319,-0.08620689064264297,-0.1379310041666031,0.37931033968925476|-0.103448286652565,0.034482765942811966,0.1551724076271057,-0.0517241470515728";
        ] );
    ( "g.la.qr",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|-0.8528029918670654,0.3167230188846588,-0.42640143632888794,-0.8709883689880371|-0.21320071816444397,0.35631343722343445,0.21320071816444397,-0.11877115070819855";
          "complex64|2,2|-4.690415859222412,-2.132007360458374,0.0,-2.2962419986724854|0.0,-0.42640143632888794,0.0,0.0";
        ] );
    ( "g.la.svdvals",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "float32|2|5.2756195068359375,2.0415291786193848" ] );
    ( "g.la.eigvalsh",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "float32|2|1.8416876792907715,5.1583123207092285" ] );
    ( "g.la.eigvals",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2|4.925052642822266,2.074946403503418|0.3245680332183838,0.6754319071769714";
        ] );
    ( "g.la.cholesky",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|4.358899116516113,0.0,2.2941572666168213,2.8003759384155273|0.0,0.0,-1.3764944076538086,0.0";
        ] );
    ( "g.la.trace",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "complex64||7.0|1.0" ] );
    ( "g.la.matrix_power",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "complex64|2,2|17.0,7.0,15.0,11.0|7.0,1.0,-5.0,-1.0" ] );
    ( "g.la.cond",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "float32||2.584151029586792" ] );
    ( "g.la.expm",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|79.64018249511719,41.302040100097656,101.34880828857422,57.08286666870117|54.4693489074707,18.744722366333008,-3.812591791152954,-5.57741117477417";
        ] );
    ( "g.la.sqrtm",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|1.985837459564209,0.26993411779403687,0.5175241827964783,1.693558931350708|0.27607762813568115,-0.022343963384628296,-0.3146221935749054,0.02848753333091736";
        ] );
    ( "g.la.det2",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some [ "complex64||10.0|4.0" ] );
    ( "g.la.inv2",
      false,
      [ "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0" ],
      Some
        [
          "complex64|2,2|0.2586206793785095,-0.08620688319206238,-0.1379310041666031,0.37931028008461|-0.10344827175140381,0.03448275476694107,0.15517239272594452,-0.051724132150411606";
        ] );
    ( "g.la.solve",
      false,
      [
        "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0";
        "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0";
      ],
      Some
        [
          "complex64|2,2|1.0,2.8214595104714135e-08,0.0,0.9999998807907104|0.0,-6.35090646738945e-09,0.0,-2.810970123334755e-09";
        ] );
    ( "g.la.lstsq",
      false,
      [
        "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0";
        "complex64|2,2|4.0,1.0,2.0,3.0|1.0,0.0,-1.0,0.0";
      ],
      Some
        [
          "complex64|2,2|1.0000001192092896,3.896221301147307e-08,-2.4789232000443917e-08,0.9999999403953552|-9.373117393352004e-09,2.5514740542575964e-08,7.260281620347087e-08,-2.9802322387695312e-08";
        ] );
    ( "g.la.inv",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.30000001192092896,-0.10000000149011612,-0.20000000298023224,0.4000000059604645";
        ] );
    ( "g.la.det",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||10.0" ] );
    ( "g.la.slogdet",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||1.0"; "float32||2.3025851249694824" ] );
    ( "g.la.norm",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||5.4772257804870605" ] );
    ( "g.la.matrix_rank",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "int32||2.0" ] );
    ( "g.la.pinv",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.2999999523162842,-0.09999997913837433,-0.20000000298023224,0.3999999761581421";
        ] );
    ( "g.la.qr",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|-0.8944270610809326,-0.4472135901451111,-0.4472135901451111,0.8944271802902222";
          "float32|2,2|-4.4721360206604,-2.2360680103302,0.0,2.2360680103302";
        ] );
    ( "g.la.svdvals",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2|5.116672992706299,1.954395055770874" ] );
    ( "g.la.eigvalsh",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2|1.9188611507415771,5.081138610839844" ] );
    ( "g.la.eigvals",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "complex64|2|5.0,1.9999998807907104|0.0,0.0" ] );
    ( "g.la.cholesky",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|4.242640495300293,0.0,2.5927250385284424,2.697735548019409";
        ] );
    ("g.la.trace", false, [ "int32|2,2|4.0,1.0,2.0,3.0" ], Some [ "int32||7.0" ]);
    ( "g.la.matrix_power",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "int32|2,2|18.0,7.0,14.0,11.0" ] );
    ( "g.la.cond",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||2.6180341243743896" ] );
    ( "g.la.expm",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|101.40640258789062,47.0086784362793,94.0173568725586,54.397735595703125";
        ] );
    ( "g.la.sqrtm",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "complex64|2,2|1.9621165990829468,0.27395161986351013,0.5479031205177307,1.6881650686264038|0.0,0.0,0.0,0.0";
        ] );
    ( "g.la.det2",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32||10.0" ] );
    ( "g.la.inv2",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.30000001192092896,-0.10000000149011612,-0.20000000298023224,0.4000000059604645";
        ] );
    ( "g.la.solve",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0"; "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float32|2,2|1.0,0.0,0.0,1.0" ] );
    ( "g.la.lstsq",
      false,
      [ "int32|2,2|4.0,1.0,2.0,3.0"; "int32|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "float32|2,2|0.9999997615814209,2.7281203074380755e-08,-2.1858618026726617e-08,1.0";
        ] );
    ("g.la.inv", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ( "g.la.det",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16||10.0" ] );
    ( "g.la.slogdet",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16||1.0"; "float16||2.302734375" ] );
    ( "g.la.norm",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16||5.4765625" ] );
    ("g.la.matrix_rank", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.pinv", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.qr", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.svdvals", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.eigvalsh", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.eigvals", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.cholesky", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ( "g.la.trace",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16||7.0" ] );
    ( "g.la.matrix_power",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16|2,2|18.0,7.0,14.0,11.0" ] );
    ("g.la.cond", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ("g.la.expm", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ( "g.la.sqrtm",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some
        [
          "complex64|2,2|1.9621165990829468,0.27395161986351013,0.5479031205177307,1.6881650686264038|0.0,0.0,0.0,0.0";
        ] );
    ( "g.la.det2",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0" ],
      Some [ "float16||10.0" ] );
    ("g.la.inv2", false, [ "float16|2,2|4.0,1.0,2.0,3.0" ], None);
    ( "g.la.solve",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0"; "float16|2,2|4.0,1.0,2.0,3.0" ],
      None );
    ( "g.la.lstsq",
      false,
      [ "float16|2,2|4.0,1.0,2.0,3.0"; "float16|2,2|4.0,1.0,2.0,3.0" ],
      None );
    ( "g.la.inv",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2,2|1.0,0.0,-1.0,1.0" ] );
    ("g.la.det", false, [ "bool|2,2|1.0,0.0,1.0,1.0" ], Some [ "float32||1.0" ]);
    ( "g.la.slogdet",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32||1.0"; "float32||0.0" ] );
    ( "g.la.norm",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32||1.7320507764816284" ] );
    ( "g.la.matrix_rank",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "int32||2.0" ] );
    ( "g.la.pinv",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,2|1.0000001192092896,4.8470916169662814e-09,-1.0000001192092896,1.0000001192092896";
        ] );
    ( "g.la.qr",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,2|-0.7071068286895752,-0.7071067690849304,-0.7071067690849304,0.7071068286895752";
          "float32|2,2|-1.4142135381698608,-0.7071067690849304,0.0,0.7071068286895752";
        ] );
    ( "g.la.svdvals",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2|1.6180340051651,0.6180340051651001" ] );
    ( "g.la.eigvalsh",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2|0.5,1.5" ] );
    ( "g.la.eigvals",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "complex64|2|1.0,1.0|0.0,0.0" ] );
    ( "g.la.cholesky",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2,2|1.0,0.0,0.5,0.8660253882408142" ] );
    ("g.la.trace", false, [ "bool|2,2|1.0,0.0,1.0,1.0" ], Some [ "int32||2.0" ]);
    ( "g.la.matrix_power",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "bool|2,2|1.0,0.0,1.0,1.0" ] );
    ( "g.la.cond",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32||2.6180338859558105" ] );
    ( "g.la.expm",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,2|2.7182817459106445,0.0,2.7182817459106445,2.7182817459106445";
        ] );
    ( "g.la.sqrtm",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "complex64|2,2|1.0,0.0,0.5,1.0|0.0,0.0,0.0,0.0" ] );
    ("g.la.det2", false, [ "bool|2,2|1.0,0.0,1.0,1.0" ], Some [ "float32||1.0" ]);
    ( "g.la.inv2",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2,2|1.0,0.0,-1.0,1.0" ] );
    ( "g.la.solve",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0"; "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some [ "float32|2,2|1.0,0.0,0.0,1.0" ] );
    ( "g.la.lstsq",
      false,
      [ "bool|2,2|1.0,0.0,1.0,1.0"; "bool|2,2|1.0,0.0,1.0,1.0" ],
      Some
        [
          "float32|2,2|1.0000001192092896,4.8470916169662814e-09,4.8470916169662814e-09,1.0000001192092896";
        ] );
  ]

module Nd = Ndarray
module Dt = Dtype

let table : (string, Types.value list -> Types.value list) Hashtbl.t =
  Hashtbl.create 500

let reg name f = Hashtbl.replace table name f

let dtype_of_name = function
  | "float16" -> Dt.F16
  | "bfloat16" -> Dt.Bf16
  | "float32" -> Dt.F32
  | "float64" -> Dt.F64
  | "int8" -> Dt.I8
  | "int16" -> Dt.I16
  | "int32" -> Dt.I32
  | "int64" -> Dt.I64
  | "uint8" -> Dt.Uint8
  | "uint16" -> Dt.Uint16
  | "uint32" -> Dt.Uint32
  | "uint64" -> Dt.Uint64
  | "bool" -> Dt.Bool
  | "complex64" -> Dt.Complex64
  | "complex128" -> Dt.Complex128
  | s -> failwith ("unknown dtype " ^ s)

let split_on c s = if s = "" then [] else String.split_on_char c s

let parse text =
  match String.split_on_char '|' text with
  | dt :: sh :: rest ->
      let dt = dtype_of_name dt in
      let sh = Array.of_list (List.map int_of_string (split_on ',' sh)) in
      let floats s =
        Array.of_list (List.map float_of_string (split_on ',' s))
      in
      let re = floats (List.hd rest) in
      let im = match rest with [ _; i ] -> floats i | _ -> [||] in
      (dt, sh, re, im)
  | _ -> failwith "bad literal"

let value_of text =
  let dt, sh, re, im = parse text in
  match dt with
  | Dt.Complex64 | Dt.Complex128 ->
      Types.Concrete
        (Nd.of_complex dt sh
           (Array.mapi (fun i r -> { Complex.re = r; im = im.(i) }) re))
  | Dt.I64 | Dt.Uint64 ->
      Types.Concrete (Nd.of_int64s dt sh (Array.map Int64.of_float re))
  | _ -> Types.Concrete (Nd.of_floats dt sh re)

let host = function
  | Types.Concrete n -> n
  | Types.Device b -> Pjrt.Buffer.to_host b
  | Types.Tracer _ -> failwith "tracer result"

let tolerance = function
  | Dt.F16 -> 3e-3
  | Dt.Bf16 -> 2e-2
  | Dt.F32 | Dt.Complex64 -> 3e-4
  | Dt.F64 | Dt.Complex128 -> 1e-9
  | _ -> 0.0

let close tol a b =
  (Float.is_nan a && Float.is_nan b)
  || a = b
  || Float.is_finite a && Float.is_finite b
     && Float.abs (a -. b)
        <= tol *. Float.max 1.0 (Float.max (Float.abs a) (Float.abs b))

let index_of_flat sh f =
  let idx = Array.make (Array.length sh) 0 in
  let r = ref f in
  for d = Array.length sh - 1 downto 0 do
    idx.(d) <- !r mod sh.(d);
    r := !r / sh.(d)
  done;
  idx

let mismatch expected got =
  let dt, sh, re, im = parse expected in
  let nd = host got in
  if Nd.dtype nd <> dt then
    Some
      (Printf.sprintf "dtype %s, want %s"
         (Dt.short_name (Nd.dtype nd))
         (Dt.short_name dt))
  else if Nd.shape nd <> sh then Some "shape differs"
  else begin
    let tol = tolerance dt in
    let bad = ref None in
    Array.iteri
      (fun f r ->
        let idx = index_of_flat sh f in
        let ok =
          match dt with
          | Dt.Complex64 | Dt.Complex128 ->
              let z = Nd.get_c nd idx in
              close tol r z.Complex.re && close tol im.(f) z.Complex.im
          | _ -> close tol r (Nd.get_f nd idx)
        in
        if (not ok) && !bad = None then
          bad := Some (Printf.sprintf "cell %d differs" f))
      re;
    !bad
  end

let run_group cases () =
  let failures = ref [] in
  let note name msg = failures := (name ^ ": " ^ msg) :: !failures in
  List.iter
    (fun (name, x64, args, expected) ->
      let f =
        match Hashtbl.find_opt table name with
        | Some f -> f
        | None -> failwith ("no function for " ^ name)
      in
      let outcome =
        try
          Ok
            (Config.with_value Config.enable_x64 x64 (fun () ->
                 let values = f (List.map value_of args) in
                 List.map host values))
        with
        | Failure m | Invalid_argument m -> Error m
        | Py_exn.Type_error m
        | Py_exn.Value_error m
        | Py_exn.Index_error m
        | Py_exn.Overflow_error m
        | Py_exn.Not_implemented_error m ->
            Error m
      in
      match (outcome, expected) with
      | Error _, None -> ()
      | Error m, Some _ -> note name ("raised " ^ m)
      | Ok _, None -> note name "should raise"
      | Ok got, Some want ->
          if List.length got <> List.length want then note name "arity"
          else
            List.iter2
              (fun w g ->
                match mismatch w (Types.Concrete g) with
                | None -> ()
                | Some m -> note name m)
              want got)
    cases;
  match !failures with
  | [] -> ()
  | l -> Alcotest.fail (String.concat "\n" (List.rev l))

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ();
  T3.register reg;
  T4.register reg;
  T5.register reg;
  T6.register reg;
  Alcotest.run "hardening_options"
    [
      ( "real jax",
        [
          Alcotest.test_case "numpy and nn options" `Quick (run_group options);
          Alcotest.test_case "tridiagonal, cholesky, eigh, lstsq" `Quick
            (run_group linalg);
          Alcotest.test_case "lax and scipy options" `Quick
            (run_group lax_scipy);
          Alcotest.test_case "numpy across dtypes" `Quick
            (run_group numpy_dtypes);
        ] );
    ]
