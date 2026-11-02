open Ojax
module LA = Numpy.Linalg
module SL = Scipy.Linalg
module SS = Scipy.Linalg_special
module F = Numpy.Fft
module X = Linalg_extras

let z ?(dtype = Dtype.F32) shape =
  Types.Concrete
    (Ndarray.of_floats dtype shape
       (Array.make (Array.fold_left ( * ) 1 shape) 0.5))

let s0 = [||]
let v1 = [| 1 |]
let v3 = [| 3 |]
let m00 = [| 0; 0 |]
let m03 = [| 0; 3 |]
let m23 = [| 2; 3 |]
let m32 = [| 3; 2 |]
let m33 = [| 3; 3 |]
let b233 = [| 2; 3; 3 |]

let cases =
  [
    ( "cholesky",
      "TypeError",
      "transpose permutation isn't a permutation of operand dimensions, got \
       permutation (0, -1) for operand shape (3,).",
      fun () -> ignore (LA.cholesky (z v3)) );
    ( "inv",
      "ValueError",
      "Argument to inv must have shape [..., n, n], got ().",
      fun () -> ignore (LA.inv (z s0)) );
    ( "det",
      "ValueError",
      "Argument to _det() must have shape [..., n, n], got (2, 3)",
      fun () -> ignore (LA.det (z m23)) );
    ( "slogdet",
      "ValueError",
      "Argument to slogdet() must have shape [..., n, n], got (3,)",
      fun () -> ignore (LA.slogdet (z v3)) );
    ( "pinv",
      "ValueError",
      "not enough values to unpack (expected 2, got 0)",
      fun () -> ignore (LA.pinv (z s0)) );
    ( "matrix_power",
      "TypeError",
      "0-dimensional array given. Array must be at least two-dimensional",
      fun () -> ignore (LA.matrix_power (z s0) 3) );
    ( "cond",
      "ValueError",
      "jnp.linalg.cond: input array must not be empty; got arr.shape=(0, 3)",
      fun () -> ignore (LA.cond (z m03)) );
    ( "solve",
      "ValueError",
      "left hand array must be at least two dimensional; got a.shape=()",
      fun () -> ignore (LA.solve (z s0) (z s0)) );
    ( "lstsq",
      "TypeError",
      "3-dimensional array given. Array must be two-dimensional",
      fun () -> ignore (LA.lstsq (z b233) (z b233)) );
    ( "cross",
      "ValueError",
      "Both input arrays must be (arrays of) 3-dimensional vectors, but they \
       have x1.shape[axis]=3 and x2.shape[axis]=2",
      fun () -> ignore (LA.cross (z v3) (z m32)) );
    ( "outer",
      "ValueError",
      "Input arrays must be one-dimensional, but they are x1.ndim=2 x2.ndim=1",
      fun () -> ignore (LA.outer (z m23) (z v3)) );
    ( "scipy lu",
      "ValueError",
      "too many values to unpack (expected 2)",
      fun () -> ignore (SL.lu (z b233)) );
    ( "scipy hessenberg",
      "IndexError",
      "tuple index out of range",
      fun () -> ignore (SL.hessenberg (z s0)) );
    ( "scipy solve_triangular",
      "ValueError",
      "inconsistent size for core dimension 'n': 1 vs 0 on vectorized function \
       with excluded=frozenset() and signature='(n,n),(n)->(n)'",
      fun () -> ignore (SL.solve_triangular (z m00) (z v1)) );
    ( "scipy qr_multiply",
      "ValueError",
      "not enough values to unpack (expected 2, got 0)",
      fun () -> ignore (SL.qr_multiply (z s0) (z s0)) );
    ( "scipy expm_frechet",
      "ValueError",
      "expected E to be a (batched) square matrix, got E.shape=(0, 3)",
      fun () -> ignore (SL.expm_frechet (z m00) (z m03)) );
    ( "companion",
      "ValueError",
      "The length of `a` along the last axis must be at least 2; got shape \
       (1,).",
      fun () -> ignore (SS.companion (z v1)) );
    ( "leslie",
      "ValueError",
      "Incorrect lengths for f and s. The length of s along the last axis must \
       be one less than the length of f; got f shape (3,) and s shape (3,).",
      fun () -> ignore (SS.leslie (z v3) (z v3)) );
    ( "lax qr dtype",
      "TypeError",
      "qr does not accept dtype int32 at position 0. Accepted dtypes at \
       position 0 are subtypes of complexfloating, floating.",
      fun () -> ignore (Lax.Linalg.qr (z ~dtype:Dtype.I32 m33)) );
    ( "lax qdwh",
      "ValueError",
      "not enough values to unpack (expected 2, got 0)",
      fun () -> ignore (X.qdwh (z s0)) );
    ( "lax cholesky_update",
      "TypeError",
      "Input 1 to cholesky_update must have a rank of exactly 1, but got \
       shape=(3, 3)",
      fun () -> ignore (X.cholesky_update (z m33) (z m33)) );
    ( "fft",
      "ValueError",
      "axis -1 is out of bounds for array of dimension 0",
      fun () -> ignore (F.fft (z s0)) );
    ( "hfft",
      "IndexError",
      "tuple index out of range",
      fun () -> ignore (F.hfft (z s0)) );
    ( "fft n",
      "ValueError",
      "Shape should be positive.",
      fun () -> ignore (F.fft ~n:0 (z v3)) );
    ( "fftn s",
      "IndexError",
      "list assignment index out of range",
      fun () -> ignore (F.fftn ~s:[ 2; 2 ] (z s0)) );
    ( "fftn repeated",
      "ValueError",
      "jax.numpy.fft.fftn does not support repeated axes. Got axes [0, 0].",
      fun () -> ignore (F.fftn ~axes:[ 0; 0 ] (z m33)) );
    ( "fft2 axes",
      "ValueError",
      "jax.numpy.fft.fft2 only supports 2 axes. Got axes = [0].",
      fun () -> ignore (F.fft2 ~axes:[ 0 ] (z m33)) );
    ( "fftshift",
      "IndexError",
      "tuple index out of range",
      fun () -> ignore (F.fftshift ~axes:[ 1 ] (z s0)) );
    ( "vmap rank",
      "ValueError",
      "vmap was requested to map its argument along axis 0, which implies that \
       its rank should be at least 1, but is only 0 (its shape is ())",
      fun () ->
        ignore (Interpreters.Batching.vmap (fun a -> a) [ Some 0 ] [ z s0 ]) );
  ]

let check (name, cls, message, f) =
  Alcotest.test_case name `Quick (fun () ->
      match f () with
      | () -> Alcotest.failf "%s did not raise" name
      | exception e ->
          Alcotest.(check (option string))
            "class" (Some cls) (Py_exn.class_name e);
          Alcotest.(check (option string))
            "message" (Some message) (Py_exn.message_of e))

let () =
  Alcotest.run "linalg_validation" [ ("reference jax", List.map check cases) ]
