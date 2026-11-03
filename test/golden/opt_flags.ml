module U = Yojson.Safe.Util
module T = Ojax.Types
module F = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module N = Ojax.Nn.Functions

let str row k = U.to_string (U.member k row)

let vector () =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 4097 |]
       (Array.init 4097 (fun i -> (float_of_int i /. 256.0) -. 8.0)))

let matrix () =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 64; 64 |]
       (Array.init 4096 (fun i -> (float_of_int (i mod 17) /. 8.0) -. 1.0)))

let op_fn = function
  | "exp" -> F.exp
  | "tanh" -> F.tanh
  | "log_abs" -> fun a -> F.log (F.absolute a)
  | "expm1" -> F.expm1
  | "log1p_abs" -> fun a -> F.log1p (F.absolute a)
  | "sin" -> F.sin
  | "cos" -> F.cos
  | "logistic" -> N.sigmoid
  | "gelu" -> fun a -> N.gelu a
  | "softmax" -> fun a -> N.softmax a
  | "logsumexp" -> fun a -> N.logsumexp a
  | "erf" -> Ojax.Scipy.Special.erf
  | "sum" -> fun a -> R.sum a
  | "mean" -> fun a -> R.mean a
  | "cumsum" -> fun a -> R.cumsum a
  | "pow" -> fun a -> F.power (F.absolute a) (Docs_p2.py_float 1.7)
  | "sqrt_abs" -> fun a -> F.sqrt (F.absolute a)
  | "dot" ->
      fun a ->
        Ojax.Numpy.Tensor_contractions.dot
          (Ojax.Numpy.Tensor_contractions.dot a a)
          a
  | other -> failwith ("opt_flags: unknown op " ^ other)

let with_flag flag thunk =
  let module C = Ojax.Config in
  match flag with
  | "base" -> thunk ()
  | "disable_most_optimizations" ->
      C.with_value C.disable_most_optimizations true thunk
  | "no_remat_pass" -> C.with_value C.compiler_enable_remat_pass false thunk
  | other -> failwith ("opt_flags: unknown flag " ^ other)

let actual_of row =
  let op = str row "op" in
  let arg = if op = "dot" then matrix () else vector () in
  let f = op_fn op in
  let thunk () =
    with_flag (str row "flag") (fun () ->
        Ojax.Api.clear_caches ();
        let out =
          match str row "mode" with "jit" -> Docs_p2b.jit1 f arg | _ -> f arg
        in
        Docs_p2.text_of out)
  in
  Docs_p2.run_case thunk

let check_row ~x64 row () =
  let id = str row "id" in
  let expected = Docs_p2.expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = "opt_flags:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = Docs_p2.same expected actual in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "OPT_FLAGS DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (Docs_p2b.show expected) (Docs_p2b.show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat
      (Filename.concat Docs_p2.goldens_root "opt_flags")
      (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "opt_flags_" ^ tag,
    List.map
      (fun row -> Alcotest.test_case (str row "id") `Quick (check_row ~x64 row))
      rows )
