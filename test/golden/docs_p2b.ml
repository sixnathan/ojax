module U = Yojson.Safe.Util
module T = Ojax.Types
module F = Ojax.Numpy.Ufuncs
module NL = Ojax.Numpy.Lax_numpy
module SL = Ojax.Scipy.Linalg
module RC = Ojax.Random.Core

let str row k = U.to_string (U.member k row)
let dt_of name = Option.get (Ojax.Dtype.of_numpy_name name)

let f32 shape vals =
  T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape vals)

let leaf = Docs_p2.leaf

let jit2 f a b =
  match
    Ojax.Api.jit
      (function
        | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] -> leaf (f a b)
        | _ -> failwith "docs_p2b: arity")
      [ leaf a; leaf b ]
  with
  | Ojax.Tree_util.Leaf r -> r
  | _ -> failwith "docs_p2b: jit result"

let jit1 f a = jit2 (fun a _ -> f a) a a

let rank_thunk op mode () =
  let x = f32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let m = f32 [| 2; 3 |] (Array.init 6 (fun k -> float_of_int k /. 2.0)) in
  let gt v = F.greater v (Docs_p2.py_int 1.0) in
  let f a b =
    match op with
    | "add" -> F.add a b
    | "multiply" -> jit2 F.multiply a b
    | "subtract" -> F.subtract b a
    | "power" -> F.power a b
    | "arctan2" -> F.arctan2 a b
    | "maximum" -> F.maximum a b
    | "logical_and" -> F.logical_and (gt a) (gt b)
    | "equal" -> F.equal a b
    | "where" -> NL.where_ (gt a) a b
    | "add_scalar" -> F.add a (Docs_p2.py_float 2.0)
    | other -> failwith ("docs_p2b: unknown rank op " ^ other)
  in
  Ojax.Api.clear_caches ();
  Ojax.Config.with_value Ojax.Config.numpy_rank_promotion mode (fun () ->
      Docs_p2.text_of (f x m))

let cast_thunk fn src dst () =
  let x =
    if src = "float32" then f32 [| 3 |] [| 1.5; -2.25; 3.0 |]
    else
      Ojax.Lax_api.complex
        (f32 [| 3 |] [| 1.5; -2.25; 0.0 |])
        (f32 [| 3 |] [| 2.0; -1.0; 3.0 |])
  in
  let dt = dt_of dst in
  Ojax.Api.clear_caches ();
  Docs_p2.text_of
    (match fn with
    | "astype" -> NL.astype x dt
    | "method" -> Ojax.Numpy.Array_methods.astype x dt
    | "lax" -> Ojax.Lax_api.convert_element_type x dt
    | "lax_jit" -> jit1 (fun a -> Ojax.Lax_api.convert_element_type a dt) x
    | "astype_jit" -> jit1 (fun a -> NL.astype a dt) x
    | other -> failwith ("docs_p2b: unknown cast " ^ other))

let one_hot_thunk dt () =
  let x =
    NL.astype
      (T.Concrete
         (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 4 |] [| 0.0; 2.0; 1.0; 1.0 |]))
      (dt_of dt)
  in
  Ojax.Api.clear_caches ();
  Docs_p2.text_of (Ojax.Nn.Functions.one_hot ~num_classes:3 x)

let solve_thunk fn kind () =
  let a =
    [| 4.0; 1.0; 0.5; 1.0; 3.0; 0.25; 0.5; 0.25; 2.0 |]
    |> Array.map (fun v -> Int32.float_of_bits (Int32.bits_of_float v))
  in
  let a1 = f32 [| 3; 3 |] a in
  let a2 =
    f32 [| 2; 3; 3 |]
      (Array.append a
         (Array.mapi (fun k v -> if k mod 4 = 0 then v +. 1.0 else v) a))
  in
  let b2 = f32 [| 2; 3 |] (Array.init 6 (fun k -> float_of_int (k + 1))) in
  let b1 = f32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let bm = f32 [| 3; 2 |] (Array.init 6 (fun k -> float_of_int (k + 1))) in
  let call a b =
    match fn with
    | "cho_solve" -> SL.cho_solve (SL.cho_factor ~lower:true a) b
    | "lu_solve" -> SL.lu_solve (SL.lu_factor a) b
    | "solve_triangular" -> SL.solve_triangular ~lower:false a b
    | other -> failwith ("docs_p2b: unknown solve " ^ other)
  in
  Ojax.Api.clear_caches ();
  Docs_p2.text_of
    (match kind with
    | "batched_1d" -> call a2 b2
    | "jit_batched_1d" -> jit2 call a2 b2
    | "vector" -> call a1 b1
    | "matrix" -> call a1 bm
    | other -> failwith ("docs_p2b: unknown solve shape " ^ other))

let promo_thunk fn d1 d2 () =
  let mk vals d =
    NL.astype
      (T.Concrete
         (Ojax.Ndarray.of_floats
            (Ojax.Dtypes.canonicalize_dtype Ojax.Dtype.I64)
            [| 3 |] vals))
      (dt_of d)
  in
  let x = mk [| 1.0; 2.0; 3.0 |] d1 and y = mk [| 3.0; 0.0; 1.0 |] d2 in
  Ojax.Api.clear_caches ();
  Ojax.Config.with_value Ojax.Config.numpy_dtype_promotion "strict" (fun () ->
      Docs_p2.text_of
        (match fn with
        | "add" -> F.add x y
        | "multiply" -> jit2 F.multiply x y
        | "where" -> NL.where_ (F.greater x y) x y
        | "concatenate" -> NL.concatenate [ x; y ]
        | other -> failwith ("docs_p2b: unknown promo " ^ other)))

let x64_thunk mode fn dt () =
  let x = f32 [| 3 |] [| 1.5; 2.0; -3.0 |] in
  let dtype = dt_of dt in
  let module A = Ojax.Numpy.Array_creation in
  Ojax.Api.clear_caches ();
  Ojax.Config.with_value Ojax.Config.explicit_x64_dtypes mode (fun () ->
      Docs_p2.text_of
        (match fn with
        | "zeros" -> A.zeros ~dtype [| 2 |]
        | "ones" -> A.ones ~dtype [| 2 |]
        | "full" -> A.full ~dtype [| 2 |] 7.0
        | "arange" -> NL.arange ~dtype 4.0
        | "astype" -> NL.astype x dtype
        | "asarray" -> Ojax.Numpy.Array_constructors.asarray ~dtype x
        | other -> failwith ("docs_p2b: unknown x64mode " ^ other)))

let legacy_thunk mode fn () =
  let raw k = RC.prng_key (Docs_p2.py_int k) in
  Ojax.Api.clear_caches ();
  Ojax.Config.with_value Ojax.Config.legacy_prng_key mode (fun () ->
      Docs_p2.text_of
        (match fn with
        | "normal" -> RC.normal (raw 3.0) ~shape:[| 2 |]
        | "split" -> RC.split (raw 5.0) 2
        | "typed_normal" ->
            RC.normal (RC.key (Docs_p2.py_int 3.0)) ~shape:[| 2 |]
        | other -> failwith ("docs_p2b: unknown legacy " ^ other)))

let actual_of row =
  let thunk =
    match str row "group" with
    | "rank" -> rank_thunk (str row "case") (str row "mode")
    | "cast" -> cast_thunk (str row "case") (str row "src") (str row "dst")
    | "one_hot" -> one_hot_thunk (str row "dtype")
    | "solve" -> solve_thunk (str row "case") (str row "shape")
    | "promo" -> promo_thunk (str row "case") (str row "d1") (str row "d2")
    | "x64mode" -> x64_thunk (str row "mode") (str row "case") (str row "dtype")
    | "legacy" -> legacy_thunk (str row "mode") (str row "case")
    | g -> failwith ("docs_p2b: unknown group " ^ g)
  in
  Docs_p2.run_case thunk

let show r =
  Docs_p2.show r ^ " warnings="
  ^ String.concat " | "
      (List.map (fun (c, m) -> c ^ ": " ^ m) r.Docs_p2.warnings)

let check_row ~x64 row () =
  let id = str row "id" in
  let expected = Docs_p2.expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = "docs_p2b:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = Docs_p2.same expected actual in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "DOCS_P2B DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat
      (Filename.concat Docs_p2.goldens_root "docs_p2b")
      (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "docs_p2b_" ^ tag,
    List.map
      (fun row -> Alcotest.test_case (str row "id") `Quick (check_row ~x64 row))
      rows )
