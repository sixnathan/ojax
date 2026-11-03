module U = Yojson.Safe.Util
module T = Ojax.Types
module TR = Ojax.Tree_util
module F = Ojax.Numpy.Ufuncs
module NL = Ojax.Numpy.Lax_numpy
module A = Ojax.Api
module C = Ojax.Checkpoint
module CB = Ojax.Custom_batching
module JU = Ojax.Numpy.Jnp_util
module G = Ojax.Numpy.Getitem

let name = "docs_p1"

let float_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32

let int_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32

let base_row = ref `Null

let base key shape default =
  match U.member key !base_row with
  | `Null -> default ()
  | v ->
      let vals = Array.of_list (List.map U.to_number (U.to_list v)) in
      T.Concrete (Ojax.Ndarray.of_floats (float_dt ()) shape vals)

let pyf = JU.py_float
let arange n = NL.arange ~dtype:(float_dt ()) n

let x () =
  base "x" [| 3 |] (fun () ->
      F.add (F.true_divide (arange 3.0) (pyf 2.0)) (pyf 0.25))

let y () = base "y" [| 3 |] (fun () -> F.subtract (arange 3.0) (pyf 1.0))

let m () =
  NL.reshape
    (base "m" [| 6 |] (fun () -> F.true_divide (arange 6.0) (pyf 4.0)))
    [| 2; 3 |]

let arg l i = List.nth l i
let one f = function [ a ] -> [ f a ] | _ -> failwith "docs_p1: arity"
let two f = function [ a; b ] -> [ f a b ] | _ -> failwith "docs_p1: arity"

let policy_of = function
  | "nothing_saveable" -> C.nothing_saveable
  | "everything_saveable" -> C.everything_saveable
  | "dots_saveable" -> C.dots_saveable
  | "dots_with_no_batch_dims_saveable" -> C.dots_with_no_batch_dims_saveable
  | p -> failwith ("docs_p1: policy " ^ p)

let first v = G.take v [ G.Int 0 ]

let remat_sin policy args =
  C.checkpoint ~policy
    (fun a ->
      [
        F.multiply
          (F.sin (Ojax.Numpy.Tensor_contractions.dot (arg a 1) (arg a 0)))
          (first (arg a 0));
      ])
    args

let residual_case case =
  let x = x () and y = y () and m = m () in
  let c =
    T.Concrete
      (Ojax.Ndarray.of_floats (float_dt ()) [| 3 |] [| 1.5; -0.5; 2.0 |])
  in
  let inner = one (fun a -> F.multiply (F.sin a) a) in
  match case with
  | "sin_mul" -> (two (fun a b -> F.multiply (F.sin a) b), [ x; y ])
  | "sin_sin" -> (one (fun a -> F.sin (F.sin a)), [ x ])
  | "mul_args" -> (two F.multiply, [ x; y ])
  | "exp_log" -> (one (fun a -> F.log (F.add (F.exp a) (pyf 2.0))), [ x ])
  | "tanh_sum" ->
      ( two (fun a b -> Ojax.Numpy.Reductions.sum (F.multiply (F.tanh a) b)),
        [ x; y ] )
  | "dot" -> (two (fun a b -> Ojax.Numpy.Tensor_contractions.dot b a), [ x; m ])
  | "literal" -> (one (fun a -> F.multiply (F.sin a) (pyf 2.0)), [ x ])
  | "constant" -> (one (fun a -> F.multiply (F.sin a) c), [ x ])
  | "named" ->
      (one (fun a -> F.multiply (C.checkpoint_name (F.sin a) "h") a), [ x ])
  | "named_two" ->
      ( two (fun a b ->
            F.multiply
              (C.checkpoint_name (F.cos a) "a")
              (C.checkpoint_name (F.sin b) "b")),
        [ x; y ] )
  | "jit_inner" ->
      ( (function
        | [ a; b ] ->
            [ F.multiply (List.hd (A.jit_flat ~name:"inner" inner [ a ])) b ]
        | _ -> failwith "docs_p1: arity"),
        [ x; y ] )
  | "remat_sin_sin" -> (C.checkpoint (one (fun a -> F.sin (F.sin a))), [ x ])
  | "remat_unused" ->
      ( two (fun _ b -> F.multiply (Ojax.Numpy.Reductions.sum b) (pyf 1.0)),
        [ x; y ] )
  | "integer_arg" ->
      ( two (fun a b -> F.multiply (F.sin a) b),
        [ x; NL.arange ~dtype:(int_dt ()) 3.0 ] )
  | "square" -> (one (fun a -> F.power a (JU.py_int 2)), [ x ])
  | "div" -> (two F.true_divide, [ x; F.add y (pyf 3.0) ])
  | p when String.length p > 7 && String.sub p 0 7 = "policy_" ->
      (remat_sin (policy_of (String.sub p 7 (String.length p - 7))), [ x; m ])
  | other -> failwith ("docs_p1: residual case " ^ other)

let strip_line l =
  let marker = " from /" in
  let n = String.length marker in
  let rec find i =
    if i + n > String.length l then l
    else if String.sub l i n = marker then String.sub l 0 (i + 6)
    else find (i + 1)
  in
  find 0

let strip_frames text =
  String.concat "\n" (List.map strip_line (String.split_on_char '\n' text))

let frames_proof =
  "(c) Python only: the text ends each line with source_info_util.summarize of \
   the equation, a Python file path, line, column and function name of the \
   traced Python code; with that file registered as excluded (text_noframe) \
   JAX prints the same lines with nothing after from, which ojax matches"

let plain = two (fun a b -> F.multiply (F.sin a) b)

let rule_f _ in_batched = function
  | [ a; b ] -> ([ F.add (F.cos a) b ], [ List.exists Fun.id in_batched ])
  | _ -> failwith "docs_p1: rule arity"

let one_fn = one (fun a -> F.multiply a (pyf 3.0))

let rule_one _ _ = function
  | [ a ] ->
      ( [
          F.add
            (Ojax.Numpy.Reductions.sum ~axis:[| 0 |] ~keepdims:true a)
            (F.multiply a (pyf 0.0));
        ],
        [ true ] )
  | _ -> failwith "docs_p1: rule arity"

let rule_unbatched _ _ = function
  | [ a ] -> ([ F.multiply (first a) (pyf 2.0) ], [ false ])
  | _ -> failwith "docs_p1: rule arity"

let rule_bad_tree _ _ = function
  | [ a ] -> ([ a; a ], [ true ])
  | _ -> failwith "docs_p1: rule arity"

let rule_bad_spec _ _ = function
  | [ a ] -> ([ a ], [ true; true ])
  | _ -> failwith "docs_p1: rule arity"

let cv rule_name f rule = CB.custom_vmap ~rule_name f ~rule
let f_cv = cv "vmap_fns.<locals>.rule" plain rule_f
let g_cv = cv "vmap_fns.<locals>.rule_one" one_fn rule_one
let h_cv = cv "vmap_fns.<locals>.rule_unbatched" one_fn rule_unbatched
let bad_cv = cv "vmap_fns.<locals>.rule_bad_tree" one_fn rule_bad_tree
let bad_spec_cv = cv "vmap_fns.<locals>.rule_bad_spec" one_fn rule_bad_spec

let seq =
  CB.sequential_vmap
    (two (fun a b -> F.multiply (Ojax.Numpy.Reductions.sum (F.sin a)) b))

let leaves l = List.map (fun v -> TR.Leaf v) l

let vmapped ?out_axes f axes args =
  TR.tree_leaves
    (A.vmap ?out_axes
       (fun xs -> TR.Tuple (leaves (f (TR.tree_leaves (TR.Tuple xs)))))
       axes (leaves args))

let jit name f args = A.jit_flat ~name f args

let vmap_case case =
  let x = x () and y = y () and m = m () in
  let m31 = NL.reshape m [| 2; 3; 1 |] in
  let s0 = Some 0 in
  match case with
  | "f_call" -> (f_cv, [ x; y ])
  | "f_vmap" -> (vmapped f_cv [ s0; s0 ], [ m; m ])
  | "f_vmap_in0" -> (vmapped f_cv [ s0; None ], [ m; y ])
  | "f_vmap_in1" -> (vmapped f_cv [ None; s0 ], [ y; m ])
  | "f_vmap_axis1" -> (vmapped f_cv [ Some 1; Some 1 ], [ m; m ])
  | "f_vmap_out1" -> (vmapped ~out_axes:1 f_cv [ s0; s0 ], [ m; m ])
  | "f_jit_vmap" -> (jit "plain" (vmapped f_cv [ s0; s0 ]), [ m; m ])
  | "f_vmap_jit" -> (vmapped (jit "plain" f_cv) [ s0; s0 ], [ m; m ])
  | "f_vmap_vmap" -> (vmapped (vmapped f_cv [ s0; s0 ]) [ s0; s0 ], [ m31; m31 ])
  | "g_vmap" -> (vmapped g_cv [ s0 ], [ m ])
  | "g_call" -> (g_cv, [ x ])
  | "h_vmap" -> (vmapped h_cv [ s0 ], [ m ])
  | "bad_vmap" -> (vmapped bad_cv [ s0 ], [ m ])
  | "bad_spec_vmap" -> (vmapped bad_spec_cv [ s0 ], [ m ])
  | "seq_vmap" ->
      ( vmapped seq [ s0; s0 ],
        [ m; G.take y [ G.Slice { start = None; stop = Some 2; step = None } ] ]
      )
  | "seq_vmap_in0" -> (vmapped seq [ s0; None ], [ m; first y ])
  | other -> failwith ("docs_p1: vmap case " ^ other)

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Ojax.Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "docs_p1: tracer result"

let text_of vs =
  String.concat "\n"
    (List.map
       (fun v ->
         let nd = host v in
         Printf.sprintf "%s [%s] %s"
           (Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd))
           (String.concat ", "
              (List.map string_of_int (Array.to_list (Ojax.Ndarray.shape nd))))
           (Parity_cases.hex_of_nd nd))
       vs)

let error_text e =
  match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
  | Some c, Some m -> "raise " ^ c ^ ": " ^ m
  | _ -> "raise OCaml: " ^ Printexc.to_string e

let protect f = try f () with e -> error_text e

let value_eval_case case =
  match case with
  | "f_vmap" ->
      let m = m () in
      (vmapped f_cv [ Some 0; Some 0 ], [ m; F.add m (pyf 1.0) ])
  | c -> vmap_case c

let inexpressible = [ "norule_call" ]

let inexpressible_proof =
  "(d) not expressible: Custom_batching.custom_vmap takes ~rule as a required \
   argument, so a custom_vmap function with no def_vmap rule cannot be built \
   in OCaml; JAX raises AttributeError only when such a function is called"

let read_rows path =
  let ic = open_in path in
  let rec loop acc =
    match input_line ic with
    | line -> loop (Yojson.Safe.from_string line :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  loop []

let str_opt row field =
  match U.member field row with `String s -> Some s | _ -> None

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  let case = U.to_string (U.member "case" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  base_row := row;
  Fun.protect ~finally:(fun () ->
      base_row := `Null;
      Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log ?(name = id) ?(defect = false) jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(String.escaped jax ^ "\t" ^ String.escaped ojax)
      ~suite name 1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical
       else if defect then "jax_defect"
       else "differ")
  in
  match U.to_string (U.member "group" row) with
  | "residuals" ->
      let ours =
        protect (fun () ->
            let f, args = residual_case case in
            C.saved_residuals_text f args)
      in
      let expected k =
        match (str_opt row k, str_opt row "error_class") with
        | Some t, _ -> t
        | None, Some c ->
            "raise " ^ c ^ ": "
            ^ Option.value (str_opt row "message") ~default:""
        | None, None -> ""
      in
      log (expected "text_noframe") ours;
      let raw = expected "text" in
      log ~name:(id ^ ":raw")
        ~defect:(raw <> ours && strip_frames raw = ours)
        raw ours;
      ignore (frames_proof, inexpressible_proof)
  | _ ->
      let field k = Option.value (str_opt row k) ~default:"" in
      let log ?name jax ojax =
        log ?name ~defect:(List.mem case inexpressible) jax ojax
      in
      Ojax.Api.clear_caches ();
      log ~name:(id ^ ":value") (field "value")
        (protect (fun () ->
             let f, args = value_eval_case case in
             text_of (f args)));
      log ~name:(id ^ ":jaxpr") (field "jaxpr")
        (protect (fun () ->
             let f, args = vmap_case case in
             Ojax.Pretty_printer.closed_jaxpr_to_jax_string
               (Ojax.Jaxpr.make_jaxpr (List.map Ojax.Core.get_aval args) f)));
      log ~name:(id ^ ":lowered") (field "lowered")
        (protect (fun () ->
             let f, args = vmap_case case in
             let wrapped =
               Ojax.Stages.jit ~name:"_lambda" (fun xs ->
                   match f (TR.tree_leaves (TR.Tuple xs)) with
                   | [ v ] -> TR.Leaf v
                   | vs -> TR.Tuple (leaves vs))
             in
             Ojax.Stages.as_text
               (wrapped.Ojax.Stages.wrapped_lower (leaves args))))

let path ~goldens_root =
  Filename.concat (Filename.concat goldens_root name) "cases.jsonl"

let suites ~goldens_root =
  if not (Sys.file_exists (path ~goldens_root)) then []
  else
    let rows = read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              let x64 = U.to_bool (U.member "x64" row) in
              if mode = "x64_on" = x64 then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
