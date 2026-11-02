module U = Yojson.Safe.Util
module T = Ojax.Types
module L = Ojax.Lax_api
module F = Ojax.Numpy.Ufuncs

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type result = {
  kind : string;
  cls : string;
  message : string;
  outs : (string * int list * string) list;
  warnings : (string * string) list;
  notices : string list;
}

let operand shape dtype offset =
  let n = Array.fold_left ( * ) 1 shape in
  let vals = Array.init n (fun i -> float_of_int ((i mod 5) + 1 + offset)) in
  T.Concrete (Ojax.Ndarray.of_floats dtype shape vals)

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Ojax.Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "jit_opts: expected a concrete result"

let encode v =
  let nd = host v in
  ( Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd),
    Array.to_list (Ojax.Ndarray.shape nd),
    Parity_cases.hex_of_nd nd )

let run_case thunk =
  let notices = ref [] in
  let saved = !Ojax.Debug_special.notice_sink in
  (Ojax.Debug_special.notice_sink := fun s -> notices := s :: !notices);
  let outcome, warnings =
    Fun.protect
      ~finally:(fun () -> Ojax.Debug_special.notice_sink := saved)
      (fun () ->
        Ojax.Py_warnings.catch (fun () ->
            match thunk () with
            | outs -> Ok (List.map encode outs)
            | exception e -> Error e))
  in
  let base =
    {
      kind = "ok";
      cls = "";
      message = "";
      outs = [];
      warnings;
      notices = List.rev !notices;
    }
  in
  match outcome with
  | Ok outs -> { base with outs }
  | Error e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some message -> { base with kind = "raise"; cls; message }
      | _ ->
          {
            base with
            kind = "raise";
            cls = "OCaml";
            message = Printexc.to_string e;
          })

let rank_fn = function
  | "add" -> F.add
  | "subtract" -> F.subtract
  | "multiply" -> F.multiply
  | "maximum" -> F.maximum
  | "minimum" -> F.minimum
  | "arctan2" -> F.arctan2
  | "float_power" -> F.float_power
  | "nextafter" -> F.nextafter
  | "equal" -> F.equal
  | "not_equal" -> F.not_equal
  | "greater" -> F.greater
  | "greater_equal" -> F.greater_equal
  | "less" -> F.less
  | "less_equal" -> F.less_equal
  | "bitwise_and" -> F.bitwise_and
  | "bitwise_or" -> F.bitwise_or
  | "bitwise_xor" -> F.bitwise_xor
  | "left_shift" -> F.left_shift
  | "right_shift" -> F.right_shift
  | "logical_and" -> F.logical_and
  | "logical_or" -> F.logical_or
  | "logical_xor" -> F.logical_xor
  | "true_divide" -> F.true_divide
  | "power" -> F.power
  | "floor_divide" -> F.floor_divide
  | "remainder" -> F.remainder
  | "fmod" -> F.fmod
  | "copysign" -> F.copysign
  | "hypot" -> F.hypot
  | "logaddexp" -> F.logaddexp
  | "logaddexp2" -> F.logaddexp2
  | "heaviside" -> F.heaviside
  | name -> failwith ("jit_opts: unknown function " ^ name)

let f32 vals =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| Array.length vals |] vals)

let jit1 f args =
  match
    Ojax.Api.jit
      (fun xs ->
        match xs with
        | [ Ojax.Tree_util.Leaf a ] -> f a
        | _ -> failwith "jit_opts: arity")
      (List.map (fun a -> Ojax.Tree_util.Leaf a) args)
  with
  | t -> Ojax.Tree_util.tree_leaves t

let leaf v = Ojax.Tree_util.Leaf v

let special_thunk name =
  let x = f32 [| -1.0; 2.0 |]
  and z = f32 [| 0.0 |]
  and n = f32 [| Float.nan |]
  and one = f32 [| 1.0 |] in
  match name with
  | "eager_log" -> fun () -> [ L.log x ]
  | "eager_sqrt" -> fun () -> [ L.sqrt x ]
  | "eager_div_zero_zero" -> fun () -> [ L.div z z ]
  | "eager_div_one_zero" -> fun () -> [ L.div one z ]
  | "eager_add_nan_input" -> fun () -> [ L.add n one ]
  | "jit_log" -> fun () -> jit1 (fun a -> leaf (L.mul (L.log a) a)) [ x ]
  | "jit_sqrt_sum" ->
      fun () -> jit1 (fun a -> leaf (L.reduce_sum (L.sqrt a) [| 0 |])) [ x ]
  | "jit_identity_nan" -> fun () -> jit1 (fun a -> leaf a) [ n ]
  | "jit_pair_nan" ->
      fun () -> jit1 (fun a -> Ojax.Tree_util.Tuple [ leaf a; leaf a ]) [ n ]
  | "jit_nested_log" ->
      fun () ->
        jit1
          (fun a ->
            match jit1 (fun b -> leaf (L.log b)) [ a ] with
            | [ r ] -> leaf r
            | _ -> failwith "jit_opts: arity")
          [ x ]
  | "jit_div_one_zero" -> fun () -> jit1 (fun a -> leaf (L.div one a)) [ z ]
  | "jit_clean" -> fun () -> jit1 (fun a -> leaf (L.exp a)) [ x ]
  | "eager_int_div" ->
      fun () ->
        let i v =
          T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 1 |] [| v |])
        in
        [ L.div (i 1.0) (i 0.0) ]
  | other -> failwith ("jit_opts: unknown case " ^ other)

let py_float x =
  Ojax.Core.py_scalar
    (if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32)
    x

let leaves_of = function
  | Ojax.Tree_util.Tuple l -> List.map (fun t -> Ojax.Tree_util.tree_leaves t) l
  | t -> [ Ojax.Tree_util.tree_leaves t ]

let two f = function
  | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] -> f a b
  | _ -> failwith "jit_opts: arity"

let one f = function
  | [ Ojax.Tree_util.Leaf a ] -> f a
  | _ -> failwith "jit_opts: arity"

let ones shape =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape
       (Array.make (Array.fold_left ( * ) 1 shape) 1.0))

let pair a b = Ojax.Tree_util.Tuple [ leaf a; leaf b ]

let lowered_text ?donate_argnums ?keep_unused f args =
  let wrapped = Ojax.Stages.jit ?donate_argnums ?keep_unused f in
  Ojax.Stages.as_text (wrapped.Ojax.Stages.wrapped_lower (List.map leaf args))

let inner_sin ?inline () = Ojax.Api.jit ?inline (one (fun b -> leaf (F.sin b)))

let text_thunk name =
  ignore leaves_of;
  let x3 = ones [| 3 |] and y2 = ones [| 2 |] and m23 = ones [| 2; 3 |] in
  let two_f = two and mul2 a = F.multiply a (py_float 2.0) in
  match name with
  | "don_mul" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (two_f (fun a _ -> leaf (mul2 a)))
          [ x3; y2 ]
  | "keep_unused" ->
      fun () ->
        lowered_text ~keep_unused:true
          (two_f (fun a _ -> leaf (mul2 a)))
          [ x3; y2 ]
  | "don_pair" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a b -> pair (mul2 a) (F.add b (py_float 1.0))))
          [ x3; y2 ]
  | "don_sum" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a _ -> leaf (Ojax.Numpy.Reductions.sum a)))
          [ x3; y2 ]
  | "don_reshape" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (one (fun a -> leaf (Ojax.Numpy.Lax_numpy.reshape a [| 6 |])))
          [ m23 ]
  | "don_astype" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (one (fun a -> leaf (Ojax.Numpy.Lax_numpy.astype a Ojax.Dtype.I32)))
          [ m23 ]
  | "don_weak" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a s -> leaf (F.multiply a s)))
          [ y2; py_float 2.0 ]
  | "don_swap" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (two_f (fun a b -> pair b a))
          [ y2; y2 ]
  | "don_dup" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a _ -> pair a a))
          [ y2; y2 ]
  | "keep_don" ->
      fun () ->
        lowered_text ~donate_argnums:[ 1 ] ~keep_unused:true
          (two_f (fun a _ -> leaf a))
          [ y2; y2 ]
  | "keep_unused_none" ->
      fun () ->
        lowered_text ~keep_unused:false
          (two_f (fun a _ -> leaf (F.add a (py_float 1.0))))
          [ y2; x3 ]
  | "don_none" ->
      fun () ->
        lowered_text ~donate_argnums:[]
          (two_f (fun a b -> leaf (F.add a b)))
          [ y2; y2 ]
  | "inline_lowered" | "noinline_lowered" ->
      let inline = name = "inline_lowered" in
      fun () ->
        lowered_text
          (one (fun a ->
               match
                 Ojax.Tree_util.tree_leaves (inner_sin ~inline () [ leaf a ])
               with
               | [ r ] -> leaf (F.add r (py_float 1.0))
               | _ -> failwith "jit_opts: arity"))
          [ x3 ]
  | "inline_jaxpr" | "noinline_jaxpr" ->
      let inline = name = "inline_jaxpr" in
      fun () ->
        Ojax.Pretty_printer.closed_jaxpr_to_jax_string
          (Ojax.Api.make_jaxpr
             (one (fun a ->
                  match
                    Ojax.Tree_util.tree_leaves (inner_sin ~inline () [ leaf a ])
                  with
                  | [ r ] -> leaf (F.add r (py_float 1.0))
                  | _ -> failwith "jit_opts: arity"))
             [ leaf x3 ])
  | other -> failwith ("jit_opts: unknown text case " ^ other)

let int_seed v =
  let dt =
    if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32
  in
  T.Concrete (Ojax.Ndarray.of_floats dt [||] [| v |])

let random_thunk name =
  let module R = Ojax.Random.Core in
  match name with
  | "key_data" -> fun () -> [ R.key_data (R.key (int_seed 11.0)) ]
  | "prng_key" -> fun () -> [ R.prng_key (int_seed 11.0) ]
  | "uniform" ->
      fun () ->
        [
          R.uniform
            (R.key (int_seed 3.0))
            ~shape:[| 4 |] ~minval:0.0 ~maxval:1.0;
        ]
  | "jit_key_data" ->
      fun () ->
        jit1
          (fun s -> leaf (R.key_data (R.key s)))
          [ T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| 11.0 |]) ]
  | "gumbel" -> fun () -> [ R.gumbel (R.key (int_seed 1.0)) ~shape:[| 5 |] ]
  | "gumbel_f64" ->
      fun () ->
        [ R.gumbel ~dtype:Ojax.Dtype.F64 (R.key (int_seed 1.0)) ~shape:[| 5 |] ]
  | "categorical" ->
      fun () ->
        let logits =
          T.Concrete
            (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 2; 4 |]
               [| 0.1; -2.0; 3.0; 0.5; 1.0; 1.0; -30.0; 2.0 |])
        in
        [ R.categorical_full ~shape:[| 6; 2 |] (R.key (int_seed 2.0)) logits ]
  | other -> failwith ("jit_opts: unknown random case " ^ other)

let strings j = List.map U.to_string (U.to_list j)

let expected_of row =
  let kind = U.to_string (U.member "kind" row) in
  let str k = match U.member k row with `String s -> s | _ -> "" in
  {
    kind;
    cls = str "class";
    message =
      (if kind = "ok" && str "group" = "text" then str "text" else str "message");
    outs =
      (match U.member "outs" row with
      | `List l ->
          List.map
            (fun o ->
              ( U.to_string (U.member "dtype" o),
                List.map U.to_int (U.to_list (U.member "shape" o)),
                U.to_string (U.member "hex" o) ))
            l
      | _ -> []);
    warnings =
      List.map
        (fun w ->
          match strings w with
          | [ c; m ] -> (c, m)
          | _ -> failwith "jit_opts: warning entry")
        (U.to_list (U.member "warnings" row));
    notices = strings (U.member "notices" row);
  }

let actual_of row =
  match U.to_string (U.member "group" row) with
  | "rank" ->
      let fn = U.to_string (U.member "fn" row) in
      let mode = U.to_string (U.member "mode" row) in
      let dtype =
        Option.get
          (Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" row)))
      in
      let shapes =
        List.map
          (fun s -> Array.of_list (List.map U.to_int (U.to_list s)))
          (U.to_list (U.member "shapes" row))
      in
      let a, b =
        match shapes with
        | [ sa; sb ] -> (operand sa dtype 0, operand sb dtype 1)
        | _ -> failwith "jit_opts: shapes"
      in
      Ojax.Config.with_value Ojax.Config.numpy_rank_promotion mode (fun () ->
          run_case (fun () -> [ rank_fn fn a b ]))
  | "special" ->
      let nans, infs =
        match U.to_string (U.member "flag" row) with
        | "nans" -> (true, false)
        | "infs" -> (false, true)
        | _ -> (true, true)
      in
      let thunk = special_thunk (U.to_string (U.member "case" row)) in
      Ojax.Config.with_value Ojax.Config.debug_nans nans (fun () ->
          Ojax.Config.with_value Ojax.Config.debug_infs infs (fun () ->
              run_case thunk))
  | "random" ->
      let arg = U.to_string (U.member "arg" row) in
      let offset = Option.value (int_of_string_opt arg) ~default:0 in
      let thunk = random_thunk (U.to_string (U.member "case" row)) in
      Ojax.Config.with_value Ojax.Config.random_seed_offset offset (fun () ->
          Ojax.Config.with_value Ojax.Config.high_dynamic_range_gumbel
            (arg = "high") (fun () -> run_case thunk))
  | "text" ->
      let thunk = text_thunk (U.to_string (U.member "case" row)) in
      let text = ref "" in
      let r =
        run_case (fun () ->
            text := thunk ();
            [])
      in
      { r with message = (if r.kind = "ok" then !text else r.message) }
  | g -> failwith ("jit_opts: unknown group " ^ g)

let same_main e a =
  e.kind = a.kind && e.cls = a.cls && e.message = a.message && e.outs = a.outs
  && e.warnings = a.warnings

let record suite name same =
  Compare.bump suite same;
  Compare.bump (Compare.raw_prefix ^ suite) same;
  Compare.note_row suite name ~bit:same ~raw:same;
  match Lazy.force Compare.exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s\n%!" name 1
          (if same then 0 else 1)
          0.0 suite
          (if same then Compare.raw_identical else "differ")
      with Sys_error _ -> ())

let show r =
  Printf.sprintf "%s %s %S outs=%d warnings=[%s] notices=%d" r.kind r.cls
    r.message (List.length r.outs)
    (String.concat "; " (List.map (fun (c, m) -> c ^ ": " ^ m) r.warnings))
    (List.length r.notices)

let check_row ~x64 row () =
  let id = U.to_string (U.member "id" row) in
  let expected = expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = "jit_opts:" ^ if x64 then "x64_on" else "x64_off" in
  let main = same_main expected actual in
  let notices = expected.notices = actual.notices in
  record suite id main;
  record ("jit_opts_notices:" ^ if x64 then "x64_on" else "x64_off") id notices;
  if not main then
    Printf.eprintf "JIT_OPTS DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "jit_opts") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "jit_opts_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~x64 row))
      rows )
