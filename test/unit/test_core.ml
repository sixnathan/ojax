module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let scalar dtype v = Nd.of_floats dtype [| 1 |] [| v |]
let read nd = Nd.get_f nd [| 0 |]

let concrete_of = function
  | T.Concrete a -> a
  | T.Tracer _ -> failwith "not concrete"
  | T.Device _ -> failwith "not concrete"

let install_impl () =
  C.rules.impl <-
    (fun prim inputs ->
      match (prim, inputs) with
      | T.Neg, [ a ] -> [ Nd.map (Nd.dtype a) (fun x -> -.x) a ]
      | T.Sin, [ a ] -> [ Nd.map (Nd.dtype a) sin a ]
      | T.Add, [ a; b ] -> [ Nd.map2 (Nd.dtype a) ( +. ) a b ]
      | _ -> failwith "unexpected prim in test impl")

let float = Alcotest.(float 1e-12)

let eval_neg () =
  install_impl ();
  let out = C.bind1 T.Neg [ T.Concrete (scalar D.F64 3.0) ] in
  Alcotest.check float "neg 3.0" (-3.0) (read (concrete_of out))

let eval_add () =
  install_impl ();
  let out =
    C.bind1 T.Add
      [ T.Concrete (scalar D.F64 2.0); T.Concrete (scalar D.F64 5.0) ]
  in
  Alcotest.check float "2 + 5" 7.0 (read (concrete_of out))

let canonicalize_in_bind () =
  install_impl ();
  let out = C.bind1 T.Sin [ T.Concrete (scalar D.F32 1.0) ] in
  let expected = Int32.float_of_bits (Int32.bits_of_float (sin 1.0)) in
  Alcotest.check float "sin rounded to f32 by bind" expected
    (read (concrete_of out))

let bind_multi_out_rejected () =
  C.rules.impl <-
    (fun _ inputs ->
      match inputs with [ a ] -> [ a; a ] | _ -> failwith "bad");
  Alcotest.check_raises "bind1 rejects multiple outputs"
    (Invalid_argument "bind1: expected a single output") (fun () ->
      ignore (C.bind1 T.Neg [ T.Concrete (scalar D.F64 1.0) ]))

let fresh_id_monotonic () =
  let a = C.fresh_id () in
  let b = C.fresh_id () in
  let c = C.fresh_id () in
  Alcotest.(check bool) "strictly increasing" true (a < b && b < c)

let get_aval_concrete () =
  let av =
    C.get_aval (T.Concrete (Nd.of_floats D.F32 [| 2; 3 |] (Array.make 6 0.0)))
  in
  Alcotest.(check (array int)) "shape" [| 2; 3 |] av.T.shape;
  Alcotest.(check bool) "dtype f32" true (av.T.dtype = D.F32);
  Alcotest.(check bool) "not weak" false av.T.weak_type

let with_new_main_levels () =
  let outer = C.with_new_main T.KJVP T.GNone (fun m -> m.T.level) in
  Alcotest.(check int) "first new_main is level 1" 1 outer;
  let nested =
    C.with_new_main T.KJVP T.GNone (fun _ ->
        C.with_new_main T.KBatch T.GNone (fun m -> m.T.level))
  in
  Alcotest.(check int) "nested new_main is level 2" 2 nested;
  let after = C.with_new_main T.KJVP T.GNone (fun m -> m.T.level) in
  Alcotest.(check int) "stack popped: back to level 1" 1 after

let find_top_trace_picks_highest () =
  let captured = C.with_new_main T.KJVP T.GNone (fun m -> m) in
  let av = { T.shape = [||]; dtype = D.F64; weak_type = false } in
  let tracer =
    T.Tracer
      { T.id = C.fresh_id (); trace = captured; aval = av; payload = T.Eval }
  in
  let top = C.find_top_trace [ T.Concrete (scalar D.F64 1.0); tracer ] in
  Alcotest.(check int) "top trace is the JVP main (level 1)" 1 top.T.level;
  let top_no_tracer = C.find_top_trace [ T.Concrete (scalar D.F64 1.0) ] in
  Alcotest.(check int)
    "no tracer: bottom eval trace level 0" 0 top_no_tracer.T.level

let register_interpreter_dispatch () =
  C.register_interpreter T.KBatch
    {
      C.i_pure = (fun _ v -> v);
      i_lift = (fun _ v -> v);
      i_full_lower = (fun v -> v);
      i_process_primitive = (fun _ _ _ -> [ T.Concrete (scalar D.F64 42.0) ]);
      i_process_custom_jvp = (fun _ ~primal ~jvp:_ args -> primal args);
      i_process_custom_vjp = (fun _ ~primal ~fwd:_ ~bwd:_ args -> primal args);
    };
  let batch_main = C.with_new_main T.KBatch T.GNone (fun m -> m) in
  let out = C.process_primitive batch_main T.Neg [] in
  match out with
  | [ v ] ->
      Alcotest.check float "batch interpreter dispatched" 42.0
        (read (concrete_of v))
  | _ -> Alcotest.fail "expected one output"

let dynamic_trace_override () =
  C.with_new_main T.KJaxpr T.GNone (fun main ->
      C.new_dynamic main (fun () ->
          let top = C.find_top_trace [] in
          Alcotest.(check int)
            "dynamic trace overrides with no tracer args" main.T.level
            top.T.level));
  let restored = C.find_top_trace [] in
  Alcotest.(check int) "dynamic trace restored after scope" 0 restored.T.level

let full_raise_cannot_lift_higher () =
  C.with_new_main T.KJVP T.GNone (fun main ->
      let higher =
        { T.level = main.T.level + 5; kind = T.KEval; global_data = T.GNone }
      in
      let av = { T.shape = [||]; dtype = D.F64; weak_type = false } in
      let tracer =
        T.Tracer
          { T.id = C.fresh_id (); trace = higher; aval = av; payload = T.Eval }
      in
      Alcotest.check_raises "lifting a higher-level tracer is rejected"
        (Invalid_argument
           (Printf.sprintf "Can't lift level %d to %d." higher.T.level
              main.T.level))
        (fun () -> ignore (C.full_raise main tracer)))

let custom_jvp_eval_arm () =
  let et = C.find_top_trace [] in
  let primal = function [ x ] -> [ x ] | _ -> Alcotest.fail "arity" in
  let jvp _ = Alcotest.fail "jvp must not run under eval" in
  let out =
    C.process_custom_jvp et primal ~jvp [ T.Concrete (scalar D.F64 7.0) ]
  in
  match out with
  | [ v ] ->
      Alcotest.check float "custom_jvp primal passthrough" 7.0
        (read (concrete_of v))
  | _ -> Alcotest.fail "arity"

let custom_vjp_eval_arm () =
  let et = C.find_top_trace [] in
  let primal = function [ x ] -> [ x ] | _ -> Alcotest.fail "arity" in
  let fwd _ = Alcotest.fail "fwd must not run under eval" in
  let bwd _ = Alcotest.fail "bwd must not run under eval" in
  let out =
    C.process_custom_vjp et primal ~fwd ~bwd [ T.Concrete (scalar D.F64 8.0) ]
  in
  match out with
  | [ v ] ->
      Alcotest.check float "custom_vjp primal passthrough" 8.0
        (read (concrete_of v))
  | _ -> Alcotest.fail "arity"

let () =
  Alcotest.run "core"
    [
      ( "eval",
        [
          Alcotest.test_case "neg" `Quick eval_neg;
          Alcotest.test_case "add" `Quick eval_add;
          Alcotest.test_case "canonicalize in bind" `Quick canonicalize_in_bind;
          Alcotest.test_case "bind1 multi-out" `Quick bind_multi_out_rejected;
        ] );
      ( "identity",
        [ Alcotest.test_case "fresh_id monotonic" `Quick fresh_id_monotonic ] );
      ( "aval",
        [ Alcotest.test_case "get_aval concrete" `Quick get_aval_concrete ] );
      ( "trace_stack",
        [
          Alcotest.test_case "with_new_main levels" `Quick with_new_main_levels;
          Alcotest.test_case "find_top_trace" `Quick
            find_top_trace_picks_highest;
          Alcotest.test_case "register + dispatch" `Quick
            register_interpreter_dispatch;
          Alcotest.test_case "dynamic trace override" `Quick
            dynamic_trace_override;
          Alcotest.test_case "cannot lift higher" `Quick
            full_raise_cannot_lift_higher;
        ] );
      ( "custom_seams",
        [
          Alcotest.test_case "custom_jvp eval arm" `Quick custom_jvp_eval_arm;
          Alcotest.test_case "custom_vjp eval arm" `Quick custom_vjp_eval_arm;
        ] );
    ]
