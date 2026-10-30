open Ojax
open Ojax.Types
module L = Lax_api
module Tree = Tree_util

let () = Lax.install ()
let f32 shape xs = Concrete (Ndarray.of_floats Dtype.F32 shape xs)
let scalar_aval = { shape = [| 2 |]; dtype = Dtype.F32; weak_type = false }

let nd = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let doubled = function
  | [ x ] -> [ L.mul x (L.full_like x 2.0) ]
  | _ -> Alcotest.fail "arity"

let plugin_present () = Sys.getenv_opt "OJAX_PJRT_PLUGIN" <> None

let stages () =
  let lowered = Stages.lower doubled [ scalar_aval ] in
  let text = Stages.as_text lowered in
  Alcotest.(check bool) "stablehlo text" true (String.length text > 0);
  let compiled = Stages.compile lowered in
  (match Stages.call compiled [ f32 [| 2 |] [| 1.0; 2.0 |] ] with
  | [ out ] ->
      Alcotest.(check (float 1e-6)) "value" 4.0 (Ndarray.get_f (nd out) [| 1 |])
  | _ -> Alcotest.fail "arity");
  match Stages.call compiled [ f32 [| 3 |] [| 1.0; 2.0; 3.0 |] ] with
  | _ -> Alcotest.fail "expected a type mismatch"
  | exception Invalid_argument _ -> ()

let export_roundtrip () =
  let exported = Export.export doubled [ scalar_aval ] in
  let restored = Export.deserialize (Export.serialize exported) in
  Alcotest.(check string)
    "module survives" exported.Export.mlir_module restored.Export.mlir_module;
  Alcotest.(check (array int))
    "input type" [| 2 |] (List.hd restored.Export.in_avals).shape;
  if plugin_present () then
    match Export.call restored [ f32 [| 2 |] [| 1.0; 2.0 |] ] with
    | [ out ] ->
        Alcotest.(check (float 1e-6))
          "call" 4.0
          (Ndarray.get_f (nd out) [| 1 |])
    | _ -> Alcotest.fail "arity"

let sharding () =
  let mesh = Sharding.make_mesh [ 1 ] [ "x" ] in
  let spec = [ Sharding.Named "x" ] in
  let s = Sharding.named_sharding mesh spec in
  Alcotest.(check int) "devices" 1 (Sharding.num_devices s);
  let x = Tree.Leaf (f32 [||] [| 3.0 |]) in
  ignore (Sharding.device_put x s);
  (match
     Sharding.shard_map
       (fun a -> List.length a)
       ~mesh ~in_specs:[ spec ] ~out_specs:[ spec ] [ 1 ]
   with
  | 1 -> ()
  | _ -> Alcotest.fail "shard_map");
  match
    Sharding.device_put x
      (Sharding.named_sharding (Sharding.make_mesh [ 4 ] [ "x" ]) spec)
  with
  | _ -> Alcotest.fail "expected too few devices"
  | exception Invalid_argument _ -> ()

let monitoring () =
  let events = ref [] in
  Monitoring.register_event_listener (fun e -> events := e :: !events);
  Monitoring.record_event "compile";
  Alcotest.(check (list string)) "event" [ "compile" ] !events;
  Monitoring.clear_event_listeners ();
  Monitoring.record_event "ignored";
  Alcotest.(check int) "cleared" 1 (List.length !events)

let distributed () =
  Distributed.initialize ();
  Alcotest.(check bool) "up" true (Distributed.is_initialized ());
  Distributed.shutdown ();
  Alcotest.(check bool) "down" false (Distributed.is_initialized ())

let shape_tuples () =
  Alcotest.(check (array int))
    "conv" [| 1; 3; 5 |]
    (L.conv_shape_tuple [| 1; 2; 7 |] [| 3; 2; 3 |] [| 1 |] [| (0, 0) |]);
  Alcotest.(check (array int))
    "window" [| 2; 3 |]
    (L.reduce_window_shape_tuple [| 2; 6 |] [| 1; 2 |] [| 1; 2 |]
       [| (0, 0); (0, 0) |]
       ())

let () =
  Alcotest.run "runtime_surface"
    [
      ( "surface",
        [
          Alcotest.test_case "stages" `Quick stages;
          Alcotest.test_case "export" `Quick export_roundtrip;
          Alcotest.test_case "sharding" `Quick sharding;
          Alcotest.test_case "monitoring" `Quick monitoring;
          Alcotest.test_case "distributed" `Quick distributed;
          Alcotest.test_case "shape tuples" `Quick shape_tuples;
        ] );
    ]
