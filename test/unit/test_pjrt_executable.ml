module A = Ojax.Pjrt.Abi
module C = Ojax.Pjrt.Client
module B = Ojax.Pjrt.Buffer
module E = Ojax.Pjrt.Executable
module D = Ojax.Pjrt.Discover
module Emit = Ojax.Stablehlo.Emit
module J = Ojax.Jaxpr
module Core = Ojax.Core
module T = Ojax.Types
module Dt = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()

let real_plugin_path () =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> Some p
  | _ -> None

let shared =
  lazy
    (match real_plugin_path () with
    | None -> None
    | Some path ->
        let plugin = A.open_plugin path in
        Some (plugin, C.create plugin))

let get_client () =
  match Lazy.force shared with Some (_, c) -> Some c | None -> None

let unravel shape flat =
  let r = Array.length shape in
  let idx = Array.make r 0 in
  let rem = ref flat in
  for i = r - 1 downto 0 do
    idx.(i) <- !rem mod shape.(i);
    rem := !rem / shape.(i)
  done;
  idx

let flat_floats nd =
  let shape = Nd.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  Array.init n (fun i -> Nd.get_f nd (unravel shape i))

let av shape dtype : T.aval = { shape; dtype; weak_type = false }

let two_op_jaxpr dtype =
  J.make_jaxpr
    [ av [| 3 |] dtype; av [| 3 |] dtype ]
    (fun args ->
      match args with
      | [ x; y ] ->
          let s = Core.bind T.Add [ x; y ] in
          Core.bind T.Mul (s @ [ y ])
      | _ -> assert false)

let host_eval jaxpr inputs =
  let vals = List.map (fun nd -> T.Concrete nd) inputs in
  match J.eval_closed_jaxpr jaxpr vals with
  | [ T.Concrete nd ] -> nd
  | _ -> assert false

let smoke () =
  match get_client () with
  | None -> ()
  | Some client ->
      let dtype = Dt.F32 in
      let x =
        Nd.canonicalize dtype (Nd.of_floats dtype [| 3 |] [| 1.5; 2.5; -3.0 |])
      in
      let y =
        Nd.canonicalize dtype (Nd.of_floats dtype [| 3 |] [| 4.0; -0.5; 2.0 |])
      in
      let jaxpr = two_op_jaxpr dtype in
      let text = Emit.emit_closed_jaxpr jaxpr in
      let exec = E.compile client text in
      Fun.protect
        ~finally:(fun () -> E.destroy exec)
        (fun () ->
          Alcotest.(check int) "num_outputs" 1 (E.num_outputs exec);
          let bx = B.of_host client x in
          let by = B.of_host client y in
          let outs = E.execute exec [| bx; by |] in
          B.destroy bx;
          B.destroy by;
          Alcotest.(check int) "output arity" 1 (Array.length outs);
          let dev = B.to_host outs.(0) in
          B.destroy outs.(0);
          let host = host_eval jaxpr [ x; y ] in
          Alcotest.(check bool) "dtype" true (Nd.dtype dev = dtype);
          Alcotest.(check (array int)) "shape" (Nd.shape host) (Nd.shape dev);
          Alcotest.(check (array (float 1e-6)))
            "values" (flat_floats host) (flat_floats dev))

let leak_smoke () =
  match get_client () with
  | None -> ()
  | Some client ->
      let dtype = Dt.F32 in
      let x =
        Nd.canonicalize dtype (Nd.of_floats dtype [| 3 |] [| 1.0; 2.0; 3.0 |])
      in
      let y =
        Nd.canonicalize dtype (Nd.of_floats dtype [| 3 |] [| 4.0; 5.0; 6.0 |])
      in
      let jaxpr = two_op_jaxpr dtype in
      let exec = E.compile client (Emit.emit_closed_jaxpr jaxpr) in
      Fun.protect
        ~finally:(fun () -> E.destroy exec)
        (fun () ->
          Gc.full_major ();
          let before = A.maxrss_bytes () in
          for i = 1 to 1000 do
            let bx = B.of_host client x in
            let by = B.of_host client y in
            let outs = E.execute exec [| bx; by |] in
            B.destroy bx;
            B.destroy by;
            Array.iter B.destroy outs;
            if i mod 100 = 0 then Gc.full_major ()
          done;
          Gc.full_major ();
          let after = A.maxrss_bytes () in
          let growth = after - before in
          Alcotest.(check bool)
            (Printf.sprintf "rss growth %d bytes bounded" growth)
            true
            (growth < 64 * 1024 * 1024))

let () =
  Alcotest.run "pjrt_executable"
    [
      ("compile_execute", [ Alcotest.test_case "two_op_f32" `Quick smoke ]);
      ("leak", [ Alcotest.test_case "execute 1000x" `Slow leak_smoke ]);
    ]
