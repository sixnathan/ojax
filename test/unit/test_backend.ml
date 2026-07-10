module Api = Ojax.Api
module Backend = Ojax.Backend
module J = Ojax.Jaxpr
module Core = Ojax.Core
module T = Ojax.Types
module Tree = Ojax.Tree_util
module Dt = Ojax.Dtype
module Nd = Ojax.Ndarray
module A = Ojax.Pjrt.Abi
module B = Ojax.Pjrt.Buffer
module D = Ojax.Pjrt.Discover

let () = Ojax.Lax.install ()

let plugin_present =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> true
  | _ -> false

let () = if plugin_present then Unix.putenv "OJAX_BACKEND" "xla"

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

let add_mul_jaxpr dtype =
  J.make_jaxpr
    [ av [| 3 |] dtype; av [| 3 |] dtype ]
    (fun args ->
      match args with
      | [ x; y ] ->
          let s = Core.bind T.Add [ x; y ] in
          Core.bind T.Mul (s @ [ y ])
      | _ -> assert false)

let host_eval cj inputs =
  List.map
    (function T.Concrete nd -> nd | _ -> assert false)
    (J.eval_closed_jaxpr cj (List.map (fun nd -> T.Concrete nd) inputs))

let concrete_nd = function T.Concrete nd -> nd | _ -> assert false
let mk dtype floats = Nd.canonicalize dtype (Nd.of_floats dtype [| 3 |] floats)

let check_executor dtype fx fy () =
  let x = mk dtype fx in
  let y = mk dtype fy in
  let cj = add_mul_jaxpr dtype in
  let outs =
    List.map concrete_nd (Backend.executor cj [ T.Concrete x; T.Concrete y ])
  in
  let host = host_eval cj [ x; y ] in
  match (outs, host) with
  | [ o ], [ h ] ->
      Alcotest.(check bool) "dtype" true (Nd.dtype o = Nd.dtype h);
      Alcotest.(check (array int)) "shape" (Nd.shape h) (Nd.shape o);
      Alcotest.(check (array (float 1e-6)))
        "values" (flat_floats h) (flat_floats o)
  | _ -> Alcotest.fail "output arity"

let interpreter_seam () =
  let dtype = Dt.F32 in
  let x = mk dtype [| 1.5; 2.5; -3.0 |] in
  let y = mk dtype [| 4.0; -0.5; 2.0 |] in
  let cj = add_mul_jaxpr dtype in
  let compiled = Backend.Interpreter.compile cj in
  let bx = Backend.Interpreter.of_host x in
  let by = Backend.Interpreter.of_host y in
  let outs = Backend.Interpreter.execute compiled [ bx; by ] in
  let host = host_eval cj [ x; y ] in
  match (outs, host) with
  | [ o ], [ h ] ->
      Alcotest.(check (array (float 1e-6)))
        "values"
        (flat_floats (Backend.Interpreter.to_host h))
        (flat_floats (Backend.Interpreter.to_host o))
  | _ -> Alcotest.fail "output arity"

let xla_seam () =
  if not plugin_present then ()
  else
    let dtype = Dt.F32 in
    let x = mk dtype [| 1.5; 2.5; -3.0 |] in
    let y = mk dtype [| 4.0; -0.5; 2.0 |] in
    let cj = add_mul_jaxpr dtype in
    let compiled = Backend.Xla.compile cj in
    Fun.protect
      ~finally:(fun () -> Ojax.Pjrt.Executable.destroy compiled)
      (fun () ->
        let bx = Backend.Xla.of_host x in
        let by = Backend.Xla.of_host y in
        let outs = Backend.Xla.execute compiled [ bx; by ] in
        B.destroy bx;
        B.destroy by;
        let host = host_eval cj [ x; y ] in
        match (outs, host) with
        | [ o ], [ h ] ->
            let dev = Backend.Xla.to_host o in
            B.destroy o;
            Alcotest.(check bool) "dtype" true (Nd.dtype dev = Nd.dtype h);
            Alcotest.(check (array int)) "shape" (Nd.shape h) (Nd.shape dev);
            Alcotest.(check (array (float 1e-6)))
              "values" (flat_floats h) (flat_floats dev)
        | _ -> Alcotest.fail "output arity")

let sq (args : T.value Tree.t list) : T.value Tree.t =
  match args with
  | [ Tree.Leaf x ] -> Tree.Leaf (List.hd (Core.bind T.Mul [ x; x ]))
  | _ -> assert false

let leaf nd = Tree.Leaf (T.Concrete nd)

let jit_concrete () =
  let x = mk Dt.F32 [| 2.0; 3.0; 4.0 |] in
  match Api.jit sq [ leaf x ] with
  | Tree.Leaf v ->
      let o = concrete_nd v in
      Alcotest.(check (array (float 1e-6)))
        "square" [| 4.0; 9.0; 16.0 |] (flat_floats o)
  | _ -> Alcotest.fail "expected leaf"

let jit_under_jvp () =
  let x = mk Dt.F32 [| 2.0; 3.0; 4.0 |] in
  let t = mk Dt.F32 [| 1.0; 1.0; 1.0 |] in
  let jsq = Api.jit sq in
  let p, d = Api.jvp jsq [ leaf x ] [ leaf t ] in
  match (p, d) with
  | Tree.Leaf pv, Tree.Leaf dv ->
      Alcotest.(check (array (float 1e-6)))
        "primal" [| 4.0; 9.0; 16.0 |]
        (flat_floats (concrete_nd pv));
      Alcotest.(check (array (float 1e-6)))
        "tangent" [| 4.0; 6.0; 8.0 |]
        (flat_floats (concrete_nd dv))
  | _ -> Alcotest.fail "expected leaves"

let leak_smoke () =
  if not plugin_present then ()
  else begin
    let cj = add_mul_jaxpr Dt.F32 in
    let run = Backend.executor cj in
    let x = mk Dt.F32 [| 1.0; 2.0; 3.0 |] in
    let y = mk Dt.F32 [| 4.0; 5.0; 6.0 |] in
    ignore (run [ T.Concrete x; T.Concrete y ]);
    Gc.full_major ();
    let before = A.maxrss_bytes () in
    for i = 1 to 1000 do
      ignore (run [ T.Concrete x; T.Concrete y ]);
      if i mod 100 = 0 then Gc.full_major ()
    done;
    Gc.full_major ();
    let after = A.maxrss_bytes () in
    let growth = after - before in
    Alcotest.(check bool)
      (Printf.sprintf "rss growth %d bytes bounded" growth)
      true
      (growth < 64 * 1024 * 1024)
  end

let () =
  Alcotest.run "backend"
    [
      ( "executor",
        [
          Alcotest.test_case "f32 add_mul" `Quick
            (check_executor Dt.F32 [| 1.5; 2.5; -3.0 |] [| 4.0; -0.5; 2.0 |]);
          Alcotest.test_case "i32 add_mul" `Quick
            (check_executor Dt.I32 [| 1.0; -2.0; 3.0 |] [| 4.0; 5.0; -6.0 |]);
          Alcotest.test_case "jit concrete" `Quick jit_concrete;
          Alcotest.test_case "jit under jvp" `Quick jit_under_jvp;
        ] );
      ( "seam",
        [
          Alcotest.test_case "interpreter" `Quick interpreter_seam;
          Alcotest.test_case "xla" `Quick xla_seam;
        ] );
      ("leak", [ Alcotest.test_case "executor 1000x" `Slow leak_smoke ]);
    ]
