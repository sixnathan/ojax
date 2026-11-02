open Ojax
open Ojax.Types
module B = Ojax.Interpreters.Batching
module Ad = Ojax.Interpreters.Ad
module P = Ojax.Parallel

let () = Lax.install ()
let vec xs = Concrete (Ndarray.of_floats Dtype.F32 [| Array.length xs |] xs)

let nd = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let close name a b = Alcotest.(check (float 1e-5)) name a b
let at v i = Ndarray.get_f (nd v) [| i |]
let sole = function [ x ] -> x | _ -> Alcotest.fail "expected one value"

let over f x =
  sole (B.vmap ~axis_name:"i" (fun a -> [ f (sole a) ]) [ Some 0 ] [ x ])

let reductions () =
  let x = vec [| 1.0; 2.0; 4.0 |] in
  let check name f expected =
    let out = over (fun v -> f v ~axis_name:"i") x in
    close name expected (at out 1)
  in
  check "psum" (fun x ~axis_name -> P.psum x ~axis_name) 7.0;
  check "pmax" (fun x ~axis_name -> P.pmax x ~axis_name) 4.0;
  check "pmin" (fun x ~axis_name -> P.pmin x ~axis_name) 1.0;
  check "pmean" (fun x ~axis_name -> P.pmean x ~axis_name) (7.0 /. 3.0)

let index_and_size () =
  let x = vec [| 5.0; 6.0; 7.0 |] in
  let out =
    over
      (fun _ ->
        Core.bind1 (Convert_element_type Dtype.F32)
          [ P.axis_index ~axis_name:"i" ])
      x
  in
  close "index" 2.0 (at out 2);
  let sized =
    over
      (fun v -> Lax_api.full_like v (float_of_int (P.axis_size ~axis_name:"i")))
      x
  in
  close "size" 3.0 (at sized 0)

let gather_and_permute () =
  let x = vec [| 1.0; 2.0 |] in
  let gathered = over (fun v -> P.all_gather v ~axis_name:"i") x in
  Alcotest.(check (array int))
    "gather shape" [| 2; 2 |] (Core.get_aval gathered).shape;
  close "gather value" 2.0 (Ndarray.get_f (nd gathered) [| 0; 1 |]);
  let swapped =
    over (fun v -> P.ppermute v ~axis_name:"i" ~perm:[ (0, 1); (1, 0) ]) x
  in
  close "permute" 2.0 (at swapped 0);
  let broadcast = over (fun v -> P.pbroadcast ~source:1 v ~axis_name:"i") x in
  close "broadcast" 2.0 (at broadcast 0)

let constant_operand () =
  let x = vec [| 1.0; 2.0; 3.0 |] in
  let out =
    over
      (fun _ ->
        P.psum
          (Concrete (Ndarray.of_floats Dtype.F32 [||] [| 2.0 |]))
          ~axis_name:"i")
      x
  in
  close "constant summed over replicas" 6.0 (at out 0)

let gradient_matches_jax () =
  let f a =
    let x = sole a in
    let total = P.psum (Core.bind1 Mul [ x; x ]) ~axis_name:"i" in
    total
  in
  let g =
    sole
      (B.vmap ~axis_name:"i"
         (fun a -> Ad.grad f a)
         [ Some 0 ]
         [ vec [| 1.0; 2.0; 3.0 |] ])
  in
  close "first" 6.0 (at g 0);
  close "third" 18.0 (at g 2)

let unbound_name () =
  match P.psum (vec [| 1.0 |]) ~axis_name:"missing" with
  | _ -> Alcotest.fail "expected an unbound axis error"
  | exception Ojax.Py_exn.Name_error _ -> ()

let debug_print () =
  let seen = ref [] in
  let f a =
    Debugging.callback
      (fun values ->
        seen := Debugging.format_message "x={} done" values :: !seen)
      a;
    a
  in
  let x = Concrete (Ndarray.of_floats Dtype.F32 [||] [| 1.5 |]) in
  ignore (f [ x ]);
  ignore (Api.jit_flat f [ x ]);
  ignore (Ad.jvp f [ x ] [ x ]);
  Alcotest.(check int) "callback ran each time" 3 (List.length !seen);
  Alcotest.(check string) "format" "x=1.5 done" (List.hd !seen)

let () =
  Alcotest.run "parallel"
    [
      ( "collectives",
        [
          Alcotest.test_case "reductions" `Quick reductions;
          Alcotest.test_case "index and size" `Quick index_and_size;
          Alcotest.test_case "gather and permute" `Quick gather_and_permute;
          Alcotest.test_case "constant operand" `Quick constant_operand;
          Alcotest.test_case "gradient" `Quick gradient_matches_jax;
          Alcotest.test_case "unbound name" `Quick unbound_name;
        ] );
      ("debug", [ Alcotest.test_case "callback" `Quick debug_print ]);
    ]
