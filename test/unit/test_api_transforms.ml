open Ojax
open Ojax.Types
module T = Ojax.Tree_util

let () = Lax.install ()
let b1 = Core.bind1
let f32 shape xs = Concrete (Ndarray.of_floats Dtype.F32 shape xs)
let vec xs = f32 [| Array.length xs |] xs
let scalar x = f32 [||] [| x |]

let nd = function
  | T.Leaf (Concrete a) -> a
  | T.Leaf (Device b) -> Pjrt.Buffer.to_host b
  | _ -> Alcotest.fail "expected a concrete leaf"

let check name expected actual =
  Alcotest.(check (float 1e-5)) name expected actual

let leaf = function T.Leaf v -> v | _ -> Alcotest.fail "expected a leaf"

let cubes = function
  | [ x ] ->
      let v = leaf x in
      T.Leaf (b1 Mul [ b1 Mul [ v; v ]; v ])
  | _ -> Alcotest.fail "arity"

let expect_diag name jac =
  let a = nd jac in
  Alcotest.(check (array int)) (name ^ " shape") [| 2; 2 |] (Ndarray.shape a);
  check (name ^ " 00") 3.0 (Ndarray.get_f a [| 0; 0 |]);
  check (name ^ " 01") 0.0 (Ndarray.get_f a [| 0; 1 |]);
  check (name ^ " 11") 12.0 (Ndarray.get_f a [| 1; 1 |])

let jacobians () =
  let x = T.Leaf (vec [| 1.0; 2.0 |]) in
  expect_diag "jacfwd" (Api.jacfwd cubes [ x ]);
  expect_diag "jacrev" (Api.jacrev cubes [ x ])

let hessian_rule () =
  let sum_cubes args = T.Leaf (b1 (Reduce_sum [| 0 |]) [ leaf (cubes args) ]) in
  let h = nd (Api.hessian sum_cubes [ T.Leaf (vec [| 1.0; 2.0 |]) ]) in
  Alcotest.(check (array int)) "shape" [| 2; 2 |] (Ndarray.shape h);
  check "00" 6.0 (Ndarray.get_f h [| 0; 0 |]);
  check "11" 12.0 (Ndarray.get_f h [| 1; 1 |]);
  check "01" 0.0 (Ndarray.get_f h [| 0; 1 |])

let pytree_jacobian () =
  let f = function
    | [ T.Dict kvs ] ->
        let a = leaf (List.assoc "a" kvs) and b = leaf (List.assoc "b" kvs) in
        T.Tuple [ T.Leaf (b1 Mul [ a; b ]); T.Leaf (b1 Add [ a; b ]) ]
    | _ -> Alcotest.fail "arity"
  in
  let x = T.Dict [ ("a", T.Leaf (scalar 2.0)); ("b", T.Leaf (scalar 3.0)) ] in
  match Api.jacrev f [ x ] with
  | T.Tuple [ T.Dict d0; T.Dict d1 ] ->
      check "d(ab)/da" 3.0 (Ndarray.get_f (nd (List.assoc "a" d0)) [||]);
      check "d(ab)/db" 2.0 (Ndarray.get_f (nd (List.assoc "b" d0)) [||]);
      check "d(a+b)/da" 1.0 (Ndarray.get_f (nd (List.assoc "a" d1)) [||])
  | _ -> Alcotest.fail "unexpected jacobian structure"

let argnums_and_aux () =
  let f = function
    | [ a; b ] ->
        let y = b1 Mul [ leaf a; leaf b ] in
        T.Tuple [ T.Leaf y; T.Leaf (scalar 42.0) ]
    | _ -> Alcotest.fail "arity"
  in
  let y, grads, aux =
    Api.value_and_grad_argnums ~has_aux:true ~argnums:[ 1; 0 ] f
      [ T.Leaf (scalar 2.0); T.Leaf (scalar 5.0) ]
  in
  check "value" 10.0 (Ndarray.get_f (nd (T.Leaf y)) [||]);
  check "aux" 42.0 (Ndarray.get_f (nd aux) [||]);
  match grads with
  | [ g1; g0 ] ->
      check "d/db" 2.0 (Ndarray.get_f (nd g1) [||]);
      check "d/da" 5.0 (Ndarray.get_f (nd g0) [||])
  | _ -> Alcotest.fail "gradient arity"

let shapes_only () =
  let f = function
    | [ x ] -> T.Leaf (b1 (Reduce_sum [| 0 |]) [ leaf x ])
    | _ -> Alcotest.fail "arity"
  in
  match Api.eval_shape f [ T.Leaf (f32 [| 3; 2 |] (Array.make 6 1.0)) ] with
  | T.Leaf a -> Alcotest.(check (array int)) "shape" [| 2 |] a.shape
  | _ -> Alcotest.fail "unexpected shape result"

let linear_transpose_rule () =
  let f = function
    | [ x ] -> T.Leaf (b1 Mul [ leaf x; scalar 3.0 ])
    | _ -> Alcotest.fail "arity"
  in
  match
    Api.linear_transpose f [ T.Leaf (scalar 0.0) ] (T.Leaf (scalar 2.0))
  with
  | [ g ] -> check "transpose" 6.0 (Ndarray.get_f (nd g) [||])
  | _ -> Alcotest.fail "arity"

let stop_gradient_rule () =
  let f = function
    | [ x ] ->
        let v = leaf x in
        T.Leaf (b1 Mul [ Custom_derivatives.stop_gradient v; v ])
    | _ -> Alcotest.fail "arity"
  in
  check "grad" 3.0
    (Ndarray.get_f (nd (Api.grad f [ T.Leaf (scalar 3.0) ])) [||])

let closure_convert_rule () =
  let g y =
    let f, hoisted =
      Custom_derivatives.closure_convert
        (fun a -> [ b1 Mul [ List.hd a; y ] ])
        [ { shape = [||]; dtype = Dtype.F32; weak_type = false } ]
    in
    (f, hoisted)
  in
  let concrete, hoisted = g (scalar 3.0) in
  Alcotest.(check int) "no hoisting for constants" 0 (List.length hoisted);
  match concrete [ scalar 2.0 ] with
  | [ v ] -> check "value" 6.0 (Ndarray.get_f (nd (T.Leaf v)) [||])
  | _ -> Alcotest.fail "arity"

let callback_rule () =
  let host = function
    | [ Concrete a ] -> [ scalar (Ndarray.get_f a [||] +. 1.0) ]
    | _ -> Alcotest.fail "arity"
  in
  let scalar_aval = { shape = [||]; dtype = Dtype.F32; weak_type = false } in
  let f args = Api.pure_callback host ~result_avals:[ scalar_aval ] args in
  (match Api.jit_flat f [ scalar 2.0 ] with
  | [ v ] -> check "under jit" 3.0 (Ndarray.get_f (nd (T.Leaf v)) [||])
  | _ -> Alcotest.fail "arity");
  (match Interpreters.Batching.vmap f [ Some 0 ] [ vec [| 1.0; 2.0 |] ] with
  | [ v ] -> check "under vmap" 3.0 (Ndarray.get_f (nd (T.Leaf v)) [| 1 |])
  | _ -> Alcotest.fail "arity");
  match Interpreters.Ad.jvp f [ scalar 2.0 ] [ scalar 1.0 ] with
  | _ -> Alcotest.fail "expected jvp of a callback to raise"
  | exception Invalid_argument _ -> ()

let disabled_jit () =
  let calls = ref 0 in
  let f args =
    incr calls;
    args
  in
  let jitted = Api.jit_flat f in
  ignore (Api.disable_jit (fun () -> jitted [ scalar 1.0 ]));
  ignore (Api.disable_jit (fun () -> jitted [ scalar 1.0 ]));
  Alcotest.(check int) "ran each call" 2 !calls

let pmap_single_device () =
  let f = function
    | [ x ] -> T.Leaf (b1 Mul [ leaf x; scalar 2.0 ])
    | _ -> Alcotest.fail "arity"
  in
  let out = nd (Api.pmap f [ T.Leaf (vec [| 3.0 |]) ]) in
  check "one replica" 6.0 (Ndarray.get_f out [| 0 |]);
  match Api.pmap f [ T.Leaf (vec [| 3.0; 4.0 |]) ] with
  | _ -> Alcotest.fail "expected a replica count error"
  | exception Invalid_argument _ -> ()

let tree_api () =
  let t = T.Dict [ ("b", T.Leaf 2); ("a", T.Tuple [ T.Leaf 1; T.Null ]) ] in
  Alcotest.(check (list int)) "leaves sorted" [ 1; 2 ] (Tree.leaves t);
  let paths = Tree.leaves_with_path t in
  Alcotest.(check (list string))
    "paths" [ "['a'][0]"; "['b']" ]
    (List.map (fun (p, _) -> T.keystr p) paths);
  Alcotest.(check int) "reduce" 3 (Tree.reduce ( + ) 0 t);
  Alcotest.(check bool) "all" true (Tree.all (Tree.map (fun x -> x > 0) t));
  let nested =
    T.Tuple [ T.List [ T.Leaf 1; T.Leaf 2 ]; T.List [ T.Leaf 3; T.Leaf 4 ] ]
  in
  let outer = T.tree_structure (T.Tuple [ T.Leaf (); T.Leaf () ]) in
  let inner = T.tree_structure (T.List [ T.Leaf (); T.Leaf () ]) in
  (match Tree.transpose outer inner nested with
  | T.List [ T.Tuple [ T.Leaf 1; T.Leaf 3 ]; T.Tuple [ T.Leaf 2; T.Leaf 4 ] ] ->
      ()
  | _ -> Alcotest.fail "unexpected transpose");
  let custom = T.Custom ("point", "2d", [ T.Leaf 1; T.Leaf 2 ]) in
  (match Tree.unflatten (Tree.structure custom) [ 5; 6 ] with
  | T.Custom ("point", "2d", [ T.Leaf 5; T.Leaf 6 ]) -> ()
  | _ -> Alcotest.fail "custom node should round trip");
  Alcotest.(check (list int))
    "broadcast" [ 9; 9 ]
    (Tree.leaves (Tree.broadcast (T.Leaf 9) (T.Tuple [ T.Leaf 1; T.Leaf 2 ])))

let () =
  Alcotest.run "api_transforms"
    [
      ( "jacobians",
        [
          Alcotest.test_case "jacfwd and jacrev" `Quick jacobians;
          Alcotest.test_case "hessian" `Quick hessian_rule;
          Alcotest.test_case "pytrees" `Quick pytree_jacobian;
        ] );
      ( "transforms",
        [
          Alcotest.test_case "argnums and aux" `Quick argnums_and_aux;
          Alcotest.test_case "eval_shape" `Quick shapes_only;
          Alcotest.test_case "linear_transpose" `Quick linear_transpose_rule;
          Alcotest.test_case "stop_gradient" `Quick stop_gradient_rule;
          Alcotest.test_case "closure_convert" `Quick closure_convert_rule;
          Alcotest.test_case "pure_callback" `Quick callback_rule;
          Alcotest.test_case "disable_jit" `Quick disabled_jit;
          Alcotest.test_case "pmap" `Quick pmap_single_device;
        ] );
      ("tree", [ Alcotest.test_case "api" `Quick tree_api ]);
    ]
