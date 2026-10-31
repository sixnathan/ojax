open Ojax
open Ojax.Types
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching
module Cd = Custom_derivatives
module T = Tree_util
module C = Checkpoint
module L = Lax_api

let () = Lax.install ()
let scalar x = Concrete (Ndarray.of_floats Dtype.F32 [||] [| x |])
let vec xs = Concrete (Ndarray.of_floats Dtype.F32 [| Array.length xs |] xs)
let mat r c xs = Concrete (Ndarray.of_floats Dtype.F32 [| r; c |] xs)

let to_nd = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected a concrete value"

let value v = Ndarray.get_f (to_nd v) [||]

let check name expected actual =
  Alcotest.(check (float 2e-5)) name expected actual

let widen n x =
  Core.bind1 (Broadcast_in_dim { shape = [| n |]; dims = [||] }) [ x ]

let floats v =
  let nd = to_nd v in
  let n = Array.fold_left ( * ) 1 (Ndarray.shape nd) in
  let out = Array.make n 0.0 in
  ignore
    (Ndarray.fold
       (fun i x ->
         out.(i) <- x;
         i + 1)
       0 nd);
  out

let check_arr name expected v =
  Alcotest.(check (array (float 2e-5))) name expected (floats v)

let add a b = Core.bind1 Add [ a; b ]
let mul a b = Core.bind1 Mul [ a; b ]
let sin x = Core.bind1 Sin [ x ]
let cos x = Core.bind1 Cos [ x ]
let sole = function [ x ] -> x | _ -> Alcotest.fail "expected one value"
let grad_of f x = sole (Ad.grad (fun a -> sole (f a)) [ x ])
let grad_flat f xs = Ad.grad (fun a -> sole (f a)) xs
let leaf v = T.Leaf v
let unleaf = function T.Leaf v -> v | _ -> Alcotest.fail "expected a leaf"
let log : string list ref = ref []
let note s = log := s :: !log
let flag b = if b then "T" else "F"
let is_sym = function Cd.Symbolic_zero _ -> true | Cd.Tangent _ -> false
let kind_of t = if is_sym t then "sym" else "val"

let run_log name expected_log thunk =
  log := [];
  thunk ();
  Alcotest.(check (list string)) (name ^ " log") expected_log (List.rev !log)

let raises_with name fragment thunk =
  match thunk () with
  | exception (Failure m | Invalid_argument m | Errors.Jax_type_error m) ->
      let contains =
        let n = String.length fragment and h = String.length m in
        let rec go i =
          i + n <= h && (String.sub m i n = fragment || go (i + 1))
        in
        go 0
      in
      if not contains then Alcotest.failf "%s: unexpected message %s" name m
  | _ -> Alcotest.failf "%s: expected an exception" name

let pair_primal a =
  match a with
  | [ x; y ] -> [ mul x y; add x y ]
  | _ -> Alcotest.fail "expected two arguments"

let pair_forward = ref pair_primal

let symbolic_pair =
  Cd.custom_vjp_symbolic pair_primal
    ~fwd:(fun args ->
      match args with
      | [ (x : Cd.custom_vjp_primal); y ] ->
          note
            (Printf.sprintf "fwd %s %s" (flag x.perturbed) (flag y.perturbed));
          (!pair_forward [ x.value; y.value ], [ x.value; y.value ])
      | _ -> Alcotest.fail "fwd arity")
    ~bwd:(fun res cts ->
      match (res, cts) with
      | [ x; y ], [ a; b ] ->
          note (Printf.sprintf "bwd %s %s" (kind_of a) (kind_of b));
          let a = Cd.instantiate_tangent a and b = Cd.instantiate_tangent b in
          [
            Some (Cd.Tangent (add (mul a y) b));
            Some (Cd.Tangent (add (mul a x) b));
          ]
      | _ -> Alcotest.fail "bwd arity")

let () = pair_forward := symbolic_pair
let first a = [ List.nth (symbolic_pair a) 0 ]
let second a = [ List.nth (symbolic_pair a) 1 ]

let vjp_flags () =
  let two = scalar 2.0 and three = scalar 3.0 in
  run_log "wrt x" [ "fwd T F"; "bwd val sym" ] (fun () ->
      check "grad" 3.0 (value (grad_of (fun a -> first [ sole a; three ]) two)));
  run_log "wrt y" [ "fwd F T"; "bwd val sym" ] (fun () ->
      check "grad" 2.0 (value (grad_of (fun a -> first [ two; sole a ]) three)));
  run_log "wrt both" [ "fwd T T"; "bwd val sym" ] (fun () ->
      match grad_flat first [ two; three ] with
      | [ a; b ] ->
          check "dx" 3.0 (value a);
          check "dy" 2.0 (value b)
      | _ -> Alcotest.fail "arity");
  run_log "second output" [ "fwd T F"; "bwd sym val" ] (fun () ->
      check "grad" 1.0 (value (grad_of (fun a -> second [ sole a; three ]) two)));
  run_log "both outputs" [ "fwd T F"; "bwd val val" ] (fun () ->
      let f a =
        match symbolic_pair [ sole a; three ] with
        | [ p; q ] -> [ add p q ]
        | _ -> Alcotest.fail "arity"
      in
      check "grad" 4.0 (value (grad_of f two)));
  run_log "computed constant" [ "fwd T F"; "bwd val sym" ] (fun () ->
      check "grad" 0.84147096
        (value (grad_of (fun a -> first [ sole a; sin (scalar 1.0) ]) two)))

let vjp_zero_derivative_inputs () =
  run_log "stop_gradient" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let f a = first [ sole a; Cd.stop_gradient (sole a) ] in
      check "grad" 2.0 (value (grad_of f (scalar 2.0))));
  run_log "floor" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let f a = first [ sole a; Core.bind1 Floor [ sole a ] ] in
      check "grad" 2.0 (value (grad_of f (scalar 2.5))));
  run_log "comparison" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let f a =
        let pred = Core.bind1 Gt [ sole a; scalar 0.0 ] in
        first [ sole a; Core.bind1 Select_n [ pred; scalar 2.0; scalar 1.0 ] ]
      in
      check "grad" 1.0 (value (grad_of f (scalar 2.5))))

let vjp_unused () =
  run_log "constant inputs" [] (fun () ->
      let f a = [ mul (sole a) (sole (first [ scalar 1.0; scalar 2.0 ])) ] in
      check "grad" 2.0 (value (grad_of f (scalar 2.0))));
  run_log "outputs unused" [ "fwd T F" ] (fun () ->
      let f a =
        ignore (symbolic_pair [ sole a; scalar 2.0 ]);
        a
      in
      check "grad" 1.0 (value (grad_of f (scalar 2.0))));
  run_log "no derivative" [] (fun () ->
      match symbolic_pair [ scalar 1.0; scalar 2.0 ] with
      | [ a; b ] ->
          check "product" 2.0 (value a);
          check "sum" 3.0 (value b)
      | _ -> Alcotest.fail "arity")

let vjp_transforms () =
  let three = scalar 3.0 in
  let wrt_x a = first [ sole a; three ] in
  run_log "jit of grad" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let g = Api.jit_flat (fun a -> [ grad_of wrt_x (sole a) ]) in
      check "grad" 3.0 (value (sole (g [ scalar 2.0 ]))));
  run_log "grad of jit" [ "fwd T F"; "bwd val sym" ] (fun () ->
      check "grad" 3.0 (value (grad_of (Api.jit_flat wrt_x) (scalar 2.0))));
  run_log "jit of grad both" [ "fwd T T"; "bwd val sym" ] (fun () ->
      match Api.jit_flat (grad_flat first) [ scalar 2.0; three ] with
      | [ a; b ] ->
          check "dx" 3.0 (value a);
          check "dy" 2.0 (value b)
      | _ -> Alcotest.fail "arity");
  run_log "vmap of grad" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let g =
        Batching.vmap
          (fun a -> [ grad_of wrt_x (sole a) ])
          [ Some 0 ]
          [ vec [| 2.0; 5.0 |] ]
      in
      check_arr "grad" [| 3.0; 3.0 |] (sole g));
  run_log "grad of vmap" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let total a =
        let ys = Batching.vmap wrt_x [ Some 0 ] a in
        [ Core.bind1 (Reduce_sum [| 0 |]) ys ]
      in
      check_arr "grad" [| 3.0; 3.0 |] (grad_of total (vec [| 2.0; 5.0 |])));
  run_log "grad of vmap, second output" [ "fwd T F"; "bwd sym val" ] (fun () ->
      let total a =
        let ys =
          Batching.vmap (fun x -> second [ sole x; three ]) [ Some 0 ] a
        in
        [ Core.bind1 (Reduce_sum [| 0 |]) ys ]
      in
      check_arr "grad" [| 1.0; 1.0 |] (grad_of total (vec [| 2.0; 5.0 |])))

let vjp_control_flow () =
  run_log "scan with a closed constant" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let f a =
        let step c = [ sole (first [ sole c; scalar 3.0 ]) ] in
        L.scan ~length:2 step a []
      in
      check "grad" 9.0 (value (grad_of f (scalar 2.0))));
  run_log "scan over constant xs" [ "fwd T F"; "bwd val sym" ] (fun () ->
      let f a =
        let step args =
          match args with
          | [ c; z ] -> [ sole (first [ c; z ]) ]
          | _ -> Alcotest.fail "arity"
        in
        L.scan step a [ vec [| 3.0; 4.0 |] ]
      in
      check "grad" 12.0 (value (grad_of f (scalar 2.0))));
  log := [];
  let f a =
    let x = sole a in
    let pred = Core.bind1 Gt [ x; scalar 0.0 ] in
    L.cond pred
      (fun ops -> first [ sole ops; scalar 3.0 ])
      (fun ops -> ops)
      [ x ]
  in
  check "cond grad" 3.0 (value (grad_of f (scalar 2.0)));
  Alcotest.(check bool)
    "cond fwd flags" true
    (List.mem "fwd T F" !log && not (List.mem "fwd T T" !log));
  Alcotest.(check bool)
    "cond bwd flags" true
    (List.mem "bwd val sym" !log && not (List.mem "bwd val val" !log))

let vjp_nested () =
  run_log "second derivative" [ "fwd T F"; "fwd T F"; "bwd val sym" ] (fun () ->
      let once a =
        [ grad_of (fun b -> first [ sole b; scalar 3.0 ]) (sole a) ]
      in
      check "hessian" 0.0 (value (grad_of once (scalar 2.0))))

let vjp_forward_raises () =
  raises_with "jvp" "can't apply forward-mode autodiff" (fun () ->
      Ad.jvp first [ scalar 2.0; scalar 3.0 ] [ scalar 1.0; scalar 1.0 ])

let double a = [ mul (sole a) (scalar 2.0) ]

let bwd_outputs () =
  let with_bwd bwd =
    Cd.custom_vjp_symbolic double
      ~fwd:(fun args -> (double [ (List.hd args).Cd.value ], []))
      ~bwd
  in
  let plus_x g a = [ add (sole (g a)) (sole a) ] in
  let symbolic =
    with_bwd (fun _ cts ->
        let aval = Core.get_aval (Cd.instantiate_tangent (List.hd cts)) in
        [ Some (Cd.Symbolic_zero { aval }) ])
  in
  check "symbolic" 1.0 (value (grad_of (plus_x symbolic) (scalar 2.0)));
  let none = with_bwd (fun _ _ -> [ None ]) in
  check "none" 1.0 (value (grad_of (plus_x none) (scalar 2.0)));
  let bad =
    with_bwd (fun _ _ ->
        let aval = Shape_dtype_struct.make [| 2 |] Dtype.F32 in
        [ Some (Cd.Symbolic_zero { aval }) ])
  in
  raises_with "bad symbolic" "had shape/dtype float32[2]" (fun () ->
      grad_of (plus_x bad) (scalar 2.0));
  let wrong =
    with_bwd (fun _ _ -> [ Some (Cd.Tangent (vec [| 1.0; 2.0 |])) ])
  in
  raises_with "bad value" "must produce an output with the same type" (fun () ->
      grad_of (plus_x wrong) (scalar 2.0))

let primal_trees () =
  let dict a b = T.Dict [ ("a", a); ("b", T.Tuple [ b ]) ] in
  let def = T.tree_structure (dict (leaf (scalar 0.0)) (leaf (scalar 0.0))) in
  let of_tree tree =
    match T.tree_leaves tree with
    | [ a; b ] -> (a, b)
    | _ -> Alcotest.fail "two leaves"
  in
  let primal args =
    match args with [ a; b ] -> [ mul a b ] | _ -> Alcotest.fail "two leaves"
  in
  let f =
    Cd.custom_vjp_symbolic primal
      ~fwd:(fun args ->
        let tree = T.tree_unflatten def args in
        let a, b = of_tree (Cd.custom_vjp_primal_tree_values tree) in
        let pa, pb = of_tree tree in
        note
          (Printf.sprintf "fwd a=%s %.1f b=%s %.1f" (flag pa.Cd.perturbed)
             (value a) (flag pb.Cd.perturbed) (value b));
        (primal [ a; b ], [ a; b ]))
      ~bwd:(fun res cts ->
        match (res, cts) with
        | [ a; b ], [ c ] ->
            let c = Cd.instantiate_tangent c in
            [ Some (Cd.Tangent (mul c b)); Some (Cd.Tangent (mul c a)) ]
        | _ -> Alcotest.fail "arity")
  in
  run_log "tree" [ "fwd a=T 2.0 b=F 3.0" ] (fun () ->
      check "grad" 3.0
        (value (grad_of (fun a -> f [ sole a; scalar 3.0 ]) (scalar 2.0))))

let symbolic_constructors () =
  let z = Cd.symbolic_zero_from_primal_value (mat 2 3 (Array.make 6 1.0)) in
  Alcotest.(check (array int)) "shape" [| 2; 3 |] z.aval.shape;
  Alcotest.(check bool) "dtype" true (z.aval.dtype = Dtype.F32);
  check_arr "instantiate" (Array.make 6 0.0)
    (Cd.instantiate_tangent (Cd.Symbolic_zero z))

let scale_primal a =
  match a with [ x; y ] -> [ mul x y ] | _ -> Alcotest.fail "two arguments"

let symbolic_scale =
  Cd.custom_jvp_symbolic scale_primal ~jvp:(fun ps ts ->
      match (ps, ts) with
      | [ x; y ], [ a; b ] ->
          note (Printf.sprintf "jvp %s %s" (kind_of a) (kind_of b));
          let out = mul x y in
          let term t k other =
            mul (mul (Cd.instantiate_tangent t) other) (scalar k)
          in
          let tangent =
            match (a, b) with
            | Cd.Symbolic_zero _, Cd.Symbolic_zero _ ->
                Cd.Symbolic_zero (Cd.symbolic_zero_from_primal_value out)
            | Cd.Symbolic_zero _, _ -> Cd.Tangent (term b 100.0 x)
            | _, Cd.Symbolic_zero _ -> Cd.Tangent (term a 10.0 y)
            | _ -> Cd.Tangent (add (term a 10.0 y) (term b 100.0 x))
          in
          ([ out ], [ tangent ])
      | _ -> Alcotest.fail "arity")

let jvp_of f ps ts =
  let po, to_ = Ad.jvp f ps ts in
  (value (sole po), value (sole to_))

let custom_jvp_forward () =
  let two = scalar 2.0 and three = scalar 3.0 and one = scalar 1.0 in
  let check_pair name (ep, et) (p, t) =
    check (name ^ " primal") ep p;
    check (name ^ " tangent") et t
  in
  run_log "x" [ "jvp val sym" ] (fun () ->
      check_pair "x" (6.0, 30.0)
        (jvp_of (fun a -> symbolic_scale [ sole a; three ]) [ two ] [ one ]));
  run_log "y" [ "jvp sym val" ] (fun () ->
      check_pair "y" (6.0, 200.0)
        (jvp_of (fun a -> symbolic_scale [ two; sole a ]) [ three ] [ one ]));
  run_log "both" [ "jvp val val" ] (fun () ->
      check_pair "both" (6.0, 230.0)
        (jvp_of symbolic_scale [ two; three ] [ one; one ]));
  run_log "explicit zero stays a value" [ "jvp val val" ] (fun () ->
      check_pair "zero" (6.0, 30.0)
        (jvp_of symbolic_scale [ two; three ] [ one; scalar 0.0 ]));
  run_log "jit" [ "jvp val sym" ] (fun () ->
      check_pair "jit" (6.0, 30.0)
        (jvp_of
           (Api.jit_flat (fun a -> symbolic_scale [ sole a; three ]))
           [ two ] [ one ]))

let custom_jvp_reverse () =
  let two = scalar 2.0 and three = scalar 3.0 in
  run_log "grad x" [ "jvp val sym" ] (fun () ->
      check "grad" 30.0
        (value (grad_of (fun a -> symbolic_scale [ sole a; three ]) two)));
  run_log "grad y" [ "jvp sym val" ] (fun () ->
      check "grad" 200.0
        (value (grad_of (fun a -> symbolic_scale [ two; sole a ]) three)));
  run_log "grad both" [ "jvp val val" ] (fun () ->
      match grad_flat symbolic_scale [ two; three ] with
      | [ a; b ] ->
          check "dx" 30.0 (value a);
          check "dy" 200.0 (value b)
      | _ -> Alcotest.fail "arity");
  run_log "constant" [] (fun () ->
      let f a = [ mul (sole a) (sole (symbolic_scale [ two; three ])) ] in
      check "grad" 6.0 (value (grad_of f two)));
  run_log "jit of grad" [ "jvp val sym" ] (fun () ->
      let g =
        Api.jit_flat (fun a ->
            [ grad_of (fun b -> symbolic_scale [ sole b; three ]) (sole a) ])
      in
      check "grad" 30.0 (value (sole (g [ two ]))));
  run_log "grad of jit" [ "jvp val sym" ] (fun () ->
      let g = Api.jit_flat (fun a -> symbolic_scale [ sole a; three ]) in
      check "grad" 30.0 (value (grad_of g two)))

let custom_jvp_batched () =
  let three = scalar 3.0 in
  let xs = vec [| 2.0; 5.0 |] in
  run_log "vmap of grad" [ "jvp val sym" ] (fun () ->
      let g =
        Batching.vmap
          (fun a ->
            [ grad_of (fun b -> symbolic_scale [ sole b; three ]) (sole a) ])
          [ Some 0 ] [ xs ]
      in
      check_arr "grad" [| 30.0; 30.0 |] (sole g));
  let batched a =
    Batching.vmap (fun b -> symbolic_scale [ sole b; three ]) [ Some 0 ] a
  in
  run_log "grad of vmap" [ "jvp val sym" ] (fun () ->
      let total a = [ Core.bind1 (Reduce_sum [| 0 |]) (batched a) ] in
      check_arr "grad" [| 30.0; 30.0 |] (grad_of total xs));
  run_log "jvp of vmap" [ "jvp val sym" ] (fun () ->
      let po, to_ = Ad.jvp batched [ xs ] [ vec [| 1.0; 1.0 |] ] in
      check_arr "primal" [| 6.0; 15.0 |] (sole po);
      check_arr "tangent" [| 30.0; 30.0 |] (sole to_))

let custom_jvp_control_flow () =
  let three = scalar 3.0 in
  run_log "scan" [ "jvp val sym" ] (fun () ->
      let f a =
        L.scan ~length:2 (fun c -> symbolic_scale [ sole c; three ]) a []
      in
      check "grad" 900.0 (value (grad_of f (scalar 2.0))));
  run_log "scan over constant xs" [ "jvp val sym" ] (fun () ->
      let f a =
        L.scan
          (fun args ->
            match args with
            | [ c; z ] -> symbolic_scale [ c; z ]
            | _ -> Alcotest.fail "arity")
          a
          [ vec [| 3.0; 4.0 |] ]
      in
      let po, to_ = Ad.jvp f [ scalar 2.0 ] [ scalar 1.0 ] in
      check "primal" 24.0 (value (sole po));
      check "tangent" 1200.0 (value (sole to_)));
  run_log "while" [ "jvp val sym" ] (fun () ->
      let f a =
        match
          L.while_loop
            (fun s -> Core.bind1 Lt [ List.hd s; scalar 2.0 ])
            (fun s ->
              match s with
              | [ i; c ] ->
                  [ add i (scalar 1.0); sole (symbolic_scale [ c; three ]) ]
              | _ -> Alcotest.fail "arity")
            [ scalar 0.0; sole a ]
        with
        | [ _; c ] -> [ c ]
        | _ -> Alcotest.fail "arity"
      in
      let po, to_ = Ad.jvp f [ scalar 2.0 ] [ scalar 1.0 ] in
      check "primal" 18.0 (value (sole po));
      check "tangent" 900.0 (value (sole to_)));
  run_log "cond" [ "jvp val sym" ] (fun () ->
      let f a =
        let x = sole a in
        L.cond
          (Core.bind1 Gt [ x; scalar 0.0 ])
          (fun ops -> symbolic_scale [ sole ops; three ])
          (fun ops -> ops)
          [ x ]
      in
      check "grad" 30.0 (value (grad_of f (scalar 2.0))))

let custom_jvp_outputs () =
  let square a = [ mul (sole a) (sole a) ] in
  let bad =
    Cd.custom_jvp_symbolic square ~jvp:(fun ps _ ->
        let aval = Shape_dtype_struct.make [| 2 |] Dtype.F32 in
        (square ps, [ Cd.Symbolic_zero { aval } ]))
  in
  raises_with "bad" "Custom JVP rule must produce primal and tangent" (fun () ->
      Ad.jvp bad [ scalar 2.0 ] [ scalar 1.0 ]);
  let zero =
    Cd.custom_jvp_symbolic square ~jvp:(fun ps _ ->
        ( square ps,
          [ Cd.Symbolic_zero (Cd.symbolic_zero_from_primal_value (sole ps)) ] ))
  in
  let p, t = jvp_of zero [ scalar 2.0 ] [ scalar 1.0 ] in
  check "primal" 4.0 p;
  check "tangent" 0.0 t;
  let plus_x a = [ add (sole (zero a)) (sole a) ] in
  check "grad" 1.0 (value (grad_of plus_x (scalar 2.0)))

let executed_during thunk =
  let counts = Hashtbl.create 8 in
  let original = Core.rules.impl in
  Core.rules.impl <-
    (fun prim inputs ->
      let name = Pretty_printer.primitive_name prim in
      let n = Option.value (Hashtbl.find_opt counts name) ~default:0 in
      Hashtbl.replace counts name (n + 1);
      original prim inputs);
  Fun.protect
    ~finally:(fun () -> Core.rules.impl <- original)
    (fun () -> thunk ());
  fun name -> Option.value (Hashtbl.find_opt counts name) ~default:0

let backward_counts f x =
  let _, pull = Ad.vjp f [ x ] in
  let seed = Concrete (Ndarray.of_floats Dtype.F32 [||] [| 1.0 |]) in
  let seen = executed_during (fun () -> ignore (pull [ seed ])) in
  (seen "sin", seen "cos", seen "mul", seen "dot_general")

let check_counts name (s, c, m, d) f x =
  let s', c', m', d' = backward_counts f x in
  Alcotest.(check (list int))
    (name ^ " backward primitive counts")
    [ s; c; m; d ] [ s'; c'; m'; d' ]

let check_residuals name expected f args =
  let got =
    List.map
      (fun (a, source) -> (Ref_state.aval_string a, source))
      (C.saved_residuals f args)
  in
  Alcotest.(check (list (pair string string)))
    (name ^ " residuals")
    (List.sort compare expected)
    (List.sort compare got)

let two_sines a =
  let y = sin (sole a) in
  [ sin y ]

let arg0 = "from the argument at flattened index 0"

let policy_sines () =
  let cases =
    [
      ("nothing", C.nothing_saveable, [ ("float32[]", arg0) ], (1, 2, 2, 0));
      ( "everything",
        C.everything_saveable,
        [ ("float32[]", "output of cos"); ("float32[]", "output of cos") ],
        (0, 0, 2, 0) );
      ( "sin only",
        (fun prim _ -> prim = Sin),
        [ ("float32[]", arg0); ("float32[]", "output of sin") ],
        (0, 2, 2, 0) );
      ( "cos only",
        (fun prim _ -> prim = Cos),
        [ ("float32[]", "output of cos"); ("float32[]", "output of cos") ],
        (0, 0, 2, 0) );
    ]
  in
  List.iter
    (fun (name, policy, residuals, counts) ->
      let g = C.checkpoint ~policy two_sines in
      check_residuals name residuals g [ scalar 2.0 ];
      check_counts name counts g (scalar 2.0);
      check (name ^ " grad") (-0.25563914) (value (grad_of g (scalar 2.0)));
      let po, _ = Ad.vjp g [ scalar 2.0 ] in
      check (name ^ " value") 0.78907233 (value (sole po)))
    cases

let default_policy () =
  let g = Api.checkpoint two_sines in
  check_residuals "default" [ ("float32[]", arg0) ] g [ scalar 2.0 ];
  check "grad" (-0.25563914) (value (grad_of g (scalar 2.0)))

let dot_dims lhs rhs =
  Dot_general
    {
      lhs_contract = [| lhs |];
      rhs_contract = [| rhs |];
      lhs_batch = [||];
      rhs_batch = [||];
    }

let dots_fn a =
  let matrix = mat 2 3 [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = Core.bind1 (dot_dims 1 0) [ matrix; sole a ] in
  let z = sin y in
  [ Core.bind1 (dot_dims 0 0) [ z; z ] ]

let policy_dots () =
  let constant = ("float32[2,3]", "from a constant") in
  let x = vec [| 1.; 2.; 3. |] in
  let expected_grad = [| 3.9510102; 5.141942; 6.3328743 |] in
  let cases =
    [
      ( "nothing",
        C.nothing_saveable,
        [ constant; ("float32[3]", arg0) ],
        (1, 1, 1, 4) );
      ( "dots",
        C.dots_saveable,
        [ constant; ("float32[2]", "output of dot_general") ],
        (1, 1, 1, 3) );
      ( "checkpoint_dots",
        C.checkpoint_dots,
        [ constant; ("float32[2]", "output of dot_general") ],
        (1, 1, 1, 3) );
      ( "dots no batch",
        C.dots_with_no_batch_dims_saveable,
        [ constant; ("float32[2]", "output of dot_general") ],
        (1, 1, 1, 3) );
      ( "checkpoint dots no batch",
        C.checkpoint_dots_with_no_batch_dims,
        [ constant; ("float32[2]", "output of dot_general") ],
        (1, 1, 1, 3) );
      ( "everything",
        C.everything_saveable,
        [
          constant;
          ("float32[2]", "output of sin");
          ("float32[2]", "output of cos");
        ],
        (0, 0, 1, 3) );
    ]
  in
  List.iter
    (fun (name, policy, residuals, counts) ->
      let g = C.checkpoint ~policy dots_fn in
      check_residuals name residuals g [ x ];
      check_counts name counts g x;
      check_arr (name ^ " grad") expected_grad (grad_of g x);
      let po, _ = Ad.vjp g [ x ] in
      check (name ^ " value") 1.2853744 (value (sole po)))
    cases

let policy_arguments () =
  let seen = ref [] in
  let spy prim avals =
    seen :=
      ( Pretty_printer.primitive_name prim,
        List.map (fun (a : aval) -> a.shape) avals )
      :: !seen;
    false
  in
  let g = C.checkpoint ~policy:spy dots_fn in
  check_arr "grad"
    [| 3.9510102; 5.141942; 6.3328743 |]
    (grad_of g (vec [| 1.; 2.; 3. |]));
  Alcotest.(check (list (pair string (list (array int)))))
    "policy sees primitive and input shapes"
    (List.sort compare
       [
         ("dot_general", [ [| 2; 3 |]; [| 3 |] ]);
         ("sin", [ [| 2 |] ]);
         ("cos", [ [| 2 |] ]);
         ("dot_general", [ [| 2 |]; [| 2 |] ]);
       ])
    (List.sort compare !seen)

let batched_dot a =
  let x = sole a in
  let y =
    Core.bind1
      (Dot_general
         {
           lhs_contract = [| 1 |];
           rhs_contract = [| 1 |];
           lhs_batch = [| 0 |];
           rhs_batch = [| 0 |];
         })
      [ x; x ]
  in
  [ Core.bind1 (Reduce_sum [| 0 |]) [ sin y ] ]

let policy_batch_dims () =
  let x = mat 2 2 [| 1.; 2.; 3.; 4. |] in
  let expected = [| 0.5673244; 1.1346488; 5.947217; 7.9296227 |] in
  let dots = C.checkpoint ~policy:C.dots_saveable batched_dot in
  check_residuals "dots"
    [ ("float32[2,2]", arg0); ("float32[2]", "output of dot_general") ]
    dots [ x ];
  check_arr "dots grad" expected (grad_of dots x);
  let no_batch =
    C.checkpoint ~policy:C.dots_with_no_batch_dims_saveable batched_dot
  in
  check_residuals "no batch" [ ("float32[2,2]", arg0) ] no_batch [ x ];
  check_arr "no batch grad" expected (grad_of no_batch x)

let named_fn a =
  let y = C.checkpoint_name (sin (sole a)) "a" in
  let z = C.checkpoint_name (sin y) "b" in
  [ mul (cos z) y ]

let policy_names () =
  let named n = ("float32[]", Printf.sprintf "named '%s'" n) in
  let argument = ("float32[]", arg0) in
  let cases =
    [
      ( "only a",
        C.save_only_these_names [ "a" ],
        [ argument; named "a" ],
        (2, 3) );
      ( "only b",
        C.save_only_these_names [ "b" ],
        [ argument; named "b" ],
        (2, 3) );
      ( "only a b",
        C.save_only_these_names [ "a"; "b" ],
        [ argument; named "a"; named "b" ],
        (1, 3) );
      ( "any but a",
        C.save_any_names_but_these [ "a" ],
        [ argument; named "b" ],
        (2, 3) );
      ( "any but none",
        C.save_any_names_but_these [],
        [ argument; named "a"; named "b" ],
        (1, 3) );
      ( "both, one named",
        C.save_from_both_policies
          (C.save_only_these_names [ "a" ])
          C.dots_saveable,
        [ argument; named "a" ],
        (2, 3) );
      ( "both, neither",
        C.save_from_both_policies C.nothing_saveable C.nothing_saveable,
        [ argument ],
        (3, 3) );
    ]
  in
  List.iter
    (fun (name, policy, residuals, (s, c)) ->
      let g = C.checkpoint ~policy named_fn in
      check_residuals name residuals g [ scalar 2.0 ];
      let s', c', m', _ = backward_counts g (scalar 2.0) in
      Alcotest.(check (list int)) (name ^ " counts") [ s; c; 5 ] [ s'; c'; m' ];
      check (name ^ " grad") (-0.12820591) (value (grad_of g (scalar 2.0))))
    cases;
  List.iter
    (fun (name, policy) ->
      let g = C.checkpoint ~policy named_fn in
      let s', c', m', _ = backward_counts g (scalar 2.0) in
      Alcotest.(check (list int)) (name ^ " counts") [ 0; 0; 5 ] [ s'; c'; m' ];
      check (name ^ " grad") (-0.12820591) (value (grad_of g (scalar 2.0))))
    [
      ("anything except", C.save_anything_except_these_names [ "a" ]);
      ( "both, one everything",
        C.save_from_both_policies C.nothing_saveable C.everything_saveable );
    ]

let policy_transforms () =
  let g = C.checkpoint ~policy:(C.save_only_these_names [ "a" ]) named_fn in
  let x = scalar 2.0 in
  let po, to_ = Ad.jvp g [ x ] [ scalar 1.0 ] in
  check "jvp primal" 0.6406036 (value (sole po));
  check "jvp tangent" (-0.1282059) (value (sole to_));
  let once a = [ grad_of g (sole a) ] in
  check "grad of grad" (-0.38481575) (value (grad_of once x));
  let forward a = snd (Ad.jvp once a [ scalar 1.0 ]) in
  check "jvp of grad" (-0.38481575) (value (sole (forward [ x ])));
  let xs = vec [| 1.0; 2.0 |] in
  check_arr "vmap of grad"
    [| 0.1914018; -0.12820591 |]
    (sole (Batching.vmap once [ Some 0 ] [ xs ]));
  let total a =
    [ Core.bind1 (Reduce_sum [| 0 |]) (Batching.vmap g [ Some 0 ] a) ]
  in
  check_arr "grad of vmap" [| 0.1914018; -0.12820591 |] (grad_of total xs);
  check "jit of grad" (-0.12820591) (value (sole (Api.jit_flat once [ x ])));
  check "grad of jit" (-0.12820591) (value (grad_of (Api.jit_flat g) x))

let policy_closures () =
  let outer a =
    let c = sole a in
    let inner args = [ mul (sin (sole args)) c ] in
    [
      add
        (sole (C.checkpoint ~policy:C.everything_saveable inner [ scalar 2.0 ]))
        c;
    ]
  in
  check "closed over, constant input" 1.9092975
    (value (grad_of outer (scalar 1.0)));
  let outer2 a =
    let c = sole a in
    let inner args = [ mul (sin (sole args)) c ] in
    C.checkpoint ~policy:(fun prim _ -> prim = Sin) inner [ c ]
  in
  check "closed over, also the input" 1.3817732
    (value (grad_of outer2 (scalar 1.0)));
  let outer3 a =
    let c = sole a in
    let inner args = [ mul (sin (sole args)) c ] in
    Api.checkpoint inner [ c ]
  in
  check "default policy, closed over" 1.3817732
    (value (grad_of outer3 (scalar 1.0)))

let policy_scan () =
  let step c =
    let x = sole c in
    [ mul (sin x) x ]
  in
  List.iter
    (fun (name, policy) ->
      let ck = C.checkpoint ~policy step in
      let f a = L.scan ~length:3 ck a [] in
      let po, pull = Ad.vjp f [ scalar 1.5 ] in
      check (name ^ " value") 1.4874666 (value (sole po));
      check (name ^ " grad") 1.363286 (value (sole (pull [ scalar 1.0 ])));
      let p, t = jvp_of f [ scalar 1.5 ] [ scalar 1.0 ] in
      check (name ^ " jvp") 1.4874666 p;
      check (name ^ " jvp tangent") 1.363286 t;
      check (name ^ " jit of grad") 1.363286
        (value
           (sole
              (Api.jit_flat (fun a -> [ grad_of f (sole a) ]) [ scalar 1.5 ]))))
    [ ("every", C.everything_saveable); ("nothing", C.nothing_saveable) ]

let policy_forward_under_jit () =
  let ck =
    C.checkpoint ~policy:C.everything_saveable (fun a ->
        [ mul (sin (sole a)) (sole a) ])
  in
  match
    Api.jit_flat
      (fun a ->
        let po, to_ = Ad.jvp ck a [ scalar 1.0 ] in
        po @ to_)
      [ scalar 2.0 ]
  with
  | [ p; t ] ->
      check "primal" 1.8185948 (value p);
      check "tangent" 0.07700372 (value t)
  | _ -> Alcotest.fail "two outputs"

let policy_constant_output () =
  let ck =
    C.checkpoint ~policy:C.everything_saveable (fun a ->
        [ sin (sole a); scalar 3.0 ])
  in
  let po, pull = Ad.vjp ck [ scalar 2.0 ] in
  (match po with
  | [ a; b ] ->
      check "first" 0.9092974 (value a);
      check "second" 3.0 (value b)
  | _ -> Alcotest.fail "two outputs");
  check "vjp" (-0.41614684) (value (sole (pull [ scalar 1.0; scalar 1.0 ])));
  match Ad.jvp ck [ scalar 2.0 ] [ scalar 1.0 ] with
  | [ _; _ ], [ a; b ] ->
      check "jvp first" (-0.41614684) (value a);
      check "jvp second" 0.0 (value b)
  | _ -> Alcotest.fail "two tangents"

let policy_partial_arguments () =
  let ck =
    C.checkpoint ~policy:C.everything_saveable (fun args ->
        match args with
        | [ x; y ] -> [ mul (sin x) (cos y) ]
        | _ -> Alcotest.fail "arity")
  in
  let x = scalar 2.0 and y = scalar 3.0 in
  check "wrt x" 0.41198227 (value (grad_of (fun a -> ck [ sole a; y ]) x));
  check "wrt y" (-0.12832005) (value (grad_of (fun a -> ck [ x; sole a ]) y));
  (match grad_flat ck [ x; y ] with
  | [ a; b ] ->
      check "both x" 0.41198227 (value a);
      check "both y" (-0.12832005) (value b)
  | _ -> Alcotest.fail "two grads");
  let cos_out = ("float32[]", "output of cos")
  and sin_out = ("float32[]", "output of sin") in
  check_residuals "both perturbed"
    [ sin_out; cos_out; cos_out; sin_out ]
    ck [ x; y ];
  check_residuals "one perturbed" [ cos_out; cos_out ]
    (fun a -> ck [ sole a; y ])
    [ x ]

let policy_with_custom_vjp () =
  let square =
    Cd.custom_vjp
      (fun a -> [ mul (sole a) (sole a) ])
      ~fwd:(fun a -> ([ mul (sole a) (sole a) ], a))
      ~bwd:(fun res cts -> [ mul (mul (scalar 10.0) (sole res)) (sole cts) ])
  in
  List.iter
    (fun (name, policy) ->
      let ck = C.checkpoint ~policy (fun a -> [ sin (sole (square a)) ]) in
      let po, pull = Ad.vjp ck [ scalar 2.0 ] in
      check (name ^ " value") (-0.7568025) (value (sole po));
      check (name ^ " grad") (-13.072872) (value (sole (pull [ scalar 1.0 ]))))
    [ ("every", C.everything_saveable); ("nothing", C.nothing_saveable) ]

let policy_nested () =
  let inner =
    C.checkpoint ~policy:C.everything_saveable (fun a -> [ sin (sole a) ])
  in
  let outer =
    C.checkpoint ~policy:C.nothing_saveable (fun a ->
        [ mul (sole (inner a)) (sole a) ])
  in
  let po, pull = Ad.vjp outer [ scalar 2.0 ] in
  check "value" 1.8185948 (value (sole po));
  check "grad" 0.07700372 (value (sole (pull [ scalar 1.0 ])));
  check_residuals "residuals" [ ("float32[]", arg0) ] outer [ scalar 2.0 ];
  let once a = [ grad_of outer (sole a) ] in
  check "second derivative" (-2.6508884) (value (grad_of once (scalar 2.0)))

let has_aux_is_primal () =
  match
    Api.value_and_grad_argnums ~has_aux:true ~argnums:[ 0 ]
      (fun args ->
        match args with
        | [ T.Leaf x ] -> T.Tuple [ leaf (mul x x); leaf (add x (scalar 1.0)) ]
        | _ -> Alcotest.fail "one arg")
      [ leaf (scalar 3.0) ]
  with
  | y, [ T.Leaf g ], T.Leaf aux ->
      check "value" 9.0 (value y);
      check "grad" 6.0 (value g);
      check "aux" 4.0 (value aux)
  | _ -> Alcotest.fail "shape"

let names_transform () =
  check "eager" 3.0 (value (C.checkpoint_name (scalar 3.0) "x"));
  check_arr "vmap" [| 0.0; 2.0 |]
    (sole
       (Batching.vmap
          (fun a -> [ mul (C.checkpoint_name (sole a) "n") (scalar 2.0) ])
          [ Some 0 ]
          [ vec [| 0.0; 1.0 |] ]));
  check "grad" 6.0
    (value
       (grad_of
          (fun a -> [ C.checkpoint_name (mul (sole a) (sole a)) "n" ])
          (scalar 3.0)));
  let cj = Api.jit_flat (fun a -> [ C.checkpoint_name (sole a) "n" ]) in
  check "jit" 3.0 (value (sole (cj [ scalar 3.0 ])));
  match C.checkpoint_name_tree (T.Dict [ ("a", leaf (scalar 1.0)) ]) "x" with
  | T.Dict [ ("a", T.Leaf v) ] -> check "tree" 1.0 (value v)
  | _ -> Alcotest.fail "tree shape"

let x_fb = vec [| 1.0; 2.0 |]
let w_fb = mat 2 2 [| 1.; 2.; 3.; 4. |]

let dot_args args =
  match args with
  | [ T.Leaf x; T.Leaf w ] -> leaf (Core.bind1 (dot_dims 0 0) [ sin x; w ])
  | _ -> Alcotest.fail "two arguments"

let fwd_bwd_two_args () =
  let fwd, bwd = Api.fwd_and_bwd ~argnums:[ 0; 1 ] dot_args in
  let y, res, _ = fwd [ leaf x_fb; leaf w_fb ] in
  check_arr "primal" [| 3.569363; 5.3201313 |] (unleaf y);
  match bwd res (leaf (vec [| 1.0; 1.0 |])) with
  | [ T.Leaf cx; T.Leaf cw ] ->
      check_arr "cx" [| 1.6209068; -2.913028 |] cx;
      check_arr "cw" [| 0.84147096; 0.84147096; 0.9092974; 0.9092974 |] cw
  | _ -> Alcotest.fail "two cotangents"

let fwd_bwd_one_arg () =
  let fwd, bwd = Api.fwd_and_bwd ~argnums:[ 0 ] dot_args in
  let y, res, _ = fwd [ leaf x_fb; leaf w_fb ] in
  check_arr "primal" [| 3.569363; 5.3201313 |] (unleaf y);
  match bwd res (leaf (vec [| 1.0; 0.0 |])) with
  | [ T.Leaf cx ] -> check_arr "cx" [| 0.5403023; -1.2484405 |] cx
  | _ -> Alcotest.fail "one cotangent"

let fwd_bwd_second_arg_unjitted () =
  let fwd, bwd = Api.fwd_and_bwd ~jitted:false ~argnums:[ 1 ] dot_args in
  let y, res, _ = fwd [ leaf x_fb; leaf w_fb ] in
  check_arr "primal" [| 3.569363; 5.3201313 |] (unleaf y);
  match bwd res (leaf (vec [| 1.0; 1.0 |])) with
  | [ T.Leaf cw ] ->
      check_arr "cw" [| 0.84147096; 0.84147096; 0.9092974; 0.9092974 |] cw
  | _ -> Alcotest.fail "one cotangent"

let fwd_bwd_aux () =
  let f args =
    match args with
    | [ T.Leaf x ] ->
        T.Tuple
          [ leaf (mul x x); T.Dict [ ("aux", leaf (add x (scalar 1.0))) ] ]
    | _ -> Alcotest.fail "one argument"
  in
  let fwd, bwd = Api.fwd_and_bwd ~has_aux:true ~argnums:[ 0 ] f in
  let y, res, aux = fwd [ leaf (scalar 3.0) ] in
  check "primal" 9.0 (value (unleaf y));
  (match aux with
  | T.Dict [ ("aux", T.Leaf a) ] -> check "aux" 4.0 (value a)
  | _ -> Alcotest.fail "aux tree");
  match bwd res (leaf (scalar 1.0)) with
  | [ T.Leaf c ] -> check "ct" 6.0 (value c)
  | _ -> Alcotest.fail "one cotangent"

let counted_square counts =
  Cd.custom_vjp
    (fun a -> [ mul (sole a) (sole a) ])
    ~fwd:(fun a ->
      incr (fst counts);
      ([ mul (sole a) (sole a) ], a))
    ~bwd:(fun res cts ->
      incr (snd counts);
      [ mul (mul (scalar 2.0) (sole res)) (sole cts) ])

let fwd_bwd_compiles_once () =
  let run jitted =
    let counts = (ref 0, ref 0) in
    let sq = counted_square counts in
    let f args =
      match args with
      | [ T.Leaf x ] -> leaf (mul (sole (sq [ x ])) (scalar 3.0))
      | _ -> Alcotest.fail "one argument"
    in
    let fwd, bwd = Api.fwd_and_bwd ~jitted ~argnums:[ 0 ] f in
    List.iter
      (fun (v, ey, ec) ->
        let y, res, _ = fwd [ leaf (scalar v) ] in
        check "primal" ey (value (unleaf y));
        match bwd res (leaf (scalar 1.0)) with
        | [ T.Leaf c ] -> check "ct" ec (value c)
        | _ -> Alcotest.fail "one cotangent")
      [ (1.0, 3.0, 6.0); (2.0, 12.0, 12.0); (3.0, 27.0, 18.0) ];
    (!(fst counts), !(snd counts))
  in
  let f_jit, b_jit = run true in
  Alcotest.(check (pair int int)) "jitted traces once" (1, 1) (f_jit, b_jit);
  let f_raw, b_raw = run false in
  Alcotest.(check (pair int int)) "unjitted runs each time" (3, 3) (f_raw, b_raw)

let fwd_bwd_trees () =
  let f args =
    match args with
    | [ T.Leaf x ] ->
        T.Dict
          [
            ("a", leaf (mul x x)); ("b", T.Tuple [ leaf (mul x (scalar 5.0)) ]);
          ]
    | _ -> Alcotest.fail "one argument"
  in
  let fwd, bwd = Api.fwd_and_bwd ~argnums:[ 0 ] f in
  let y, res, _ = fwd [ leaf (scalar 1.0) ] in
  (match y with
  | T.Dict [ ("a", T.Leaf a); ("b", T.Tuple [ T.Leaf b ]) ] ->
      check "a" 1.0 (value a);
      check "b" 5.0 (value b)
  | _ -> Alcotest.fail "out tree");
  let ct =
    T.Dict [ ("a", leaf (scalar 2.0)); ("b", T.Tuple [ leaf (scalar 3.0) ]) ]
  in
  (match bwd res ct with
  | [ T.Leaf c ] -> check "ct" 19.0 (value c)
  | _ -> Alcotest.fail "ct");
  raises_with "wrong tree" "unexpected tree structure" (fun () ->
      bwd res (T.Tuple [ leaf (scalar 1.0) ]));
  let g args =
    match args with
    | [ T.Dict [ ("a", T.Leaf a); ("b", T.Leaf b) ] ] -> leaf (mul a b)
    | _ -> Alcotest.fail "dict"
  in
  let fwd2, bwd2 = Api.fwd_and_bwd ~argnums:[ 0 ] g in
  let input = T.Dict [ ("a", leaf (scalar 1.0)); ("b", leaf (scalar 2.0)) ] in
  let y2, res2, _ = fwd2 [ input ] in
  check "dict primal" 2.0 (value (unleaf y2));
  match bwd2 res2 (leaf (scalar 1.0)) with
  | [ T.Dict [ ("a", T.Leaf a); ("b", T.Leaf b) ] ] ->
      check "da" 2.0 (value a);
      check "db" 1.0 (value b)
  | _ -> Alcotest.fail "dict ct"

let fwd_bwd_nondiff_and_nested () =
  let f args =
    match args with
    | [ T.Leaf x; T.Leaf c ] -> leaf (mul x c)
    | _ -> Alcotest.fail "two arguments"
  in
  let fwd, bwd = Api.fwd_and_bwd ~argnums:[ 0 ] f in
  List.iter
    (fun c ->
      let _, res, _ = fwd [ leaf (scalar 2.0); leaf (scalar c) ] in
      Alcotest.(check bool)
        "residuals are exposed" true
        (Api.residual_values res <> []);
      match bwd res (leaf (scalar 1.0)) with
      | [ T.Leaf ct ] -> check "non-differentiated arg is fresh" c (value ct)
      | _ -> Alcotest.fail "ct")
    [ 3.0; 4.0 ];
  let cube args =
    match args with
    | [ T.Leaf x ] -> leaf (mul x (mul x x))
    | _ -> Alcotest.fail "one argument"
  in
  let outer a =
    let fwd3, bwd3 = Api.fwd_and_bwd ~argnums:[ 0 ] cube in
    let _, res, _ = fwd3 [ leaf (sole a) ] in
    match bwd3 res (leaf (scalar 1.0)) with
    | [ T.Leaf ct ] -> [ ct ]
    | _ -> Alcotest.fail "ct"
  in
  check "grad through fwd and bwd" 12.0 (value (grad_of outer (scalar 2.0)));
  raises_with "argnums out of range" "argnums out of range" (fun () ->
      let fwd4, _ = Api.fwd_and_bwd ~argnums:[ 2 ] cube in
      fwd4 [ leaf (scalar 1.0) ])

module S = Shape_dtype_struct

let struct_basics () =
  let s = S.make [| 2; 3 |] Dtype.F32 in
  Alcotest.(check string)
    "repr" "ShapeDtypeStruct(shape=(2, 3), dtype=float32)" (S.to_string s);
  Alcotest.(check int) "ndim" 2 (S.ndim s);
  Alcotest.(check int) "size" 6 (S.size s);
  Alcotest.(check bool) "weak" false (S.weak_type s);
  Alcotest.(check bool) "equal" true (S.equal s (S.make [| 2; 3 |] Dtype.F32));
  Alcotest.(check bool)
    "weak differs" false
    (S.equal s (S.make ~weak_type:true [| 2; 3 |] Dtype.F32));
  Alcotest.(check bool)
    "shape differs" false
    (S.equal s (S.make [| 2 |] Dtype.F32));
  Alcotest.(check bool)
    "hash" true
    (S.hash s = S.hash (S.make [| 2; 3 |] Dtype.F32));
  Alcotest.(check string)
    "update shape" "ShapeDtypeStruct(shape=(4,), dtype=float32)"
    (S.to_string (S.update ~shape:[| 4 |] s));
  Alcotest.(check string)
    "update dtype" "ShapeDtypeStruct(shape=(2, 3), dtype=int32)"
    (S.to_string (S.update ~dtype:Dtype.I32 s));
  Alcotest.(check string)
    "update weak"
    "ShapeDtypeStruct(shape=(2, 3), dtype=float32, weak_type=True)"
    (S.to_string (S.update ~weak_type:true s));
  Alcotest.(check string)
    "f64 is kept" "ShapeDtypeStruct(shape=(), dtype=float64)"
    (S.to_string (S.make [||] Dtype.F64));
  Alcotest.(check string)
    "of value" "ShapeDtypeStruct(shape=(2,), dtype=float32)"
    (S.to_string (S.of_value (vec [| 1.0; 2.0 |])));
  match T.tree_leaves (T.Dict [ ("a", T.Leaf s) ]) with
  | [ l ] -> Alcotest.(check bool) "tree leaf" true (S.equal l s)
  | _ -> Alcotest.fail "one leaf"

let struct_eval_shape () =
  let matmul args =
    match args with
    | [ T.Leaf a; T.Leaf b ] ->
        let product = Core.bind1 (dot_dims 1 0) [ a; b ] in
        T.Tuple
          [ leaf product; leaf (Core.bind1 (Reduce_sum [| 0; 1 |]) [ a ]) ]
    | _ -> Alcotest.fail "two args"
  in
  (match
     Api.eval_shape_structs matmul
       [
         T.Leaf (S.make [| 2; 3 |] Dtype.F32);
         T.Leaf (S.make [| 3; 4 |] Dtype.F32);
       ]
   with
  | T.Tuple [ T.Leaf a; T.Leaf b ] ->
      Alcotest.(check string)
        "product" "ShapeDtypeStruct(shape=(2, 4), dtype=float32)"
        (S.to_string a);
      Alcotest.(check string)
        "sum" "ShapeDtypeStruct(shape=(), dtype=float32)" (S.to_string b)
  | _ -> Alcotest.fail "out tree");
  let twice args =
    match args with
    | [ T.Leaf a ] ->
        leaf
          (mul a
             (Ad_util.zeros_like_aval
                (S.update ~weak_type:false (S.of_value a))))
    | _ -> Alcotest.fail "one arg"
  in
  (match Api.eval_shape_structs twice [ T.Leaf (S.make [||] Dtype.F64) ] with
  | T.Leaf a ->
      Alcotest.(check string)
        "f64 canonicalizes when traced"
        "ShapeDtypeStruct(shape=(), dtype=float32)" (S.to_string a)
  | _ -> Alcotest.fail "leaf");
  let vmapped args =
    Api.vmap
      (fun inner ->
        match inner with
        | [ T.Leaf a ] -> leaf (Core.bind1 (Reduce_sum [| 0 |]) [ a ])
        | _ -> Alcotest.fail "one arg")
      [ Some 0 ] args
  in
  (match
     Api.eval_shape_structs vmapped [ T.Leaf (S.make [| 4; 2 |] Dtype.F32) ]
   with
  | T.Leaf a ->
      Alcotest.(check string)
        "vmap" "ShapeDtypeStruct(shape=(4,), dtype=float32)" (S.to_string a)
  | _ -> Alcotest.fail "leaf");
  let graded args =
    Api.grad
      (fun inner ->
        match inner with
        | [ T.Leaf a ] -> leaf (Core.bind1 (Reduce_sum [| 0; 1 |]) [ a ])
        | _ -> Alcotest.fail "one arg")
      args
  in
  match
    Api.eval_shape_structs graded [ T.Leaf (S.make [| 4; 2 |] Dtype.F32) ]
  with
  | T.Leaf a ->
      Alcotest.(check string)
        "grad" "ShapeDtypeStruct(shape=(4, 2), dtype=float32)" (S.to_string a)
  | _ -> Alcotest.fail "leaf"

let tree_keys () =
  T.register_attribute_keys "P" [ "x"; "y" ];
  let point =
    T.Custom ("P", "", [ T.Leaf 1.0; T.Dict [ ("a", T.Leaf 2.0) ] ])
  in
  let paths = T.tree_leaves_with_path point in
  Alcotest.(check (list string))
    "keystr attr" [ ".x"; ".y['a']" ]
    (List.map (fun (p, _) -> T.keystr p) paths);
  Alcotest.(check bool)
    "attr key" true
    (fst (List.hd paths) = [ T.Get_attr_key "x" ]);
  Alcotest.(check string)
    "keystr mixed" ".x['a'][1][<flat index 2>]"
    (T.keystr
       [
         T.Get_attr_key "x";
         T.Dict_key "a";
         T.Sequence_key 1;
         T.Flattened_index_key 2;
       ]);
  let mapped = T.tree_map_with_path (fun p _ -> T.keystr p) point in
  (match mapped with
  | T.Custom ("P", "", [ T.Leaf a; T.Dict [ ("a", T.Leaf b) ] ]) ->
      Alcotest.(check (pair string string)) "map" (".x", ".y['a']") (a, b)
  | _ -> Alcotest.fail "mapped shape");
  Alcotest.(check bool)
    "attr key differs from dict key" false
    (T.Get_attr_key "x" = T.Dict_key "x")

let one_level_keys () =
  let keys t = List.map fst (T.flatten_one_level_with_keys t) in
  Alcotest.(check bool)
    "dict sorted" true
    (keys (T.Dict [ ("b", T.Leaf 1); ("a", T.Leaf 2) ])
    = [ T.Dict_key "a"; T.Dict_key "b" ]);
  (match
     T.flatten_one_level_with_keys (T.Dict [ ("b", T.Leaf 1); ("a", T.Leaf 2) ])
   with
  | [ (_, T.Leaf 2); (_, T.Leaf 1) ] -> ()
  | _ -> Alcotest.fail "dict children");
  Alcotest.(check bool)
    "list" true
    (keys (T.List [ T.Leaf 1; T.List [ T.Leaf 2 ] ])
    = [ T.Sequence_key 0; T.Sequence_key 1 ]);
  Alcotest.(check bool)
    "tuple" true
    (keys (T.Tuple [ T.Leaf 1; T.Leaf 2 ])
    = [ T.Sequence_key 0; T.Sequence_key 1 ]);
  Alcotest.(check bool) "none" true (keys (T.Null : int T.t) = []);
  T.register_attribute_keys "Q" [ "x"; "y" ];
  Alcotest.(check bool)
    "registered attributes" true
    (keys (T.Custom ("Q", "", [ T.Leaf 1; T.Leaf 2 ]))
    = [ T.Get_attr_key "x"; T.Get_attr_key "y" ]);
  Alcotest.(check bool)
    "unregistered custom" true
    (keys (T.Custom ("C", "", [ T.Leaf 1; T.Leaf 2 ]))
    = [ T.Flattened_index_key 0; T.Flattened_index_key 1 ]);
  raises_with "leaf" "can't tree-flatten" (fun () ->
      T.flatten_one_level_with_keys (T.Leaf 1))

module R = Ref

let ref_basics () =
  let r = R.new_ref (vec [| 1.; 2.; 3. |]) in
  R.set ~index:[ R.At 1 ] r (scalar 5.0);
  let before = R.get r in
  R.addupdate ~index:[ R.At 0 ] r (scalar 10.0);
  let old = R.swap ~index:[ R.At 2 ] r (scalar 7.0) in
  check_arr "before" [| 1.; 5.; 3. |] before;
  check_arr "after" [| 11.; 5.; 7. |] (R.get r);
  check "swapped out" 3.0 (value old);
  Alcotest.(check (array int)) "shape" [| 3 |] (R.shape r);
  Alcotest.(check bool) "dtype" true (R.dtype r = Dtype.F32);
  Alcotest.(check int) "ndim" 1 (R.ndim r);
  Alcotest.(check int) "size" 3 (R.size r);
  Alcotest.(check string) "repr" "Ref{float32[3]}" (R.to_string r)

let ref_indexing () =
  let r = R.new_ref (mat 2 3 [| 0.; 1.; 2.; 3.; 4.; 5. |]) in
  check "at at" 1.0 (value (R.get ~index:[ R.At 0; R.At 1 ] r));
  check_arr "row" [| 3.; 4.; 5. |] (R.get ~index:[ R.At 1 ] r);
  check_arr "column" [| 1.; 4. |] (R.get ~index:[ R.All; R.At 1 ] r);
  check_arr "ranges" [| 0.; 1. |]
    (R.get ~index:[ R.Range (Some 0, Some 1); R.Range (Some 0, Some 2) ] r);
  check_arr "swap row" [| 3.; 4.; 5. |]
    (R.swap ~index:[ R.At 1 ] r (vec [| 0.; 0.; 0. |]));
  check_arr "after swap" [| 0.; 1.; 2.; 0.; 0.; 0. |] (R.get r);
  let q = R.new_ref (mat 2 3 [| 0.; 1.; 2.; 3.; 4.; 5. |]) in
  R.set ~index:[ R.At 1; R.At 2 ] q (scalar 9.0);
  R.addupdate ~index:[ R.At 0; R.At 0 ] q (scalar 1.0);
  R.addupdate ~index:[ R.At 1 ] q (vec [| 1.; 1.; 1. |]);
  check_arr "updates" [| 1.; 1.; 2.; 4.; 5.; 10. |] (R.get q);
  check "indexed get" 1.0 (value (R.get ~index:[ R.At 0; R.At 1 ] q));
  let s = R.new_ref (vec [| 0.; 1.; 2.; 3. |]) in
  R.set ~index:[ R.Range (Some 1, Some 3) ] s (vec [| 7.; 8. |]);
  check_arr "slice set" [| 0.; 7.; 8.; 3. |] (R.get s);
  R.set
    ~index:[ R.Dynamic (Concrete (Ndarray.of_floats Dtype.I32 [||] [| 2. |])) ]
    s (scalar 1.0);
  check_arr "dynamic set" [| 0.; 7.; 1.; 3. |] (R.get s);
  check "dynamic get" 3.0
    (value
       (R.get
          ~index:
            [ R.Dynamic (Concrete (Ndarray.of_floats Dtype.I32 [||] [| 3. |])) ]
          s));
  check "negative" 3.0 (value (R.get ~index:[ R.At (-1) ] s));
  raises_with "oob get" "Out of bound indexer: idx=10, dim=4." (fun () ->
      R.get ~index:[ R.At 10 ] s);
  raises_with "oob set" "Out of bound indexer: idx=10, dim=4." (fun () ->
      R.set ~index:[ R.At 10 ] s (scalar 1.0))

let ref_checks () =
  let r = R.new_ref (vec [| 0.; 0. |]) in
  raises_with "whole shape"
    "Invalid shape for `swap`. Ref shape: (2,). Expected shape: (2,). Value \
     shape: (3,)." (fun () -> R.set r (vec [| 1.; 2.; 3. |]));
  raises_with "scalar is not broadcast"
    "Invalid shape for `swap`. Ref shape: (2,). Expected shape: (2,). Value \
     shape: ()." (fun () -> R.set r (scalar 1.0));
  let four = R.new_ref (vec [| 0.; 1.; 2.; 3. |]) in
  raises_with "slice shape"
    "Invalid shape for `swap`. Ref shape: (4,). Expected shape: (2,). Value \
     shape: ()." (fun () ->
      R.set ~index:[ R.Range (Some 1, Some 3) ] four (scalar 5.0));
  raises_with "addupdate shape"
    "Invalid shape for `addupdate`. Ref shape: (4,). Expected shape: (2,). \
     Value shape: ()." (fun () ->
      R.addupdate ~index:[ R.Range (Some 1, Some 3) ] four (scalar 5.0));
  raises_with "dtype"
    "Invalid dtype for `swap`. Ref dtype: float32. Value dtype: int32."
    (fun () ->
      R.set r (Concrete (Ndarray.of_floats Dtype.I32 [| 2 |] [| 1.; 1. |])))

let ref_freeze_free () =
  let r = R.new_ref (vec [| 1.; 1. |]) in
  let f = R.freeze r in
  R.set ~index:[ R.At 0 ] r (scalar 3.0);
  check_arr "frozen value is detached" [| 1.; 1. |] f;
  check_arr "eager ref still usable" [| 3.; 1. |] (R.get r);
  let g = R.new_ref (vec [| 1.; 1. |]) in
  R.free_ref g;
  check_arr "eager free is inert" [| 1.; 1. |] (R.get g);
  let jit0 f = Api.jit_flat f [ scalar 1.0 ] in
  check "freeze in jit" 2.0
    (value
       (sole
          (jit0 (fun a ->
               let q = R.new_ref (sole a) in
               R.set q (mul (R.get q) (scalar 2.0));
               [ R.freeze q ]))));
  raises_with "set after freeze" "`swap` must be called on `Ref` types"
    (fun () ->
      jit0 (fun a ->
          let q = R.new_ref (sole a) in
          let y = R.freeze q in
          R.set q (mul y (scalar 2.0));
          [ R.get q ]));
  raises_with "get after freeze" "`get` must be called on `Ref` types"
    (fun () ->
      jit0 (fun a ->
          let q = R.new_ref (sole a) in
          let y = R.freeze q in
          [ add (R.get q) y ]));
  raises_with "get after free" "`get` must be called on `Ref` types" (fun () ->
      jit0 (fun a ->
          let q = R.new_ref (sole a) in
          R.free_ref q;
          [ R.get q ]));
  check "free in jit" 1.0
    (value
       (sole
          (jit0 (fun a ->
               R.free_ref (R.new_ref (sole a));
               a))));
  let outer = R.new_ref (vec [| 1.; 1. |]) in
  raises_with "freeze outer ref in jit" "created outside" (fun () ->
      jit0 (fun a ->
          ignore (R.freeze outer);
          a));
  raises_with "free outer ref in jit" "created outside" (fun () ->
      jit0 (fun a ->
          R.free_ref outer;
          a));
  check_arr "outer ref is untouched" [| 1.; 1. |] (R.get outer)

let ref_empty () =
  let st = S.make [| 2 |] Dtype.F32 in
  raises_with "eager" "Evaluation rule for 'empty_ref' not implemented"
    (fun () -> R.empty_ref st);
  let f a =
    let r = R.empty_ref st in
    R.set r (widen 2 (sole a));
    [ R.get r ]
  in
  check_arr "jit" [| 2.; 2. |] (sole (Api.jit_flat f [ scalar 2.0 ]));
  let read a =
    ignore a;
    [ R.get (R.empty_ref st) ]
  in
  let out = sole (Api.jit_flat read [ scalar 2.0 ]) in
  Alcotest.(check (array int))
    "jit uninitialized shape" [| 2 |] (Core.get_aval out).shape;
  let scalar_st = S.make [||] Dtype.F32 in
  let sq a =
    let r = R.empty_ref scalar_st in
    R.set r (mul (sole a) (sole a));
    [ R.freeze r ]
  in
  let sq_jit = Api.jit_flat sq in
  check "value" 9.0 (value (sole (sq_jit [ scalar 3.0 ])));
  check "grad of jit" 6.0 (value (grad_of sq_jit (scalar 3.0)));
  check "jit of grad" 6.0
    (value
       (sole
          (Api.jit_flat (fun a -> [ grad_of sq_jit (sole a) ]) [ scalar 3.0 ])));
  raises_with "grad, no jit" "Evaluation rule for 'empty_ref' not implemented"
    (fun () -> grad_of sq (scalar 3.0))

let ref_jit () =
  let r = R.new_ref (vec [| 0.; 0.; 0. |]) in
  let traces = ref 0 in
  let f =
    Api.jit_flat (fun a ->
        incr traces;
        R.set ~index:[ R.At 0 ] r (add (R.get ~index:[ R.At 0 ] r) (sole a));
        [ mul (R.get ~index:[ R.At 0 ] r) (scalar 2.0) ])
  in
  check "first" 4.0 (value (sole (f [ scalar 2.0 ])));
  check_arr "after first" [| 2.; 0.; 0. |] (R.get r);
  check "second" 14.0 (value (sole (f [ scalar 5.0 ])));
  check_arr "after second" [| 7.; 0.; 0. |] (R.get r);
  Alcotest.(check int) "traced once" 1 !traces;
  let one = R.new_ref (scalar 1.0) in
  let read_traces = ref 0 in
  let g =
    Api.jit_flat (fun a ->
        incr read_traces;
        [ add (sole a) (R.get one) ])
  in
  check "reads contents" 2.0 (value (sole (g [ scalar 1.0 ])));
  R.set one (scalar 10.0);
  check "reads new contents" 11.0 (value (sole (g [ scalar 1.0 ])));
  Alcotest.(check int) "no retrace" 1 !read_traces;
  let t = R.new_ref (scalar 1.0) in
  let tree_jit =
    Api.jit (fun args ->
        match args with
        | [ T.Leaf x ] ->
            R.set t (add (R.get t) x);
            leaf (R.get t)
        | _ -> Alcotest.fail "one arg")
  in
  check "tree jit" 3.0 (value (unleaf (tree_jit [ leaf (scalar 2.0) ])));
  check "tree jit state" 3.0 (value (R.get t));
  let n = R.new_ref (scalar 1.0) in
  let inner a =
    R.set n (add (R.get n) (sole a));
    [ R.get n ]
  in
  let outer =
    Api.jit_flat (fun a ->
        let y = sole (Api.jit_flat inner a) in
        [ add (mul y (scalar 2.0)) (R.get n) ])
  in
  check "nested jit" 6.0 (value (sole (outer [ scalar 1.0 ])));
  check "nested jit state" 2.0 (value (R.get n));
  let local =
    Api.jit_flat (fun a ->
        let q = R.new_ref (vec [| 1.; 1. |]) in
        R.set ~index:[ R.At 0 ] q (sole a);
        [ R.get q ])
  in
  check_arr "jit local" [| 2.; 1. |] (sole (local [ scalar 2.0 ]));
  let indexed = R.new_ref (vec [| 0.; 1.; 2.; 3. |]) in
  let traced_index =
    Api.jit_flat (fun a ->
        let i = [ R.Dynamic (sole a) ] in
        R.set ~index:i indexed (mul (R.get ~index:i indexed) (scalar 10.0));
        [])
  in
  let i32 x = Concrete (Ndarray.of_floats Dtype.I32 [||] [| x |]) in
  ignore (traced_index [ i32 1.0 ]);
  check_arr "traced index" [| 0.; 10.; 2.; 3. |] (R.get indexed);
  ignore (traced_index [ i32 2.0 ]);
  check_arr "traced index again" [| 0.; 10.; 20.; 3. |] (R.get indexed);
  let bad = R.new_ref (scalar 1.0) in
  raises_with "jit shape" "Invalid shape for `swap`" (fun () ->
      Api.jit_flat
        (fun a ->
          R.set bad (sole a);
          a)
        [ vec [| 1.; 1. |] ])

let ref_eval_shape () =
  let r = R.new_ref (scalar 1.0) in
  let f args =
    match args with
    | [ T.Leaf x ] ->
        R.set r x;
        leaf x
    | _ -> Alcotest.fail "one arg"
  in
  ignore (Api.eval_shape f [ leaf (scalar 2.0) ]);
  check "eval_shape leaves state alone" 1.0 (value (R.get r));
  ignore (Api.make_jaxpr f [ leaf (scalar 2.0) ]);
  check "make_jaxpr leaves state alone" 1.0 (value (R.get r));
  match
    Api.eval_shape_structs
      (fun args ->
        match args with
        | [ T.Leaf x ] -> leaf (R.get (R.new_ref x))
        | _ -> Alcotest.fail "one arg")
      [ T.Leaf (S.make [| 2 |] Dtype.F32) ]
  with
  | T.Leaf a ->
      Alcotest.(check string)
        "local ref" "ShapeDtypeStruct(shape=(2,), dtype=float32)"
        (S.to_string a)
  | _ -> Alcotest.fail "leaf"

let ref_grad_local () =
  let f a =
    let x = sole a in
    let r = R.new_ref x in
    R.set r (mul (sin (R.get r)) x);
    [ R.get r ]
  in
  let po, pull = Ad.vjp f [ scalar 2.0 ] in
  check "value" 1.8185948 (value (sole po));
  check "grad" 0.07700372 (value (sole (pull [ scalar 1.0 ])));
  let indexed a =
    let x = sole a in
    let r = R.new_ref (widen 3 x) in
    R.set ~index:[ R.At 1 ] r (mul x x);
    R.addupdate ~index:[ R.At 2 ] r (sin x);
    [ Core.bind1 (Reduce_sum [| 0 |]) [ R.get r ] ]
  in
  check "indexed grad" 5.5838532 (value (grad_of indexed (scalar 2.0)));
  let p, t = jvp_of indexed [ scalar 2.0 ] [ scalar 1.0 ] in
  check "indexed jvp primal" 8.909298 p;
  check "indexed jvp tangent" 5.5838532 t;
  check "indexed jit of grad" 5.5838532
    (value
       (sole
          (Api.jit_flat (fun a -> [ grad_of indexed (sole a) ]) [ scalar 2.0 ])));
  let frozen a =
    let r = R.new_ref (sole a) in
    let y = R.get r in
    R.set r (mul y y);
    [ R.freeze r ]
  in
  let po, pull = Ad.vjp frozen [ scalar 3.0 ] in
  check "vjp value" 9.0 (value (sole po));
  check "vjp grad" 6.0 (value (sole (pull [ scalar 1.0 ])));
  let lin_out, lin = Ad.linearize frozen [ scalar 3.0 ] in
  check "linearize value" 9.0 (value (sole lin_out));
  check "linearize tangent" 6.0 (value (sole (lin [ scalar 1.0 ])))

let ref_constant_init_under_differentiation () =
  let f a =
    let x = sole a in
    let r = R.new_ref (vec [| 0.; 0.; 0. |]) in
    R.set ~index:[ R.At 0 ] r x;
    R.set ~index:[ R.At 1 ] r (mul x x);
    R.addupdate ~index:[ R.At 2 ] r x;
    let old = R.swap ~index:[ R.At 0 ] r (scalar 0.0) in
    [ mul (Core.bind1 (Reduce_sum [| 0 |]) [ R.freeze r ]) old ]
  in
  let p, t = jvp_of f [ scalar 2.0 ] [ scalar 1.0 ] in
  check "jvp primal" 12.0 p;
  check "jvp tangent" 16.0 t;
  check "grad agrees with jvp" 16.0 (value (grad_of f (scalar 2.0)))

let ref_grad_inner_jit () =
  let f a =
    let x = sole a in
    let r = R.new_ref x in
    let inner y =
      R.set r (mul (R.get r) (sole y));
      [ add (R.get r) (scalar 1.0) ]
    in
    let v = sole (Api.jit_flat inner [ x ]) in
    [ add v (R.get r) ]
  in
  let po, pull = Ad.vjp f [ scalar 2.0 ] in
  check "value" 9.0 (value (sole po));
  check "grad" 8.0 (value (sole (pull [ scalar 1.0 ])));
  check "jit of grad" 8.0
    (value
       (sole (Api.jit_flat (fun a -> [ grad_of f (sole a) ]) [ scalar 2.0 ])))

let ref_grad_outer () =
  let r = R.new_ref (scalar 1.0) in
  raises_with "set differentiated"
    "performing a set/swap operation with a differentiated value on a \
     non-differentiated array reference of type Ref{float32[]}" (fun () ->
      grad_of
        (fun a ->
          R.set r (mul (sole a) (sole a));
          [ mul (sole a) (scalar 3.0) ])
        (scalar 2.0));
  raises_with "addupdate differentiated" "with a differentiated value"
    (fun () ->
      grad_of
        (fun a ->
          R.addupdate r (sole a);
          a)
        (scalar 2.0));
  raises_with "through jit" "with a differentiated value" (fun () ->
      let f =
        Api.jit_flat (fun a ->
            R.set r (mul (sole a) (scalar 2.0));
            [ mul (sole a) (scalar 3.0) ])
      in
      grad_of f (scalar 3.0));
  check "state survives errors" 1.0 (value (R.get r));
  let read = R.new_ref (scalar 1.0) in
  check "read outer" 1.0
    (value (grad_of (fun a -> [ mul (sole a) (R.get read) ]) (scalar 2.0)));
  let sg = R.new_ref (scalar 0.0) in
  check "stopped value" 2.0
    (value
       (grad_of
          (fun a ->
            R.set sg (Cd.stop_gradient (sole a));
            [ mul (sole a) (scalar 2.0) ])
          (scalar 3.0)));
  check "stopped value stored" 3.0 (value (R.get sg));
  let swap_ref = R.new_ref (scalar 0.0) in
  check "swap stopped" 0.0
    (value
       (grad_of
          (fun a ->
            let o = R.swap swap_ref (Cd.stop_gradient (sole a)) in
            [ mul (mul (sole a) (scalar 2.0)) o ])
          (scalar 3.0)));
  check "swap stored" 3.0 (value (R.get swap_ref));
  let persist = R.new_ref (scalar 1.0) in
  let f a =
    let x2 = Cd.stop_gradient (sole a) in
    R.set persist (add (R.get persist) x2);
    [ mul (sole a) (R.get persist) ]
  in
  let vg = Ad.vjp f [ scalar 3.0 ] in
  check "persist value" 12.0 (value (sole (fst vg)));
  check "persist grad" 4.0 (value (sole (snd vg [ scalar 1.0 ])));
  check "persist state" 4.0 (value (R.get persist));
  let jg =
    Api.jit_flat (fun a ->
        let po, pull = Ad.vjp f a in
        po @ pull [ scalar 1.0 ])
  in
  (match jg [ scalar 3.0 ] with
  | [ v; g ] ->
      check "jit persist value" 21.0 (value v);
      check "jit persist grad" 7.0 (value g)
  | _ -> Alcotest.fail "two outputs");
  check "jit persist state" 7.0 (value (R.get persist));
  let inner_stopped = R.new_ref (scalar 0.0) in
  let jitted =
    Api.jit_flat (fun a ->
        R.set inner_stopped (mul (sole a) (scalar 2.0));
        [ mul (sole a) (scalar 3.0) ])
  in
  let h a = [ add (sole (jitted [ Cd.stop_gradient (sole a) ])) (sole a) ] in
  check "jit with stopped input" 1.0 (value (grad_of h (scalar 3.0)));
  check "jit with stopped input state" 6.0 (value (R.get inner_stopped))

let ref_no_grad_kind () =
  let f a =
    let x = sole a in
    let r = R.new_ref ~kind:R.No_grad_no_remat (scalar 0.0) in
    R.set r (mul x (scalar 2.0));
    [ add (mul x x) (R.get r) ]
  in
  let po, pull = Ad.vjp f [ scalar 3.0 ] in
  check "local value" 15.0 (value (sole po));
  check "local grad" 6.0 (value (sole (pull [ scalar 1.0 ])));
  let p, t = jvp_of f [ scalar 3.0 ] [ scalar 1.0 ] in
  check "local jvp" 15.0 p;
  check "local jvp tangent" 6.0 t;
  let outer = R.new_ref ~kind:R.No_grad_no_remat (scalar 0.0) in
  let g a =
    let x = sole a in
    R.set outer (mul x (scalar 2.0));
    [ add (mul x x) (R.get outer) ]
  in
  let po, pull = Ad.vjp g [ scalar 3.0 ] in
  check "outer value" 15.0 (value (sole po));
  check "outer grad" 6.0 (value (sole (pull [ scalar 1.0 ])));
  check "outer state" 6.0 (value (R.get outer));
  let h a =
    let x = sole a in
    let r = R.new_ref ~kind:R.No_grad_no_remat x in
    [ mul (R.get r) x ]
  in
  let po, pull = Ad.vjp h [ scalar 3.0 ] in
  check "init value" 9.0 (value (sole po));
  check "init grad" 3.0 (value (sole (pull [ scalar 1.0 ])))

let ref_vmap () =
  let f a =
    let r = R.new_ref (sole a) in
    R.set r (mul (R.get r) (R.get r));
    [ R.freeze r ]
  in
  let xs = vec [| 0.; 1.; 2. |] in
  check_arr "local" [| 0.; 1.; 4. |] (sole (Batching.vmap f [ Some 0 ] [ xs ]));
  check_arr "vmap of grad" [| 0.; 2.; 4. |]
    (sole (Batching.vmap (fun a -> [ grad_of f (sole a) ]) [ Some 0 ] [ xs ]));
  let total a =
    [ Core.bind1 (Reduce_sum [| 0 |]) (Batching.vmap f [ Some 0 ] a) ]
  in
  check_arr "grad of vmap" [| 0.; 2.; 4. |] (grad_of total xs);
  let unbatched_init a =
    let r = R.new_ref (vec [| 0.; 0. |]) in
    R.set ~index:[ R.At 0 ] r (sole a);
    [ R.get r ]
  in
  check_arr "unbatched init"
    [| 0.; 0.; 1.; 0.; 2.; 0. |]
    (sole (Batching.vmap unbatched_init [ Some 0 ] [ xs ]));
  let outer = R.new_ref (scalar 0.0) in
  check_arr "read outer" [| 0.; 1.; 2. |]
    (sole
       (Batching.vmap
          (fun a -> [ add (sole a) (R.get outer) ])
          [ Some 0 ] [ xs ]));
  raises_with "set outer"
    "performing a set/swap operation with vmapped value on an unbatched array \
     reference of type Ref{float32[]}" (fun () ->
      Batching.vmap
        (fun a ->
          R.set outer (sole a);
          a)
        [ Some 0 ] [ xs ]);
  raises_with "addupdate outer"
    "performing an addupdate operation with vmapped value on an unbatched"
    (fun () ->
      Batching.vmap
        (fun a ->
          R.addupdate outer (sole a);
          a)
        [ Some 0 ] [ xs ]);
  raises_with "jit set outer" "with vmapped value" (fun () ->
      Batching.vmap
        (Api.jit_flat (fun a ->
             R.set outer (sole a);
             a))
        [ Some 0 ] [ xs ])

let ref_control_flow () =
  let r = R.new_ref (scalar 1.0) in
  let scanned =
    L.scan
      (fun args ->
        match args with
        | [ c; x ] ->
            R.set r (add (mul (R.get r) (scalar 2.0)) x);
            [ add c (scalar 1.0); R.get r ]
        | _ -> Alcotest.fail "arity")
      [ scalar 0.0 ]
      [ vec [| 0.; 1.; 2. |] ]
  in
  (match scanned with
  | [ c; ys ] ->
      check "scan carry" 3.0 (value c);
      check_arr "scan ys" [| 2.; 5.; 12. |] ys
  | _ -> Alcotest.fail "scan outs");
  check "scan state" 12.0 (value (R.get r));
  let w = R.new_ref (scalar 1.0) in
  let i =
    L.while_loop
      (fun s -> Core.bind1 Lt [ sole s; scalar 3.0 ])
      (fun s ->
        R.set w (mul (R.get w) (scalar 2.0));
        [ add (sole s) (scalar 1.0) ])
      [ scalar 0.0 ]
  in
  check "while result" 3.0 (value (sole i));
  check "while state" 8.0 (value (R.get w));
  let c = R.new_ref (scalar 1.0) in
  let branch pred =
    L.cond
      (Concrete (Ndarray.of_floats Dtype.Bool [||] [| pred |]))
      (fun _ ->
        R.set c (scalar 5.0);
        [ scalar 1.0 ])
      (fun _ -> [ scalar 2.0 ])
      []
  in
  check "cond taken" 1.0 (value (sole (branch 1.0)));
  check "cond state" 5.0 (value (R.get c));
  let d = R.new_ref (scalar 1.0) in
  let jitted =
    Api.jit_flat (fun a ->
        L.cond (sole a)
          (fun _ ->
            R.set d (scalar 5.0);
            [ scalar 1.0 ])
          (fun _ -> [ mul (R.get d) (scalar 3.0) ])
          [])
  in
  let bool_of x = Concrete (Ndarray.of_floats Dtype.Bool [||] [| x |]) in
  check "jit cond false" 3.0 (value (sole (jitted [ bool_of 0.0 ])));
  check "jit cond false state" 1.0 (value (R.get d));
  check "jit cond true" 1.0 (value (sole (jitted [ bool_of 1.0 ])));
  check "jit cond true state" 5.0 (value (R.get d))

let ref_control_flow_grad () =
  let f a =
    let x = sole a in
    let r = R.new_ref x in
    let ys =
      L.scan
        (fun args ->
          match args with
          | [ c; y ] ->
              R.set r (mul (R.get r) y);
              [ c; R.get r ]
          | _ -> Alcotest.fail "arity")
        [ scalar 0.0 ]
        [ vec [| 1.; 2.; 3. |] ]
    in
    [ Core.bind1 (Reduce_sum [| 0 |]) [ List.nth ys 1 ] ]
  in
  let po, pull = Ad.vjp f [ scalar 2.0 ] in
  check "scan value" 18.0 (value (sole po));
  check "scan grad" 9.0 (value (sole (pull [ scalar 1.0 ])));
  let p, t = jvp_of f [ scalar 2.0 ] [ scalar 1.0 ] in
  check "scan jvp" 18.0 p;
  check "scan jvp tangent" 9.0 t;
  check "scan jit of grad" 9.0
    (value
       (sole (Api.jit_flat (fun a -> [ grad_of f (sole a) ]) [ scalar 2.0 ])));
  let g a =
    let x = sole a in
    let r = R.new_ref x in
    ignore
      (L.while_loop
         (fun s -> Core.bind1 Lt [ sole s; scalar 2.0 ])
         (fun s ->
           R.set r (mul (R.get r) x);
           [ add (sole s) (scalar 1.0) ])
         [ scalar 0.0 ]);
    [ R.get r ]
  in
  let p, t = jvp_of g [ scalar 2.0 ] [ scalar 1.0 ] in
  check "while jvp" 8.0 p;
  check "while jvp tangent" 12.0 t;
  check "while jit" 8.0 (value (sole (Api.jit_flat g [ scalar 2.0 ])));
  let h a =
    let x = sole a in
    let r = R.new_ref x in
    ignore
      (L.cond
         (Core.bind1 Gt [ x; scalar 1.0 ])
         (fun _ ->
           R.set r (mul (R.get r) x);
           [])
         (fun _ -> [])
         []);
    [ R.get r ]
  in
  let vg x =
    let po, pull = Ad.vjp h [ scalar x ] in
    (value (sole po), value (sole (pull [ scalar 1.0 ])))
  in
  let v, d = vg 2.0 in
  check "cond true value" 4.0 v;
  check "cond true grad" 4.0 d;
  let v, d = vg 0.5 in
  check "cond false value" 0.5 v;
  check "cond false grad" 1.0 d;
  check "cond jit of grad" 4.0
    (value
       (sole (Api.jit_flat (fun a -> [ grad_of h (sole a) ]) [ scalar 2.0 ])))

let ref_under_checkpoint () =
  let r = R.new_ref (scalar 1.0) in
  raises_with "closed ref" "Ref closed over by a function under checkpoint"
    (fun () ->
      grad_of
        (C.checkpoint ~policy:C.everything_saveable (fun a ->
             [ mul (sole a) (R.get r) ]))
        (scalar 2.0));
  let local =
    C.checkpoint ~policy:C.everything_saveable (fun a ->
        let q = R.new_ref (sole a) in
        R.set q (mul (R.get q) (sole a));
        [ R.get q ])
  in
  check "local ref" 6.0 (value (grad_of local (scalar 3.0)))

let () =
  Alcotest.run "autodiff_api"
    [
      ( "custom_vjp symbolic zeros",
        [
          Alcotest.test_case "perturbed flags" `Quick vjp_flags;
          Alcotest.test_case "zero derivative inputs" `Quick
            vjp_zero_derivative_inputs;
          Alcotest.test_case "unused" `Quick vjp_unused;
          Alcotest.test_case "transforms" `Quick vjp_transforms;
          Alcotest.test_case "control flow" `Quick vjp_control_flow;
          Alcotest.test_case "nested" `Quick vjp_nested;
          Alcotest.test_case "forward raises" `Quick vjp_forward_raises;
          Alcotest.test_case "bwd outputs" `Quick bwd_outputs;
          Alcotest.test_case "primal trees" `Quick primal_trees;
          Alcotest.test_case "constructors" `Quick symbolic_constructors;
        ] );
      ( "custom_jvp symbolic zeros",
        [
          Alcotest.test_case "forward" `Quick custom_jvp_forward;
          Alcotest.test_case "reverse" `Quick custom_jvp_reverse;
          Alcotest.test_case "batched" `Quick custom_jvp_batched;
          Alcotest.test_case "control flow" `Quick custom_jvp_control_flow;
          Alcotest.test_case "rule outputs" `Quick custom_jvp_outputs;
        ] );
      ( "checkpoint policies",
        [
          Alcotest.test_case "sines" `Quick policy_sines;
          Alcotest.test_case "default" `Quick default_policy;
          Alcotest.test_case "dots" `Quick policy_dots;
          Alcotest.test_case "policy arguments" `Quick policy_arguments;
          Alcotest.test_case "batch dims" `Quick policy_batch_dims;
          Alcotest.test_case "names" `Quick policy_names;
          Alcotest.test_case "transforms" `Quick policy_transforms;
          Alcotest.test_case "closures" `Quick policy_closures;
          Alcotest.test_case "scan" `Quick policy_scan;
          Alcotest.test_case "forward under jit" `Quick policy_forward_under_jit;
          Alcotest.test_case "constant output" `Quick policy_constant_output;
          Alcotest.test_case "partial arguments" `Quick policy_partial_arguments;
          Alcotest.test_case "custom_vjp inside" `Quick policy_with_custom_vjp;
          Alcotest.test_case "nested" `Quick policy_nested;
          Alcotest.test_case "has_aux primal" `Quick has_aux_is_primal;
          Alcotest.test_case "checkpoint_name" `Quick names_transform;
        ] );
      ( "fwd_and_bwd",
        [
          Alcotest.test_case "two args" `Quick fwd_bwd_two_args;
          Alcotest.test_case "one arg" `Quick fwd_bwd_one_arg;
          Alcotest.test_case "unjitted" `Quick fwd_bwd_second_arg_unjitted;
          Alcotest.test_case "aux" `Quick fwd_bwd_aux;
          Alcotest.test_case "compiles once" `Quick fwd_bwd_compiles_once;
          Alcotest.test_case "trees" `Quick fwd_bwd_trees;
          Alcotest.test_case "nondiff and nested" `Quick
            fwd_bwd_nondiff_and_nested;
        ] );
      ( "ShapeDtypeStruct",
        [
          Alcotest.test_case "basics" `Quick struct_basics;
          Alcotest.test_case "eval_shape" `Quick struct_eval_shape;
        ] );
      ( "tree keys",
        [
          Alcotest.test_case "GetAttrKey" `Quick tree_keys;
          Alcotest.test_case "one level with keys" `Quick one_level_keys;
        ] );
      ( "Ref",
        [
          Alcotest.test_case "basics" `Quick ref_basics;
          Alcotest.test_case "indexing" `Quick ref_indexing;
          Alcotest.test_case "checks" `Quick ref_checks;
          Alcotest.test_case "freeze and free" `Quick ref_freeze_free;
          Alcotest.test_case "empty_ref" `Quick ref_empty;
          Alcotest.test_case "jit" `Quick ref_jit;
          Alcotest.test_case "eval_shape" `Quick ref_eval_shape;
          Alcotest.test_case "grad of local" `Quick ref_grad_local;
          Alcotest.test_case "constant init" `Quick
            ref_constant_init_under_differentiation;
          Alcotest.test_case "grad with inner jit" `Quick ref_grad_inner_jit;
          Alcotest.test_case "grad with outer ref" `Quick ref_grad_outer;
          Alcotest.test_case "no_grad_no_remat" `Quick ref_no_grad_kind;
          Alcotest.test_case "vmap" `Quick ref_vmap;
          Alcotest.test_case "control flow" `Quick ref_control_flow;
          Alcotest.test_case "control flow under grad" `Quick
            ref_control_flow_grad;
          Alcotest.test_case "under checkpoint" `Quick ref_under_checkpoint;
        ] );
    ]
