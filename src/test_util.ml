open Types

let flat_floats v =
  let nd =
    match v with
    | Concrete nd -> nd
    | Device b -> Pjrt.Buffer.to_host b
    | Tracer _ -> invalid_arg "test_util: expected a concrete value"
  in
  Array.of_list (List.rev (Ndarray.fold (fun acc x -> x :: acc) [] nd))

let perturbed v deltas =
  let a = Core.get_aval v in
  let base = flat_floats v in
  Concrete
    (Ndarray.of_floats a.dtype a.shape
       (Array.mapi (fun i x -> x +. deltas.(i)) base))

let random_direction seed n =
  let state = ref (seed + 12345) in
  Array.init n (fun _ ->
      state := ((!state * 1103515245) + 12345) land 0x3FFFFFFF;
      (float_of_int !state /. float_of_int 0x3FFFFFFF) -. 0.5)

let default_eps dtype = match dtype with Dtype.F64 -> 1e-6 | _ -> 3e-3
let default_tol dtype = match dtype with Dtype.F64 -> 1e-5 | _ -> 5e-2

let assert_close name ~tol a b =
  let scale = 1.0 +. Float.max (Float.abs a) (Float.abs b) in
  if Float.abs (a -. b) > tol *. scale then
    failwith
      (Printf.sprintf "%s: analytic %.10g differs from numeric %.10g" name a b)

let directions args =
  List.mapi
    (fun i v -> random_direction (i + 1) (Array.length (flat_floats v)))
    args

let dot_flat a b =
  let total = ref 0.0 in
  Array.iteri (fun i x -> total := !total +. (x *. b.(i))) a;
  !total

let eps_and_tol ?eps ?tol args =
  let dt = (Core.get_aval (List.hd args)).dtype in
  ( Option.value eps ~default:(default_eps dt),
    Option.value tol ~default:(default_tol dt) )

let numeric_directional f args dirs eps =
  let shifted sign =
    List.map2
      (fun v d -> perturbed v (Array.map (fun x -> sign *. eps *. x) d))
      args dirs
  in
  let up = f (shifted 1.0) and down = f (shifted (-1.0)) in
  List.map2
    (fun u d ->
      Array.map2
        (fun a b -> (a -. b) /. (2.0 *. eps))
        (flat_floats u) (flat_floats d))
    up down

let tangents_of args dirs =
  List.map2
    (fun v d ->
      let a = Core.get_aval v in
      Concrete (Ndarray.of_floats a.dtype a.shape d))
    args dirs

let check_jvp ?eps ?tol f args =
  let eps, tol = eps_and_tol ?eps ?tol args in
  let dirs = directions args in
  let _, tangent_out = Interpreters.Ad.jvp f args (tangents_of args dirs) in
  let numeric = numeric_directional f args dirs eps in
  List.iter2
    (fun t n ->
      Array.iteri
        (fun i x -> assert_close "check_jvp" ~tol x n.(i))
        (flat_floats t))
    tangent_out numeric

let check_vjp ?eps ?tol f args =
  let eps, tol = eps_and_tol ?eps ?tol args in
  let dirs = directions args in
  let outs, pull = Interpreters.Ad.vjp f args in
  let cts =
    List.mapi
      (fun i o ->
        let a = Core.get_aval o in
        Concrete
          (Ndarray.of_floats a.dtype a.shape
             (random_direction (100 + i) (Array.length (flat_floats o)))))
      outs
  in
  let grads = pull cts in
  let lhs =
    List.fold_left2
      (fun acc g d -> acc +. dot_flat (flat_floats g) d)
      0.0 grads dirs
  in
  let numeric = numeric_directional f args dirs eps in
  let rhs =
    List.fold_left2
      (fun acc c n -> acc +. dot_flat (flat_floats c) n)
      0.0 cts numeric
  in
  assert_close "check_vjp" ~tol lhs rhs

let check_grads ?(modes = [ `Fwd; `Rev ]) ?(order = 1) ?eps ?tol f args =
  let rec check f order =
    List.iter
      (function
        | `Fwd -> check_jvp ?eps ?tol f args
        | `Rev -> check_vjp ?eps ?tol f args)
      modes;
    if order > 1 then
      let derivative inputs =
        let outs, pull = Interpreters.Ad.vjp f inputs in
        let cts =
          List.map
            (fun o ->
              let a = Core.get_aval o in
              let n = Array.fold_left ( * ) 1 a.shape in
              Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 1.0)))
            outs
        in
        pull cts
      in
      check derivative (order - 1)
  in
  check f order
