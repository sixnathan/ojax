open Ojax
open Ojax.Types
module R = Ojax.Random.Core

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ()

let key seed =
  R.key (Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int seed |]))

let nd = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let sole = function [ x ] -> x | _ -> Alcotest.fail "arity"

let matches name sampler =
  let k = key 7 in
  let direct = nd (sampler k) in
  let jitted = nd (sole (Api.jit_flat (fun a -> [ sampler (sole a) ]) [ k ])) in
  let shape = Ndarray.shape direct in
  Alcotest.(check (array int)) (name ^ " shape") shape (Ndarray.shape jitted);
  let n = Array.fold_left ( * ) 1 shape in
  let flat a =
    Array.of_list (List.rev (Ndarray.fold (fun acc x -> x :: acc) [] a))
  in
  let d = flat direct and j = flat jitted in
  for i = 0 to n - 1 do
    Alcotest.(check (float 1e-5)) name d.(i) j.(i)
  done

let shape = [| 3 |]

let cases =
  [
    ("uniform", fun k -> R.uniform k ~shape ~minval:0.0 ~maxval:1.0);
    ("normal", fun k -> R.normal k ~shape);
    ("randint", fun k -> R.randint k ~shape ~minval:0 ~maxval:10);
    ("bernoulli", fun k -> R.bernoulli k ~shape ~p:0.5);
    ("exponential", fun k -> R.exponential k ~shape);
    ( "categorical",
      fun k ->
        R.categorical k
          ~logits:
            (Concrete (Ndarray.of_floats Dtype.F32 [| 3 |] [| 0.1; 0.2; 0.3 |]))
          ~axis:(-1) );
    ("gumbel", fun k -> R.gumbel k ~shape);
    ("laplace", fun k -> R.laplace k ~shape);
    ("cauchy", fun k -> R.cauchy k ~shape);
    ("logistic", fun k -> R.logistic k ~shape);
    ("split", fun k -> R.key_data (R.split k 2));
    ( "truncated_normal",
      fun k -> R.truncated_normal k ~lower:(-1.0) ~upper:1.0 ~shape );
    ("permutation", fun k -> R.permutation k 4);
    ("gamma", fun k -> R.gamma k ~shape ~a:2.0);
    ("poisson", fun k -> R.poisson k ~shape ~lam:3.0);
    ( "vmap over keys",
      fun k ->
        let keys = R.split k 3 in
        List.hd
          (Interpreters.Batching.vmap
             (fun a -> [ R.normal (List.hd a) ~shape:[||] ])
             [ Some 0 ] [ keys ]) );
  ]

let () =
  Alcotest.run "random_traced"
    [
      ( "jit over key",
        List.map
          (fun (name, f) ->
            Alcotest.test_case name `Quick (fun () -> matches name f))
          cases );
    ]
