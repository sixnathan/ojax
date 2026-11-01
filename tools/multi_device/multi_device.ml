module T = Ojax.Types
module D = Ojax.Dtype
module S = Ojax.Sharding
module Tr = Ojax.Tree_util
module NL = Ojax.Numpy.Lax_numpy
module U = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module L = Ojax.Lax_api

let leaf = function Tr.Leaf v -> v | _ -> failwith "expected a leaf"
let host v = Ojax.Backend.to_host_value v

let values v =
  match host v with
  | T.Concrete nd ->
      Yojson.Safe.(
        `List
          (List.map
             (fun f -> `String (Int64.to_string (Int64.bits_of_float f)))
             (List.rev (Ojax.Ndarray.fold (fun acc x -> x :: acc) [] nd))))
  | _ -> failwith "expected a host array"

let arange16 =
  T.Concrete
    (Ojax.Ndarray.of_floats D.F32 [| 4; 4 |] (Array.init 16 float_of_int))

let arange8 =
  T.Concrete
    (Ojax.Ndarray.of_floats D.F32 [| 4; 2 |] (Array.init 8 float_of_int))

let vector8 =
  T.Concrete (Ojax.Ndarray.of_floats D.F32 [| 8 |] (Array.init 8 float_of_int))

let mesh22 = lazy (S.make_mesh [ 2; 2 ] [ "a"; "b" ])
let mesh2 = lazy (S.make_mesh [ 2 ] [ "x" ])
let mesh4 = lazy (S.make_mesh [ 4 ] [ "i" ])

let on mesh spec x =
  leaf (S.device_put (Tr.Leaf x) (S.named_sharding (Lazy.force mesh) spec))

let two = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 2.0 |])
let body x = U.add (U.multiply (U.sin x) two) (R.sum x)
let unary f = function [ Tr.Leaf a ] -> Tr.Leaf (f a) | _ -> failwith "arity"

let cases () =
  let ab = on mesh22 [ S.Named "a"; S.Named "b" ] arange16 in
  let x_only = on mesh2 [ S.Named "x"; S.Unconstrained ] arange8 in
  let i_only = on mesh4 [ S.Named "i" ] vector8 in
  [
    ("sin_sum", unary body, [ ab ], None);
    ( "out_sharded",
      unary body,
      [ ab ],
      Some
        [
          S.named_sharding (Lazy.force mesh22) [ S.Named "a"; S.Unconstrained ];
        ] );
    ( "two_results",
      (function
      | [ Tr.Leaf x; Tr.Leaf y ] ->
          Tr.Tuple
            [
              Tr.Leaf
                (Ojax.Numpy.Tensor_contractions.dot x (NL.matrix_transpose y));
              Tr.Leaf (U.exp y);
            ]
      | _ -> failwith "arity"),
      [ x_only; x_only ],
      None );
    ( "unused_argument",
      (function
      | [ Tr.Leaf x; Tr.Leaf _ ] -> Tr.Leaf (U.multiply x two)
      | _ -> failwith "arity"),
      [ ab; ab ],
      None );
    ( "constraint",
      unary (fun x ->
          leaf
            (S.with_sharding_constraint
               (Tr.Leaf (U.multiply x two))
               (S.named_sharding (Lazy.force mesh22)
                  [ S.Unconstrained; S.Named "a" ]))),
      [ ab ],
      None );
    ( "scan_in_shard_map",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [] ]
               (unary (fun y ->
                    let body = function
                      | [ carry; elt ] ->
                          [
                            L.add carry elt;
                            Ojax.Parallel.psum elt ~axis_name:"i";
                          ]
                      | _ -> failwith "arity"
                    in
                    match
                      L.scan body
                        [
                          T.Concrete
                            (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |]);
                        ]
                        [ y ]
                    with
                    | [ _; ys ] -> ys
                    | _ -> failwith "arity"))
               [ Tr.Leaf x ])),
      [ i_only ],
      None );
    ( "shard_map",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [ S.Named "i" ] ]
               (unary (fun y ->
                    U.add (U.sin y) (Ojax.Parallel.psum y ~axis_name:"i")))
               [ Tr.Leaf x ])),
      [ i_only ],
      None );
  ]

let () =
  let oc = open_out Sys.argv.(1) in
  List.iter
    (fun (name, f, args, out_shardings) ->
      let row =
        try
          let text =
            S.lower_text ?out_shardings ~name:"f" f
              (List.map (fun v -> Tr.Leaf v) args)
          in
          let result =
            S.jit ?out_shardings f (List.map (fun v -> Tr.Leaf v) args)
          in
          let leaves, _ = Tr.tree_flatten result in
          [ ("text", `String text); ("values", `List (List.map values leaves)) ]
        with e -> [ ("error", `String (Printexc.to_string e)) ]
      in
      Yojson.Safe.to_channel oc (`Assoc (("case", `String name) :: row));
      output_char oc '\n')
    (cases ());
  close_out oc
