module T = Ojax.Types
module D = Ojax.Dtype
module S = Ojax.Sharding
module Tr = Ojax.Tree_util
module U = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions

let leaf = function Tr.Leaf v -> v | _ -> failwith "expected a leaf"

let values v =
  match Ojax.Backend.to_host_value v with
  | T.Concrete nd ->
      Yojson.Safe.(
        `List
          (List.map
             (fun f -> `String (Int64.to_string (Int64.bits_of_float f)))
             (List.rev (Ojax.Ndarray.fold (fun acc x -> x :: acc) [] nd))))
  | _ -> failwith "expected a host array"

let arange =
  T.Concrete
    (Ojax.Ndarray.of_floats D.F32 [| 4; 4 |] (Array.init 16 float_of_int))

let mesh = lazy (S.make_mesh [ 1 ] [ "x" ])
let two = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 2.0 |])
let body x = U.add (U.multiply (U.sin x) two) (R.sum x)
let unary f = function [ Tr.Leaf a ] -> Tr.Leaf (f a) | _ -> failwith "arity"
let spec = [ S.Named "x"; S.Unconstrained ]

let cases () =
  let placed =
    leaf
      (S.device_put (Tr.Leaf arange) (S.named_sharding (Lazy.force mesh) spec))
  in
  [
    ("sin_sum", unary body, [ placed ], None);
    ( "out_sharded",
      unary body,
      [ placed ],
      Some [ S.named_sharding (Lazy.force mesh) spec ] );
    ( "constraint",
      unary (fun x ->
          leaf
            (S.with_sharding_constraint
               (Tr.Leaf (U.multiply x two))
               (S.named_sharding (Lazy.force mesh)
                  [ S.Unconstrained; S.Named "x" ]))),
      [ placed ],
      None );
    ( "shard_map",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh) ~in_specs:[ [ S.Named "x" ] ]
               ~out_specs:[ [ S.Named "x" ] ]
               (unary (fun y ->
                    U.add (U.sin y) (Ojax.Parallel.psum y ~axis_name:"x")))
               [ Tr.Leaf x ])),
      [ placed ],
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
