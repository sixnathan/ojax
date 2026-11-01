module T = Ojax.Types
module D = Ojax.Dtype
module E = Ojax.Numpy.Extras
module U = Ojax.Numpy.Ufuncs

let f32 shape data = T.Concrete (Ojax.Ndarray.of_floats D.F32 shape data)
let i32 shape data = T.Concrete (Ojax.Ndarray.of_floats D.I32 shape data)

let bits_of nd =
  let dt = Ojax.Ndarray.dtype nd in
  match dt with
  | D.Complex64 | D.Complex128 ->
      let shape = Ojax.Ndarray.shape nd in
      let total = Array.fold_left ( * ) 1 shape in
      let index i =
        let out = Array.make (Array.length shape) 0 in
        let rest = ref i in
        for d = Array.length shape - 1 downto 0 do
          out.(d) <- !rest mod shape.(d);
          rest := !rest / shape.(d)
        done;
        out
      in
      List.concat
        (List.init total (fun i ->
             let c = Ojax.Ndarray.get_c nd (index i) in
             [
               Int64.to_string (Int64.bits_of_float c.Complex.re);
               Int64.to_string (Int64.bits_of_float c.Complex.im);
             ]))
  | _ ->
      List.map
        (fun x -> Int64.to_string (Int64.bits_of_float x))
        (List.rev (Ojax.Ndarray.fold (fun acc x -> x :: acc) [] nd))

let value_row v =
  match Ojax.Backend.to_host_value v with
  | T.Concrete nd ->
      `Assoc
        [
          ("dtype", `String (D.numpy_name (Ojax.Ndarray.dtype nd)));
          ( "shape",
            `List
              (Array.to_list
                 (Array.map (fun d -> `Int d) (Ojax.Ndarray.shape nd))) );
          ("bits", `List (List.map (fun s -> `String s) (bits_of nd)));
        ]
  | _ -> failwith "expected a host array"

let a9 = [| 3.; 1.; 2.; 3.; 0.; -1.; 1.; 5.; 0. |]
let b6 = [| 1.; 7.; 2.; 9.; 0.; 3. |]
let x9 = f32 [| 9 |] a9
let y6 = f32 [| 6 |] b6
let m23 = f32 [| 2; 3 |] [| 0.; 1.; 0.; 2.; 0.; 3. |]
let ints8 = i32 [| 8 |] [| 1.; 3.; 3.; 0.; 7.; 2.; 1.; 1. |]

let cond =
  T.Concrete
    (Ojax.Ndarray.of_floats D.Bool [| 9 |]
       (Array.map (fun v -> if v > 0.0 then 1.0 else 0.0) a9))

let one f = function [ x ] -> f x | _ -> failwith "arity"
let two f = function [ x; y ] -> f x y | _ -> failwith "arity"

let cases =
  [
    ("nonzero_size", one (fun x -> E.nonzero ~size:4 x), [ m23 ]);
    ( "nonzero_size_fill",
      one (fun x -> E.nonzero ~size:5 ~fill_value:7 x),
      [ m23 ] );
    ("flatnonzero_size", one (fun x -> [ E.flatnonzero ~size:3 x ]), [ x9 ]);
    ("argwhere_size", one (fun x -> [ E.argwhere ~size:4 x ]), [ m23 ]);
    ("unique_size", one (fun x -> [ E.unique_values ~size:4 x ]), [ x9 ]);
    ( "unique_size_fill",
      one (fun x -> [ E.unique_values ~size:8 ~fill_value:(-9.) x ]),
      [ x9 ] );
    ( "unique_counts_size",
      one (fun x ->
          let v, c = E.unique_counts ~size:5 x in
          [ v; c ]),
      [ x9 ] );
    ( "unique_inverse_size",
      one (fun x ->
          let v, c = E.unique_inverse ~size:5 x in
          [ v; c ]),
      [ x9 ] );
    ( "unique_all_size",
      one (fun x ->
          let r = E.unique_all ~size:5 x in
          [ r.E.values; r.E.indices; r.E.inverse; r.E.counts ]),
      [ x9 ] );
    ("union1d_size", two (fun x y -> [ E.union1d ~size:8 x y ]), [ x9; y6 ]);
    ( "union1d_size_fill",
      two (fun x y -> [ E.union1d ~size:12 ~fill_value:(-1.) x y ]),
      [ x9; y6 ] );
    ( "intersect1d_size",
      two (fun x y -> [ E.intersect1d ~size:3 x y ]),
      [ x9; y6 ] );
    ( "intersect1d_size_fill",
      two (fun x y -> [ E.intersect1d ~size:6 ~fill_value:(-1.) x y ]),
      [ x9; y6 ] );
    ("setdiff1d_size", two (fun x y -> [ E.setdiff1d ~size:3 x y ]), [ x9; y6 ]);
    ( "setdiff1d_size_fill",
      two (fun x y -> [ E.setdiff1d ~size:6 ~fill_value:(-1.) x y ]),
      [ x9; y6 ] );
    ("setxor1d_size", two (fun x y -> [ E.setxor1d ~size:4 x y ]), [ x9; y6 ]);
    ( "setxor1d_size_fill",
      two (fun x y -> [ E.setxor1d ~size:9 ~fill_value:(-1.) x y ]),
      [ x9; y6 ] );
    ( "compress_size",
      two (fun c x -> [ E.compress ~size:4 ~fill_value:(-5.) c x ]),
      [ cond; x9 ] );
    ("extract_size", two (fun c x -> [ E.extract ~size:4 c x ]), [ cond; x9 ]);
    ("bincount_length", one (fun x -> [ E.bincount ~length:5 x ]), [ ints8 ]);
    ( "bincount_weights",
      two (fun x w -> [ E.bincount ~weights:w ~length:9 x ]),
      [ ints8; f32 [| 8 |] (Array.init 8 float_of_int) ] );
    ( "roots_nostrip",
      one (fun x -> [ E.roots ~strip_zeros:false x ]),
      [ f32 [| 3 |] [| 1.; -3.; 2. |] ] );
    ( "frexp_edges",
      one (fun x ->
          let m, e = E.frexp x in
          [ m; e ]),
      [
        f32 [| 8 |] [| 0.; -0.; 1.5; -3.; infinity; neg_infinity; nan; 1e-40 |];
      ] );
  ]

let () =
  let oc = open_out Sys.argv.(1) in
  List.iter
    (fun (name, f, args) ->
      let avals = List.map Ojax.Core.get_aval args in
      let row =
        let text =
          try
            let cj = Ojax.Jaxpr.make_jaxpr avals f in
            [
              ("text", `String (Ojax.Stablehlo.Emit.emit_jax_text ~name:"f" cj));
            ]
          with e -> [ ("text_error", `String (Printexc.to_string e)) ]
        in
        let jit =
          try
            let cj = Ojax.Jaxpr.make_jaxpr avals f in
            let outs = Ojax.Jaxpr.eval_closed_jaxpr cj args in
            [ ("jit", `List (List.map value_row outs)) ]
          with e -> [ ("jit_error", `String (Printexc.to_string e)) ]
        in
        let eager =
          try [ ("eager", `List (List.map value_row (f args))) ]
          with e -> [ ("eager_error", `String (Printexc.to_string e)) ]
        in
        text @ jit @ eager
      in
      Yojson.Safe.to_channel oc (`Assoc (("case", `String name) :: row));
      output_char oc '\n')
    cases;
  close_out oc
