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
    ( "sort_complex",
      one (fun x -> [ E.sort_complex x ]),
      [
        T.Concrete
          (Ojax.Ndarray.of_complex D.Complex64 [| 5 |]
             [|
               { Complex.re = 1.; im = 2. };
               { Complex.re = 1.; im = -1. };
               { Complex.re = Int64.float_of_bits 0x7FF8000000000000L; im = 0. };
               { Complex.re = -3.; im = 4. };
               { Complex.re = infinity; im = 0. };
             |]);
      ] );
    ( "hessenberg_1",
      one (fun x -> Ojax.Core.bind Hessenberg [ x ]),
      [ f32 [| 1; 1 |] [| 3. |] ] );
    ( "pure_callback",
      one (fun x ->
          let aval = Ojax.Core.get_aval x in
          let out =
            Ojax.Custom_derivatives.pure_callback
              (function
                | [ T.Concrete nd ] ->
                    [
                      T.Concrete (Ojax.Ndarray.map D.F32 (fun v -> v +. 1.0) nd);
                    ]
                | _ -> failwith "arity")
              ~result_avals:[ aval ] [ x ]
          in
          [
            U.multiply (List.hd out)
              (T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 2.0 |]));
          ]),
      [ f32 [| 3 |] [| 1.; 2.; 3. |] ] );
    ( "debug_callback",
      one (fun x ->
          Ojax.Debugging.callback (fun _ -> ()) [ x ];
          [ U.add x (T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 1.0 |])) ]),
      [ f32 [| 3 |] [| 1.; 2.; 3. |] ] );
    ( "frexp_edges",
      one (fun x ->
          let m, e = E.frexp x in
          [ m; e ]),
      [
        f32 [| 8 |]
          [|
            0.;
            -0.;
            1.5;
            -3.;
            infinity;
            neg_infinity;
            Int64.float_of_bits 0x7FF8000000000000L;
            1e-40;
          |];
      ] );
  ]

module JU = Yojson.Safe.Util

let array_of_json j =
  let re = List.map JU.to_number (JU.to_list (JU.member "re" j)) in
  let im = List.map JU.to_number (JU.to_list (JU.member "im" j)) in
  let shape =
    Array.of_list (List.map JU.to_int (JU.to_list (JU.member "shape" j)))
  in
  match JU.to_string (JU.member "dtype" j) with
  | "float32" ->
      T.Concrete (Ojax.Ndarray.of_floats D.F32 shape (Array.of_list re))
  | "float64" ->
      T.Concrete (Ojax.Ndarray.of_floats D.F64 shape (Array.of_list re))
  | name ->
      let dt = if name = "complex128" then D.Complex128 else D.Complex64 in
      T.Concrete
        (Ojax.Ndarray.of_complex dt shape
           (Array.of_list
              (List.map2 (fun r i -> { Complex.re = r; im = i }) re im)))

let tridiagonal_fuzz path =
  let cases = JU.to_list (Yojson.Safe.from_file path) in
  let total = ref 0 and same = ref 0 in
  Ojax.Config.with_value Ojax.Config.enable_x64 true @@ fun () ->
  List.iteri
    (fun i case ->
      let get n = array_of_json (JU.member n case) in
      let dl = get "dl" and d = get "d" and du = get "du" and b = get "b" in
      let wanted = List.map JU.to_string (JU.to_list (JU.member "x" case)) in
      let x =
        Ojax.Lax.Linalg.tridiagonal_solve ~perturb_singular:true dl d du b
      in
      let got =
        match Ojax.Backend.to_host_value x with
        | T.Concrete nd -> bits_of nd
        | _ -> failwith "host"
      in
      incr total;
      if got = wanted then incr same
      else
        Printf.printf "case %d differs\n  jax  %s\n  ojax %s\n" i
          (String.concat " " wanted) (String.concat " " got))
    cases;
  Printf.printf "tridiagonal perturbed: %d of %d identical\n" !same !total

let quiet_nan = Int64.float_of_bits 0x7FF8000000000000L

let specials =
  [| 0.0; -0.0; 1.0; -1.0; infinity; neg_infinity; quiet_nan; 1e30 |]

let grid dt =
  let n = Array.length specials in
  T.Concrete
    (Ojax.Ndarray.of_complex dt
       [| n * n |]
       (Array.init (n * n) (fun i ->
            { Complex.re = specials.(i / n); im = specials.(i mod n) })))

let complex_unary =
  [
    ("negative", U.negative);
    ("absolute", U.absolute);
    ("sign", U.sign);
    ("sqrt", U.sqrt);
    ("exp", U.exp);
    ("log", U.log);
    ("log2", U.log2);
    ("log10", U.log10);
    ("log1p", U.log1p);
    ("expm1", U.expm1);
    ("sin", U.sin);
    ("cos", U.cos);
    ("tan", U.tan);
    ("sinh", U.sinh);
    ("cosh", U.cosh);
    ("tanh", U.tanh);
    ("arcsin", U.arcsin);
    ("arccos", U.arccos);
    ("arctan", U.arctan);
    ("arcsinh", U.arcsinh);
    ("arccosh", U.arccosh);
    ("arctanh", U.arctanh);
    ("square", U.square);
    ("conj", U.conj);
    ("real", U.real);
    ("imag", U.imag);
    ("angle", fun v -> Ojax.Numpy.Lax_numpy.angle v);
    ("isnan", U.isnan);
    ("isinf", U.isinf);
    ("isfinite", U.isfinite);
    ("reciprocal", U.reciprocal);
  ]

let complex_binary =
  [
    ("add", U.add);
    ("subtract", U.subtract);
    ("multiply", U.multiply);
    ("divide", U.true_divide);
    ("power", U.power);
  ]

let complex_edges out =
  let oc = open_out out in
  let write set name mode dtype f =
    let row =
      try
        let v = f () in
        [ ("row", value_row v) ]
      with e -> [ ("error", `String (Printexc.to_string e)) ]
    in
    Yojson.Safe.to_channel oc
      (`Assoc
         (("key", `String (String.concat "/" [ set; name; mode; dtype ])) :: row));
    output_char oc '\n'
  in
  List.iter
    (fun (dt, label, x64) ->
      Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
          let g = grid dt in
          let aval = Ojax.Core.get_aval g in
          List.iter
            (fun (name, f) ->
              write "u" name "eager" label (fun () -> f g);
              write "u" name "jit" label (fun () ->
                  let cj =
                    Ojax.Jaxpr.make_jaxpr [ aval ] (function
                      | [ x ] -> [ f x ]
                      | _ -> failwith "arity")
                  in
                  List.hd (Ojax.Jaxpr.eval_closed_jaxpr cj [ g ])))
            complex_unary;
          let n = Array.length specials * Array.length specials in
          let left =
            Ojax.Numpy.Lax_numpy.reshape
              (Ojax.Numpy.Lax_numpy.repeat ~axis:0 g n)
              [| n * n |]
          in
          let right = Ojax.Numpy.Lax_numpy.tile g [| n |] in
          let la = Ojax.Core.get_aval left in
          List.iter
            (fun (name, f) ->
              write "b" name "eager" label (fun () -> f left right);
              write "b" name "jit" label (fun () ->
                  let cj =
                    Ojax.Jaxpr.make_jaxpr [ la; la ] (function
                      | [ x; y ] -> [ f x y ]
                      | _ -> failwith "arity")
                  in
                  List.hd (Ojax.Jaxpr.eval_closed_jaxpr cj [ left; right ])))
            complex_binary))
    [ (D.Complex64, "c64", false); (D.Complex128, "c128", true) ];
  close_out oc

let () =
  if Sys.argv.(1) = "tri" then begin
    tridiagonal_fuzz Sys.argv.(2);
    exit 0
  end;
  if Sys.argv.(1) = "complex" then begin
    complex_edges Sys.argv.(2);
    exit 0
  end;
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
