module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module S = Ojax.Sharding
module Tr = Ojax.Tree_util
module LA = Ojax.Numpy.Linalg
module SL = Ojax.Scipy.Linalg
module U = Ojax.Numpy.Ufuncs

let total shape = Array.fold_left ( * ) 1 shape

let zeros dt shape =
  T.Concrete (Nd.of_floats dt shape (Array.make (total shape) 0.))

let hex_of nd =
  let b = Buffer.create 64 in
  let dt = Nd.dtype nd in
  let add32 x = Buffer.add_int32_le b x in
  let add64 x = Buffer.add_int64_le b x in
  let shape = Nd.shape nd in
  let index i =
    let out = Array.make (Array.length shape) 0 in
    let rest = ref i in
    for d = Array.length shape - 1 downto 0 do
      out.(d) <- !rest mod shape.(d);
      rest := !rest / shape.(d)
    done;
    out
  in
  (match dt with
  | D.Complex64 | D.Complex128 ->
      for i = 0 to total shape - 1 do
        let c = Nd.get_c nd (index i) in
        if dt = D.Complex64 then (
          add32 (Int32.bits_of_float c.Complex.re);
          add32 (Int32.bits_of_float c.Complex.im))
        else (
          add64 (Int64.bits_of_float c.Complex.re);
          add64 (Int64.bits_of_float c.Complex.im))
      done
  | _ ->
      List.iter
        (fun x ->
          match dt with
          | D.F32 -> add32 (Int32.bits_of_float x)
          | D.F64 -> add64 (Int64.bits_of_float x)
          | D.I32 -> add32 (Int32.of_float x)
          | D.Bool -> Buffer.add_char b (if x <> 0. then '\001' else '\000')
          | _ -> add64 (Int64.of_float x))
        (List.rev (Nd.fold (fun acc x -> x :: acc) [] nd)));
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq (Buffer.contents b))))

let value_row v =
  match Ojax.Backend.to_host_value v with
  | T.Concrete nd ->
      `Assoc
        [
          ("dtype", `String (D.numpy_name (Nd.dtype nd)));
          ( "shape",
            `List (Array.to_list (Array.map (fun d -> `Int d) (Nd.shape nd))) );
          ("hex", `String (hex_of nd));
        ]
  | _ -> failwith "expected a host array"

let pair (a, b) = [ a; b ]
let f32 = D.F32
let c64 = D.Complex64
let named_qr = [ "result.Q"; "result.R" ]
let named_svd = [ "result.U"; "result.S"; "result.Vh" ]
let named_eig = [ "result.eigenvalues"; "result.eigenvectors" ]
let tuple2 = [ "result[0]"; "result[1]" ]
let tuple3 = [ "result[0]"; "result[1]"; "result[2]" ]
let one f xs = [ f (List.hd xs) ]
let plain f xs = f (List.hd xs)

let linalg_family suffix dt =
  let row name f shape paths =
    (name ^ suffix, plain f, [ zeros dt shape ], paths)
  in
  [
    row "qr_0x3" (fun x -> LA.qr x) [| 0; 3 |] (Some named_qr);
    row "qr_3x0" (fun x -> LA.qr x) [| 3; 0 |] (Some named_qr);
    row "qr_0x3_complete"
      (fun x -> LA.qr ~mode:"complete" x)
      [| 0; 3 |] (Some named_qr);
    row "qr_3x0_complete"
      (fun x -> LA.qr ~mode:"complete" x)
      [| 3; 0 |] (Some named_qr);
    row "qr_b2_0x3" (fun x -> LA.qr x) [| 2; 0; 3 |] (Some named_qr);
    row "lu_0x0" (fun x -> SL.lu x) [| 0; 0 |] (Some tuple3);
    row "lu_0x3" (fun x -> SL.lu x) [| 0; 3 |] (Some tuple3);
    row "lu_3x0" (fun x -> SL.lu x) [| 3; 0 |] (Some tuple3);
    row "svd_0x3" (fun x -> LA.svd x) [| 0; 3 |] (Some named_svd);
    row "svd_3x0" (fun x -> LA.svd x) [| 3; 0 |] (Some named_svd);
    row "svd_0x0_reduced"
      (fun x -> LA.svd ~full_matrices:false x)
      [| 0; 0 |] (Some named_svd);
    row "svd_0x3_reduced"
      (fun x -> LA.svd ~full_matrices:false x)
      [| 0; 3 |] (Some named_svd);
    row "svd_0x3_novec" (fun x -> LA.svd ~compute_uv:false x) [| 0; 3 |] None;
    row "eigh_0x0" (fun x -> pair (LA.eigh x)) [| 0; 0 |] (Some named_eig);
    row "eig_0x0" (fun x -> pair (LA.eig x)) [| 0; 0 |] (Some named_eig);
    row "schur_0x0" (fun x -> SL.schur x) [| 0; 0 |] (Some tuple2);
    row "hessenberg_0x0" (fun x -> SL.hessenberg x) [| 0; 0 |] None;
    row "hessenberg_q_0x0"
      (fun x -> SL.hessenberg ~calc_q:true x)
      [| 0; 0 |] (Some tuple2);
  ]

let eye2 = T.Concrete (Nd.of_floats f32 [| 2; 2 |] [| 2.; 1.; 1.; 3. |])

let named_tuples =
  [
    ("named_qr", plain (fun x -> LA.qr x), [ eye2 ], Some named_qr);
    ("named_svd", plain (fun x -> LA.svd x), [ eye2 ], Some named_svd);
    ("named_eigh", plain (fun x -> pair (LA.eigh x)), [ eye2 ], Some named_eig);
    ("named_eig", plain (fun x -> pair (LA.eig x)), [ eye2 ], Some named_eig);
    ( "named_slogdet",
      plain (fun x -> pair (LA.slogdet x)),
      [ eye2 ],
      Some [ "result.sign"; "result.logabsdet" ] );
  ]

let lu_rows =
  let m3 =
    T.Concrete (Nd.of_floats f32 [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 7. |])
  in
  let m2 =
    T.Concrete (Nd.of_floats f32 [| 2; 3 |] [| 1.; 4.; 2.; 5.; 3.; 7. |])
  in
  let c =
    T.Concrete
      (Nd.of_complex c64 [| 2; 2 |]
         [|
           { re = 1.; im = 1. };
           { re = 2.; im = 0. };
           { re = 0.; im = 1. };
           { re = 3.; im = -1. };
         |])
  in
  let scalar = T.Concrete (Nd.of_floats f32 [| 1; 1 |] [| -2. |]) in
  [
    ("lu_2x2", plain (fun x -> SL.lu x), [ eye2 ], Some tuple3);
    ("lu_3x2", plain (fun x -> SL.lu x), [ m3 ], Some tuple3);
    ("lu_2x3", plain (fun x -> SL.lu x), [ m2 ], Some tuple3);
    ("lu_c2x2", plain (fun x -> SL.lu x), [ c ], Some tuple3);
    ( "lu_permute_l",
      plain (fun x -> SL.lu ~permute_l:true x),
      [ eye2 ],
      Some tuple2 );
    ( "slogdet_c2x2",
      plain (fun x -> pair (LA.slogdet x)),
      [ c ],
      Some [ "result.sign"; "result.logabsdet" ] );
    ( "slogdet_1x1",
      plain (fun x -> pair (LA.slogdet x)),
      [ scalar ],
      Some [ "result.sign"; "result.logabsdet" ] );
  ]

let run oc (name, f, args, paths) =
  let row =
    let guard g =
      try g () with e -> `String ("ERR " ^ Printexc.to_string e)
    in
    let avals = List.map Ojax.Core.get_aval args in
    let lowered = lazy (Ojax.Stages.lower f avals) in
    [
      ( "text",
        guard (fun () ->
            `String
              (Ojax.Stages.as_jax_text ~name:"f" ?result_names:paths
                 (Lazy.force lowered))) );
      ( "jit",
        guard (fun () ->
            `List
              (List.map value_row
                 (Ojax.Stages.call
                    (Ojax.Stages.compile (Lazy.force lowered))
                    args))) );
      ("eager", guard (fun () -> `List (List.map value_row (f args))));
    ]
  in
  Yojson.Safe.to_channel oc (`Assoc (("case", `String name) :: row));
  output_char oc '\n';
  flush oc

module SP = Ojax.Shape_poly

let write_sym oc =
  let shape text = SP.symbolic_shape text in
  let one text = (shape (Some text)).(0) in
  let row name text =
    Yojson.Safe.to_channel oc
      (`Assoc
         [
           ("case", `String name);
           ("text", `String text);
           ("jit", `List []);
           ("eager", `List []);
         ]);
    output_char oc '\n'
  in
  let a = one "a" in
  let bc = shape (Some "b, c") in
  let b = bc.(0) and c = bc.(1) in
  let two = SP.of_int 2 and n1 = SP.of_int 1 in
  row "sym_a" (SP.dim_to_string a);
  row "sym_a1" (SP.dim_to_string (SP.add a n1));
  row "sym_2a" (SP.dim_to_string (SP.mul a two));
  row "sym_2a_1" (SP.dim_to_string (SP.sub (SP.mul two a) n1));
  row "sym_bc1" (SP.dim_to_string (SP.add (SP.mul b c) n1));
  row "sym_parse" (SP.shape_to_string (shape (Some "b+c, 2*b")));
  row "sym_floordiv" (SP.dim_to_string (SP.floordiv_dim a two));
  row "sym_mod" (SP.dim_to_string (SP.mod_dim a two));
  row "sym_max" (SP.dim_to_string (SP.max_dim a two));
  row "sym_min" (SP.dim_to_string (SP.min_dim b c));
  row "sym_pow" (SP.dim_to_string (SP.pow a 3));
  let f = function
    | [ x ] ->
        [
          Ojax.Numpy.Reductions.sum ~axis:[| 0 |]
            (U.multiply x (T.Concrete (Nd.of_floats f32 [||] [| 2. |])));
        ]
    | _ -> failwith "arity"
  in
  let s =
    Ojax.Export.export_symbolic f
      [ { SP.shape = shape (Some "a, 3"); dtype = f32 } ]
  in
  let ones = T.Concrete (Nd.of_floats f32 [| 4; 3 |] (Array.make 12 1.)) in
  Yojson.Safe.to_channel oc
    (`Assoc
       [
         ("case", `String "sym_call");
         ("text", `String "");
         ( "jit",
           `List (List.map value_row (Ojax.Export.call_symbolic s [ ones ])) );
         ("eager", `List []);
       ]);
  output_char oc '\n'

let () =
  let oc = open_out Sys.argv.(1) in
  write_sym oc;
  List.iter (run oc)
    (linalg_family "_f32" f32 @ linalg_family "_c64" c64 @ named_tuples
   @ lu_rows);
  ignore (one, S.make_mesh, Tr.keystr, U.sin);
  close_out oc
