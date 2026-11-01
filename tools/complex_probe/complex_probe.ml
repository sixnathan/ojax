module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module U = Ojax.Numpy.Ufuncs
module NL = Ojax.Numpy.Lax_numpy

let py_nan = Int64.float_of_bits 0x7FF8000000000000L
let specials = [| 0.; -0.; 1.; -1.; infinity; neg_infinity; py_nan; 1e30 |]

let grid =
  Array.concat
    (Array.to_list
       (Array.map
          (fun re ->
            Array.map (fun im -> { Complex.re; im }) specials)
          specials))

let hex_of nd =
  let b = Buffer.create 64 in
  let dt = Nd.dtype nd in
  let shape = Nd.shape nd in
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
  (match dt with
  | D.Complex64 | D.Complex128 ->
      for i = 0 to total - 1 do
        let c = Nd.get_c nd (index i) in
        if dt = D.Complex64 then (
          Buffer.add_int32_le b (Int32.bits_of_float c.Complex.re);
          Buffer.add_int32_le b (Int32.bits_of_float c.Complex.im))
        else (
          Buffer.add_int64_le b (Int64.bits_of_float c.Complex.re);
          Buffer.add_int64_le b (Int64.bits_of_float c.Complex.im))
      done
  | D.F32 ->
      List.iter
        (fun x -> Buffer.add_int32_le b (Int32.bits_of_float x))
        (List.rev (Nd.fold (fun acc x -> x :: acc) [] nd))
  | D.Bool ->
      List.iter
        (fun x -> Buffer.add_char b (if x <> 0. then '\001' else '\000'))
        (List.rev (Nd.fold (fun acc x -> x :: acc) [] nd))
  | _ ->
      List.iter
        (fun x -> Buffer.add_int64_le b (Int64.bits_of_float x))
        (List.rev (Nd.fold (fun acc x -> x :: acc) [] nd)));
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq (Buffer.contents b))))

let row v =
  match Ojax.Backend.to_host_value v with
  | T.Concrete nd ->
      `Assoc
        [
          ("dtype", `String (D.numpy_name (Nd.dtype nd)));
          ("hex", `String (hex_of nd));
        ]
  | _ -> failwith "expected a host array"

let unary =
  [
    ("sin", U.sin); ("cos", U.cos); ("tan", U.tan); ("sinh", U.sinh);
    ("cosh", U.cosh); ("tanh", U.tanh); ("arcsin", U.arcsin);
    ("arccos", U.arccos); ("arctan", U.arctan); ("arcsinh", U.arcsinh);
    ("arccosh", U.arccosh); ("arctanh", U.arctanh); ("exp", U.exp);
    ("expm1", U.expm1); ("log", U.log); ("log1p", U.log1p); ("log2", U.log2);
    ("log10", U.log10); ("sqrt", U.sqrt); ("square", U.square);
    ("sign", U.sign); ("abs", U.abs); ("conj", U.conj); ("real", U.real);
    ("imag", U.imag); ("negative", U.negative); ("positive", U.positive);
    ("reciprocal", U.reciprocal); ("angle", fun x -> NL.angle x);
    ("exp2", U.exp2); ("isnan", U.isnan); ("isinf", U.isinf);
    ("isfinite", U.isfinite);
  ]

let binary =
  [
    ("add", U.add); ("subtract", U.subtract); ("multiply", U.multiply);
    ("divide", U.divide); ("power", U.power); ("equal", U.equal);
    ("not_equal", U.not_equal);
  ]

let () =
  let oc = open_out Sys.argv.(1) in
  let write name f =
    let v = try `List [ f () ] with e -> `String (Printexc.to_string e) in
    Yojson.Safe.to_channel oc (`Assoc [ ("case", `String name); ("value", v) ]);
    output_char oc '\n'
  in
  List.iter
    (fun (tag, dt, x64) ->
      Ojax.Config.set Ojax.Config.enable_x64 x64;
      let n = Array.length grid in
      let x = T.Concrete (Nd.of_complex dt [| n |] grid) in
      let swapped =
        T.Concrete
          (Nd.of_complex dt [| n |]
             (Array.init n (fun i -> grid.((i * 37) mod n))))
      in
      let jit1 f = Ojax.Api.jit_flat (function [ a ] -> [ f a ] | _ -> failwith "arity") in
      let jit2 f =
        Ojax.Api.jit_flat (function [ a; b ] -> [ f a b ] | _ -> failwith "arity")
      in
      List.iter
        (fun (name, f) ->
          write (Printf.sprintf "%s_%s_eager" name tag) (fun () -> row (f x));
          write (Printf.sprintf "%s_%s_jit" name tag) (fun () ->
              row (List.hd (jit1 f [ x ]))))
        unary;
      List.iter
        (fun (name, f) ->
          write (Printf.sprintf "%s_%s_eager" name tag) (fun () ->
              row (f x swapped));
          write (Printf.sprintf "%s_%s_jit" name tag) (fun () ->
              row (List.hd (jit2 f [ x; swapped ]))))
        binary)
    [ ("c64", D.Complex64, false); ("c128", D.Complex128, true) ];
  Ojax.Config.set Ojax.Config.enable_x64 true;
  let f64 shape data = T.Concrete (Nd.of_floats D.F64 shape data) in
  let bad = f64 [| 2; 2 |] [| -1.; 0.; 0.; 1. |] in
  let all_nan = f64 [| 2 |] [| py_nan; py_nan |] in
  let empty = f64 [| 0 |] [||] in
  let module R = Ojax.Numpy.Reductions in
  let module LA = Ojax.Numpy.Linalg in
  let module SP = Ojax.Scipy.Special in
  List.iter
    (fun (name, f, x) ->
      let g a = f a in
      write (name ^ "_eager") (fun () -> row (g x));
      write (name ^ "_jit") (fun () ->
          row
            (List.hd
               (Ojax.Api.jit_flat
                  (function [ a ] -> [ g a ] | _ -> failwith "arity")
                  [ x ]))))
    [
      ("cholesky_bad", (fun x -> LA.cholesky x), bad);
      ("inv_singular", LA.inv, f64 [| 2; 2 |] [| 0.; 0.; 0.; 0. |]);
      ("nanmean_all", (fun x -> R.nanmean x), all_nan);
      ("nanmax_all", (fun x -> R.nanmax x), all_nan);
      ("mean_empty", (fun x -> R.mean x), empty);
      ("erfinv_two", SP.erfinv, f64 [| 1 |] [| 2. |]);
      ("arcsin_two", U.arcsin, f64 [| 1 |] [| 2. |]);
      ("log_neg", U.log, f64 [| 1 |] [| -2. |]);
      ("ndtri_bad", SP.ndtri, f64 [| 1 |] [| 2. |]);
    ];
  close_out oc
