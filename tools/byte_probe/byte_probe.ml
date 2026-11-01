module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module U = Ojax.Numpy.Ufuncs
module S = Ojax.Scipy.Special

let unary =
  [
    ("sin", U.sin);
    ("cos", U.cos);
    ("tan", U.tan);
    ("exp", U.exp);
    ("expm1", U.expm1);
    ("log", U.log);
    ("log1p", U.log1p);
    ("sqrt", U.sqrt);
    ("cbrt", U.cbrt);
    ("sinh", U.sinh);
    ("cosh", U.cosh);
    ("tanh", U.tanh);
    ("arcsin", U.arcsin);
    ("arccos", U.arccos);
    ("arctan", U.arctan);
    ("arcsinh", U.arcsinh);
    ("arccosh", U.arccosh);
    ("arctanh", U.arctanh);
    ("erf", S.erf);
    ("erfc", S.erfc);
    ("gammaln", S.gammaln);
    ("digamma", S.digamma);
  ]

let binary = [ ("power", U.power); ("arctan2", U.arctan2) ]
let sizes = [ 3; 8; 32; 64; 128; 256; 512; 1024 ]

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr ic)
    (fun () -> In_channel.input_all ic)

let width = function D.F32 -> 4 | _ -> 8

let floats_of dt text offset count =
  Array.init count (fun i ->
      let at = offset + (i * width dt) in
      match dt with
      | D.F32 -> Int32.float_of_bits (String.get_int32_le text at)
      | _ -> Int64.float_of_bits (String.get_int64_le text at))

let write_floats dt path nd =
  let shape = Nd.shape nd in
  let total = Array.fold_left ( * ) 1 shape in
  let bytes = Bytes.create (total * width dt) in
  let rank = Array.length shape in
  let index = Array.make rank 0 in
  for flat = 0 to total - 1 do
    let rest = ref flat in
    for d = rank - 1 downto 0 do
      index.(d) <- !rest mod shape.(d);
      rest := !rest / shape.(d)
    done;
    let value = Nd.get_f nd index in
    let at = flat * width dt in
    match dt with
    | D.F32 -> Bytes.set_int32_le bytes at (Int32.bits_of_float value)
    | _ -> Bytes.set_int64_le bytes at (Int64.bits_of_float value)
  done;
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr oc)
    (fun () -> output_bytes oc bytes)

let host = function
  | T.Concrete nd -> nd
  | T.Device buffer -> Ojax.Pjrt.Buffer.to_host buffer
  | T.Tracer _ -> failwith "byte_probe: tracer result"

let run_one ~source ~target tag dt name shapes f =
  let stem suffix dir =
    Filename.concat dir (Printf.sprintf "%s_%s.%s" tag name suffix)
  in
  let text = read_file (stem "in" source) in
  let _, args =
    List.fold_left
      (fun (offset, args) shape ->
        let count = Array.fold_left ( * ) 1 shape in
        ( offset + (count * width dt),
          T.Concrete (Nd.of_floats dt shape (floats_of dt text offset count))
          :: args ))
      (0, []) shapes
  in
  match Ojax.Api.jit_flat (fun xs -> [ f xs ]) (List.rev args) with
  | [ out ] -> write_floats dt (stem "bin" target) (host out)
  | _ -> failwith "byte_probe: expected one result"

let () =
  let source = Sys.argv.(1) and target = Sys.argv.(2) in
  Ojax.Lax.install ();
  Ojax.Random.Prng.install ();
  Ojax.Random.Threefry.install ();
  List.iter
    (fun (x64, dt, tag) ->
      Ojax.Config.set Ojax.Config.enable_x64 x64;
      let grid = [ [| 100000 |] ] in
      List.iter
        (fun (name, f) ->
          run_one ~source ~target tag dt name grid (function
            | [ x ] -> f x
            | _ -> failwith "byte_probe: arity"))
        unary;
      List.iter
        (fun (name, f) ->
          run_one ~source ~target tag dt name (grid @ grid) (function
            | [ x; y ] -> f x y
            | _ -> failwith "byte_probe: arity"))
        binary;
      List.iter
        (fun n ->
          run_one ~source ~target tag dt
            (Printf.sprintf "matmul%d" n)
            [ [| n; n |]; [| n; n |] ]
            (function
              | [ x; y ] -> Ojax.Numpy.Tensor_contractions.matmul x y
              | _ -> failwith "byte_probe: arity"))
        sizes)
    [ (false, D.F32, "f32"); (true, D.F64, "f64") ]
