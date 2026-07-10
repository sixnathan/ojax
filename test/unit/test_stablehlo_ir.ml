module Ir = Ojax.Stablehlo.Ir
module Dtype = Ojax.Dtype
module Types = Ojax.Types
module U = Yojson.Safe.Util

let spec_path =
  match Sys.getenv_opt "OJAX_SPEC" with
  | Some d -> Filename.concat d "stablehlo_ir.cases.json"
  | None ->
      Filename.concat
        (Filename.concat (Filename.concat ".." "..") "spec")
        "stablehlo_ir.cases.json"

let spec = lazy (Yojson.Safe.from_file spec_path)

let dtype_of_tag = function
  | "F32" -> Dtype.F32
  | "F64" -> Dtype.F64
  | "I32" -> Dtype.I32
  | "I64" -> Dtype.I64
  | "Bool" -> Dtype.Bool
  | "Uint32" -> Dtype.Uint32
  | s -> Alcotest.failf "unknown dtype tag %s" s

let float_of_bits tag bits =
  match tag with
  | "F32" -> Int32.float_of_bits (Int32.of_string bits)
  | "F64" -> Int64.float_of_bits (Int64.of_string bits)
  | s -> Alcotest.failf "unknown float dtype %s" s

let element_types () =
  U.member "element_types" (Lazy.force spec)
  |> U.to_assoc
  |> List.iter (fun (tag, name) ->
      Alcotest.(check string)
        ("element_type " ^ tag) (U.to_string name)
        (Ir.element_type (dtype_of_tag tag)))

let tensor_types () =
  U.member "tensor_types" (Lazy.force spec)
  |> U.to_list
  |> List.iter (fun e ->
      let tag = U.member "dtype" e |> U.to_string in
      let shape =
        U.member "shape" e |> U.to_list |> List.map U.to_int |> Array.of_list
      in
      let want = U.member "text" e |> U.to_string in
      Alcotest.(check string)
        (Printf.sprintf "tensor_type %s" want)
        want
        (Ir.tensor_type (dtype_of_tag tag) shape))

let float_literals () =
  U.member "float_literals" (Lazy.force spec)
  |> U.to_list
  |> List.iter (fun e ->
      let tag = U.member "dtype" e |> U.to_string in
      let bits = U.member "bits" e |> U.to_string in
      let want = U.member "text" e |> U.to_string in
      let v = float_of_bits tag bits in
      Alcotest.(check string)
        (Printf.sprintf "float_literal %s %s" tag bits)
        want
        (Ir.float_literal (dtype_of_tag tag) v))

let int_literals () =
  U.member "int_literals" (Lazy.force spec)
  |> U.to_list
  |> List.iter (fun e ->
      let value = U.member "value" e |> U.to_string in
      let want = U.member "text" e |> U.to_string in
      Alcotest.(check string)
        (Printf.sprintf "int_literal %s" value)
        want
        (Ir.int_literal (Int64.of_string value)))

let bool_literals () =
  U.member "bool_literals" (Lazy.force spec)
  |> U.to_list
  |> List.iter (fun e ->
      let value = U.member "value" e |> U.to_bool in
      let want = U.member "text" e |> U.to_string in
      Alcotest.(check string)
        (Printf.sprintf "bool_literal %b" value)
        want (Ir.bool_literal value))

let scalar_tensor () =
  Alcotest.(check string)
    "scalar f32" "tensor<f32>"
    (Ir.tensor_type Dtype.F32 [||]);
  Alcotest.(check string)
    "bool i1" "tensor<i1>"
    (Ir.tensor_type Dtype.Bool [||]);
  Alcotest.(check string)
    "ranked ui32" "tensor<2x3xui32>"
    (Ir.tensor_type Dtype.Uint32 [| 2; 3 |])

let aval_type () =
  let a = { Types.shape = [| 4; 5 |]; dtype = Dtype.F64; weak_type = false } in
  Alcotest.(check string)
    "aval tensor" "tensor<4x5xf64>" (Ir.tensor_type_of_aval a)

let attrs () =
  Alcotest.(check string) "dense wrap" "dense<1.0>" (Ir.dense "1.0");
  Alcotest.(check string) "int array" "[1, 0]" (Ir.int_array_attr [| 1; 0 |]);
  Alcotest.(check string) "int array empty" "[]" (Ir.int_array_attr [||]);
  Alcotest.(check string) "int array single" "[3]" (Ir.int_array_attr [| 3 |]);
  Alcotest.(check string)
    "enum" "#stablehlo<comparison_direction LT>"
    (Ir.enum_attr "comparison_direction" "LT")

let version () =
  Alcotest.(check string) "target version" "1.17.0" Ir.target_version

let () =
  Alcotest.run "stablehlo_ir"
    [
      ( "types",
        [
          Alcotest.test_case "element types vs jax" `Quick element_types;
          Alcotest.test_case "tensor types vs jax" `Quick tensor_types;
          Alcotest.test_case "scalar/ranked" `Quick scalar_tensor;
          Alcotest.test_case "aval" `Quick aval_type;
        ] );
      ( "literals",
        [
          Alcotest.test_case "float literals vs jax" `Quick float_literals;
          Alcotest.test_case "int literals vs jax" `Quick int_literals;
          Alcotest.test_case "bool literals vs jax" `Quick bool_literals;
        ] );
      ( "attrs",
        [
          Alcotest.test_case "attribute printers" `Quick attrs;
          Alcotest.test_case "target version" `Quick version;
        ] );
    ]
