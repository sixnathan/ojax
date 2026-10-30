open Types

type exported = {
  mlir_module : string;
  in_avals : aval list;
  out_avals : aval list;
  platforms : string list;
}

let export f avals =
  let lowered = Stages.lower f avals in
  {
    mlir_module = Stages.as_text lowered;
    in_avals = lowered.Stages.in_avals;
    out_avals = lowered.Stages.out_avals;
    platforms = [ "cpu" ];
  }

let aval_to_string (a : aval) =
  Dtype.short_name a.dtype ^ ":"
  ^ String.concat "," (Array.to_list (Array.map string_of_int a.shape))

let dtype_of_name name =
  let all =
    [
      Dtype.F16;
      Dtype.Bf16;
      Dtype.F32;
      Dtype.F64;
      Dtype.I8;
      Dtype.I16;
      Dtype.I32;
      Dtype.I64;
      Dtype.Uint8;
      Dtype.Uint16;
      Dtype.Uint32;
      Dtype.Uint64;
      Dtype.Bool;
      Dtype.Complex64;
      Dtype.Complex128;
    ]
  in
  match List.find_opt (fun d -> Dtype.short_name d = name) all with
  | Some d -> d
  | None -> invalid_arg ("deserialize: unknown dtype " ^ name)

let aval_of_string s =
  match String.split_on_char ':' s with
  | [ dt; dims ] ->
      let shape =
        if dims = "" then [||]
        else
          Array.of_list (List.map int_of_string (String.split_on_char ',' dims))
      in
      { shape; dtype = dtype_of_name dt; weak_type = false }
  | _ -> invalid_arg "deserialize: malformed type"

let magic = "ojax-exported-v1"

let serialize e =
  String.concat "\n"
    [
      magic;
      String.concat ";" (List.map aval_to_string e.in_avals);
      String.concat ";" (List.map aval_to_string e.out_avals);
      String.concat ";" e.platforms;
      e.mlir_module;
    ]

let deserialize text =
  let split_once s =
    match String.index_opt s '\n' with
    | Some i ->
        (String.sub s 0 i, String.sub s (i + 1) (String.length s - i - 1))
    | None -> invalid_arg "deserialize: truncated module"
  in
  let header, rest = split_once text in
  if header <> magic then invalid_arg "deserialize: unknown format";
  let avals s =
    if s = "" then [] else List.map aval_of_string (String.split_on_char ';' s)
  in
  let inputs, rest = split_once rest in
  let outputs, rest = split_once rest in
  let platforms, mlir_module = split_once rest in
  {
    mlir_module;
    in_avals = avals inputs;
    out_avals = avals outputs;
    platforms = String.split_on_char ';' platforms;
  }

let call e args =
  let actual = List.map Core.get_aval args in
  List.iter2
    (fun (a : aval) (expected : aval) ->
      if a.shape <> expected.shape || a.dtype <> expected.dtype then
        invalid_arg "Exported function called with mismatched argument types")
    actual e.in_avals;
  Backend.run_text e.mlir_module args
