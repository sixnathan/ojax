module U = Yojson.Safe.Util
module SP = Ojax.Shape_poly

let name = "shape_poly_exprs"
let suite = "shape_poly_exprs:x64_off"

type v = Dim of SP.dim | Bool of bool

let dim = function Dim d -> d | Bool _ -> failwith "shape_poly_exprs: bool"

let rec ev dims e =
  let l = U.to_list e in
  let arg i = ev dims (List.nth l i) in
  let bin f = Dim (f (dim (arg 1)) (dim (arg 2))) in
  let cmp f = Bool (f (dim (arg 1)) (dim (arg 2))) in
  match U.to_string (List.hd l) with
  | "v" -> Dim dims.(U.to_int (List.nth l 1))
  | "c" -> Dim (SP.of_int (U.to_int (List.nth l 1)))
  | "neg" -> Dim (SP.neg (dim (arg 1)))
  | "pow2" -> Dim (SP.pow (dim (arg 1)) 2)
  | "pow3" -> Dim (SP.pow (dim (arg 1)) 3)
  | "add" -> bin SP.add
  | "sub" -> bin SP.sub
  | "mul" -> bin SP.mul
  | "floordiv" -> bin SP.floordiv_dim
  | "mod" -> bin SP.mod_dim
  | "max" -> bin SP.max_dim
  | "min" -> bin SP.min_dim
  | "eq" -> cmp SP.equal
  | "ne" -> cmp (fun a b -> not (SP.equal a b))
  | "ge" -> cmp SP.ge
  | "gt" -> cmp SP.gt
  | "le" -> cmp SP.le
  | "lt" -> cmp SP.lt
  | op -> failwith ("shape_poly_exprs: op " ^ op)

let expected row =
  let s k = U.to_string (U.member k row) in
  match s "kind" with
  | "raise" -> Printf.sprintf "raise %s %s" (s "class") (s "message")
  | kind -> kind ^ " " ^ s "text"

let actual row =
  let constraints =
    List.map U.to_string (U.to_list (U.member "constraints" row))
  in
  match
    let dims =
      SP.symbolic_shape ~constraints (Some (U.to_string (U.member "spec" row)))
    in
    ev dims (U.member "expr" row)
  with
  | Dim d -> "dim " ^ SP.dim_to_string d
  | Bool b -> "bool " ^ if b then "True" else "False"
  | exception SP.Inconclusive_dimension_operation m ->
      "raise InconclusiveDimensionOperation " ^ m
  | exception e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some m -> Printf.sprintf "raise %s %s" cls m
      | _ -> "raise " ^ Printexc.to_string e)

let check row () =
  let id = U.to_string (U.member "id" row) in
  let e = expected row and a = actual row in
  let ok = e = a in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "SHAPE_POLY_EXPRS DIFFER %s\n  jax:  %s\n  ojax: %s\n%!" id e
      a

let suites ~goldens_root =
  let path = Filename.concat (Filename.concat goldens_root name) "rows.json" in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  [
    ( "shape_poly_exprs_x64_off",
      List.map
        (fun row ->
          Alcotest.test_case
            (U.to_string (U.member "id" row))
            `Quick (check row))
        rows );
  ]
