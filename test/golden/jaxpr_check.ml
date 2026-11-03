module U = Yojson.Safe.Util
module T = Ojax.Types

let str row k = U.to_string (U.member k row)
let aval shape dtype : T.aval = { shape; dtype; weak_type = false }

let var shape dtype : T.var =
  { vid = Ojax.Core.fresh_id (); vaval = aval shape dtype }

let jaxpr_of f =
  (Ojax.Jaxpr.make_jaxpr
     [ aval [| 3 |] Ojax.Dtype.F32 ]
     (function [ x ] -> [ f x ] | _ -> failwith "jaxpr_check: arity"))
    .T.jaxpr

let small () =
  jaxpr_of (fun x ->
      Ojax.Numpy.Ufuncs.multiply (Ojax.Numpy.Ufuncs.sin x)
        (Docs_p2.py_float 2.0))

let long () =
  jaxpr_of (fun x ->
      let rec go k x =
        if k = 0 then x else go (k - 1) (Ojax.Numpy.Ufuncs.sin x)
      in
      go 30 x)

let replace_nth l k f = List.mapi (fun i x -> if i = k then f x else x) l

let case name : T.jaxpr =
  let f32 = Ojax.Dtype.F32 in
  match name with
  | "good" -> small ()
  | "out_type" -> (
      let j = small () in
      match j.eqns with
      | [ e0; e1 ] ->
          let bad = var [| 3 |] Ojax.Dtype.I32 in
          {
            j with
            eqns =
              [
                { e0 with outs = [ bad ] };
                { e1 with inputs = T.A_var bad :: List.tl e1.inputs };
              ];
          }
      | _ -> failwith "jaxpr_check: small")
  | "out_shape" ->
      let j = small () in
      {
        j with
        eqns =
          replace_nth j.eqns 0 (fun (e : T.eqn) ->
              { e with outs = [ var [| 2 |] f32 ] });
      }
  | "undefined" ->
      let j = small () in
      { j with eqns = List.tl j.eqns }
  | "rebound" ->
      let j = small () in
      { j with eqns = List.hd j.eqns :: j.eqns }
  | "rebound_input" ->
      let j = small () in
      { j with in_binders = j.in_binders @ j.in_binders }
  | "undefined_out" ->
      let j = small () in
      { j with outs = [ T.A_var (var [| 3 |] f32) ] }
  | "long_undefined_25" ->
      let j = long () in
      { j with eqns = List.filteri (fun i _ -> i <> 20) j.eqns }
  | long_case -> (
      match String.split_on_char '_' long_case with
      | [ "long"; k ] ->
          let j = long () in
          {
            j with
            eqns =
              replace_nth j.eqns (int_of_string k) (fun (e : T.eqn) ->
                  { e with outs = [ var [| 3 |] Ojax.Dtype.F64 ] });
          }
      | _ -> failwith ("jaxpr_check: unknown case " ^ long_case))

let check_row row () =
  let id = str row "id" in
  let expected = Docs_p2.expected_of row in
  let actual =
    Docs_p2.run_case (fun () ->
        Ojax.Jaxpr.typecheck_jaxpr (case id);
        "")
  in
  let suite = "jaxpr_check:all" in
  let ok = Docs_p2.same expected actual in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "JAXPR_CHECK DIFFER %s\n  jax:  %s\n  ojax: %s\n%!" id
      (Docs_p2.show expected) (Docs_p2.show actual)

let suite () =
  let path =
    Filename.concat
      (Filename.concat Docs_p2.goldens_root "jaxpr_check")
      "rows.json"
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "jaxpr_check",
    List.map
      (fun row -> Alcotest.test_case (str row "id") `Quick (check_row row))
      rows )
