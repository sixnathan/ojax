open Parsetree
open Ast_helper

let counter = ref 0

let fresh () =
  incr counter;
  Printf.sprintf "ltr__%d" !counter

let rec pure e =
  match e.pexp_desc with
  | Pexp_ident _ | Pexp_constant _ | Pexp_function _ | Pexp_lazy _ | Pexp_pack _
    ->
      true
  | Pexp_construct (_, None) | Pexp_variant (_, None) -> true
  | Pexp_construct (_, Some a) | Pexp_variant (_, Some a) -> pure a
  | Pexp_tuple items -> List.for_all (fun (_, x) -> pure x) items
  | Pexp_array items -> List.for_all pure items
  | Pexp_record (fields, base) ->
      List.for_all (fun (_, x) -> pure x) fields
      && Option.fold ~none:true ~some:pure base
  | Pexp_constraint (x, _) | Pexp_coerce (x, _, _) | Pexp_field (x, _) -> pure x
  | _ -> false

let rec impure_leaves e =
  match e.pexp_desc with
  | Pexp_construct (_, Some a) | Pexp_variant (_, Some a) -> impure_leaves a
  | Pexp_tuple items ->
      List.fold_left (fun n (_, x) -> n + impure_leaves x) 0 items
  | Pexp_array items -> List.fold_left (fun n x -> n + impure_leaves x) 0 items
  | Pexp_record (fields, base) ->
      List.fold_left
        (fun n (_, x) -> n + impure_leaves x)
        (Option.fold ~none:0 ~some:impure_leaves base)
        fields
  | Pexp_constraint (x, _) -> impure_leaves x
  | _ -> if pure e then 0 else 1

let short_circuit fn =
  match fn.pexp_desc with
  | Pexp_ident { txt = Longident.Lident ("&&" | "||" | "&" | "or"); _ } -> true
  | _ -> false

let rec spine mapper binds remaining e =
  let recur = spine mapper binds remaining in
  match e.pexp_desc with
  | Pexp_construct (c, Some a) ->
      { e with pexp_desc = Pexp_construct (c, Some (recur a)) }
  | Pexp_variant (c, Some a) ->
      { e with pexp_desc = Pexp_variant (c, Some (recur a)) }
  | Pexp_tuple items ->
      {
        e with
        pexp_desc = Pexp_tuple (List.map (fun (l, x) -> (l, recur x)) items);
      }
  | Pexp_array items -> { e with pexp_desc = Pexp_array (List.map recur items) }
  | Pexp_record (fields, base) ->
      let base = Option.map recur base in
      let fields = List.map (fun (l, x) -> (l, recur x)) fields in
      { e with pexp_desc = Pexp_record (fields, base) }
  | Pexp_constraint (x, t) ->
      { e with pexp_desc = Pexp_constraint (recur x, t) }
  | _ when pure e -> mapper.Ast_mapper.expr mapper e
  | _ ->
      let e' = mapper.Ast_mapper.expr mapper e in
      decr remaining;
      if !remaining = 0 then e'
      else
        let name = fresh () in
        let loc = { e.pexp_loc with loc_ghost = true } in
        binds := (name, loc, e') :: !binds;
        Exp.ident ~loc { txt = Longident.Lident name; loc }

let wrap binds body =
  List.fold_left
    (fun acc (name, loc, value) ->
      Exp.let_ ~loc Asttypes.Nonrecursive
        [ Vb.mk ~loc (Pat.var ~loc { txt = name; loc }) value ]
        acc)
    body binds

let aggregate mapper e =
  let total = impure_leaves e in
  if total < 2 then None
  else
    let binds = ref [] in
    let remaining = ref total in
    let body = spine mapper binds remaining e in
    Some (wrap !binds body)

let expr mapper e =
  match e.pexp_desc with
  | Pexp_apply (fn, args) when not (short_circuit fn) ->
      let total =
        List.fold_left
          (fun n (_, x) -> n + impure_leaves x)
          (impure_leaves fn) args
      in
      if total < 2 then Ast_mapper.default_mapper.expr mapper e
      else
        let binds = ref [] in
        let remaining = ref total in
        let fn = spine mapper binds remaining fn in
        let args =
          List.map (fun (l, x) -> (l, spine mapper binds remaining x)) args
        in
        wrap !binds { e with pexp_desc = Pexp_apply (fn, args) }
  | Pexp_tuple _
  | Pexp_construct (_, Some _)
  | Pexp_variant (_, Some _)
  | Pexp_array _ | Pexp_record _ -> (
      match aggregate mapper e with
      | Some rewritten -> rewritten
      | None -> Ast_mapper.default_mapper.expr mapper e)
  | _ -> Ast_mapper.default_mapper.expr mapper e

let mapper = { Ast_mapper.default_mapper with expr }

let copy path =
  let ic = open_in_bin path in
  set_binary_mode_out stdout true;
  output_string stdout (really_input_string ic (in_channel_length ic));
  close_in ic

let implementation path =
  let ic = open_in_bin path in
  let lexbuf = Lexing.from_channel ic in
  Location.init lexbuf path;
  Location.input_name := path;
  let ast = Parse.implementation lexbuf in
  close_in ic;
  let ast = mapper.Ast_mapper.structure mapper ast in
  set_binary_mode_out stdout true;
  output_string stdout Config.ast_impl_magic_number;
  output_value stdout path;
  output_value stdout ast

let () =
  let path = Sys.argv.(1) in
  if Filename.check_suffix path ".ml" then implementation path else copy path
