open Typedtree
module SS = Set.Make (String)

let uname id = Ident.unique_name id

let rec arrow_labels ty =
  match Types.get_desc ty with
  | Types.Tarrow (l, _, r, _) -> l :: arrow_labels r
  | Types.Tpoly (t, _) -> arrow_labels t
  | _ -> []

let is_arrow ty =
  match Types.get_desc ty with
  | Types.Tarrow _ -> true
  | Types.Tpoly (t, _) -> (
      match Types.get_desc t with Types.Tarrow _ -> true | _ -> false)
  | _ -> false

let has_optional name ty =
  List.exists
    (function Asttypes.Optional n -> n = name | _ -> false)
    (arrow_labels ty)

let last_name path =
  match path with
  | Path.Pident id -> Ident.name id
  | Path.Pdot (_, s) -> s
  | _ -> Path.last path

let wrappers : (string, unit) Hashtbl.t = Hashtbl.create 16

let is_target path (vd : Types.value_description) =
  let n = last_name path in
  let ty = vd.val_type in
  (List.mem n [ "jit"; "jit1"; "named_jit" ] && has_optional "key" ty)
  || n = "jitted_keyed"
  || (List.mem n [ "one"; "many"; "pair" ] && has_optional "jit" ty)
  || Hashtbl.mem wrappers n

let used_idents expr =
  let acc = ref [] in
  let it =
    {
      Tast_iterator.default_iterator with
      expr =
        (fun self e ->
          (match e.exp_desc with
          | Texp_ident (Path.Pident id, _, vd) -> acc := (id, vd) :: !acc
          | _ -> ());
          Tast_iterator.default_iterator.expr self e);
    }
  in
  it.expr it expr;
  !acc

let bound_idents expr =
  let acc = ref SS.empty in
  let add id = acc := SS.add (uname id) !acc in
  let it =
    {
      Tast_iterator.default_iterator with
      pat =
        (fun self p ->
          List.iter add (Typedtree.pat_bound_idents p);
          Tast_iterator.default_iterator.pat self p);
      expr =
        (fun self e ->
          (match e.exp_desc with
          | Texp_function (params, body) -> (
              List.iter (fun fp -> add fp.fp_param) params;
              match body with
              | Tfunction_cases { param; _ } -> add param
              | Tfunction_body _ -> ())
          | Texp_for (id, _, _, _, _, _) -> add id
          | Texp_letop { param; _ } -> add param
          | _ -> ());
          Tast_iterator.default_iterator.expr self e);
    }
  in
  it.expr it expr;
  !acc

let toplevel = ref SS.empty
let bindings : (string, expression) Hashtbl.t = Hashtbl.create 4096

let collect_toplevel str =
  let it =
    {
      Tast_iterator.default_iterator with
      structure_item =
        (fun self si ->
          (match si.str_desc with
          | Tstr_value (_, vbs) ->
              List.iter
                (fun vb ->
                  List.iter
                    (fun id -> toplevel := SS.add (uname id) !toplevel)
                    (Typedtree.pat_bound_idents vb.vb_pat))
                vbs
          | _ -> ());
          Tast_iterator.default_iterator.structure_item self si);
      value_binding =
        (fun self vb ->
          (match vb.vb_pat.pat_desc with
          | Tpat_var (id, _, _) ->
              Hashtbl.replace bindings (uname id) vb.vb_expr
          | _ -> ());
          Tast_iterator.default_iterator.value_binding self vb);
    }
  in
  it.structure it str

let free_locals expr =
  let bound = bound_idents expr in
  let seen = Hashtbl.create 8 in
  List.filter_map
    (fun (id, vd) ->
      let u = uname id in
      if
        SS.mem u bound || SS.mem u !toplevel || Hashtbl.mem seen u
        || Ident.global id
      then None
      else (
        Hashtbl.replace seen u ();
        Some (id, vd)))
    (List.rev (used_idents expr))

let type_string ty =
  Format.asprintf "%a" Printtyp.type_expr ty
  |> String.map (function '\n' -> ' ' | c -> c)

let is_array_type ty =
  let s = type_string ty in
  List.exists
    (fun t -> s = t || s = t ^ " list" || s = t ^ " option")
    [
      "value"; "Types.value"; "T.value"; "Ojax.Types.value"; "Ojax__Types.value";
    ]

let rec derived covered depth id =
  depth < 8
  &&
  match Hashtbl.find_opt bindings (uname id) with
  | None -> false
  | Some e ->
      List.for_all
        (fun (d, _) ->
          SS.mem (uname d) covered || derived covered (depth + 1) d)
        (free_locals e)

let findings = ref []
let new_wrappers = ref []

let analyse file enclosing =
  {
    Tast_iterator.default_iterator with
    expr =
      (fun self e ->
        (match e.exp_desc with
        | Texp_apply ({ exp_desc = Texp_ident (path, _, vd); _ }, args)
          when is_target path vd ->
            let present =
              List.filter_map
                (fun (l, a) ->
                  match a with Arg x -> Some (l, x) | Omitted () -> None)
                args
            in
            let fns, others =
              List.partition
                (fun (l, (x : expression)) ->
                  l = Asttypes.Nolabel && is_arrow x.exp_type)
                present
            in
            let rec cover depth s id =
              let u = uname id in
              if SS.mem u s then s
              else
                let s = SS.add u s in
                match Hashtbl.find_opt bindings u with
                | Some b when depth < 8 ->
                    List.fold_left
                      (fun s (d, _) -> cover (depth + 1) s d)
                      s (free_locals b)
                | _ -> s
            in
            let covered =
              List.fold_left
                (fun s (_, x) ->
                  List.fold_left
                    (fun s (id, _) -> cover 0 s id)
                    s (used_idents x))
                SS.empty others
            in
            List.iter
              (fun (_, (f : expression)) ->
                match f.exp_desc with
                | Texp_ident (Path.Pident id, _, _)
                  when not (SS.mem (uname id) !toplevel) -> (
                    match !enclosing with
                    | Some name when not (Hashtbl.mem wrappers name) ->
                        new_wrappers := name :: !new_wrappers
                    | _ -> ())
                | _ ->
                    List.iter
                      (fun (id, (vd : Types.value_description)) ->
                        (if
                           is_arrow vd.val_type
                           && (not (Hashtbl.mem bindings (uname id)))
                           && not (SS.mem (uname id) covered)
                         then
                           match !enclosing with
                           | Some name when not (Hashtbl.mem wrappers name) ->
                               new_wrappers := name :: !new_wrappers
                           | _ -> ());
                        if not (SS.mem (uname id) covered) then
                          let kind =
                            if is_array_type vd.val_type then "array"
                            else if is_arrow vd.val_type then "closure"
                            else if derived covered 0 id then "derived"
                            else "static"
                          in
                          let pos = e.exp_loc.loc_start in
                          findings :=
                            ( file,
                              pos.pos_lnum,
                              Option.value !enclosing ~default:"?",
                              last_name path,
                              kind,
                              Ident.name id,
                              type_string vd.val_type )
                            :: !findings)
                      (free_locals f))
              fns
        | _ -> ());
        Tast_iterator.default_iterator.expr self e);
    structure_item =
      (fun self si ->
        match si.str_desc with
        | Tstr_value (_, vbs) ->
            List.iter
              (fun vb ->
                let saved = !enclosing in
                (match vb.vb_pat.pat_desc with
                | Tpat_var (id, _, _) -> enclosing := Some (Ident.name id)
                | _ -> ());
                self.value_binding self vb;
                enclosing := saved)
              vbs
        | _ -> Tast_iterator.default_iterator.structure_item self si);
  }

let () =
  let dir = Sys.argv.(1) in
  let files =
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".cmt")
    |> List.sort compare
  in
  let loaded =
    List.filter_map
      (fun f ->
        let cmt = Cmt_format.read_cmt (Filename.concat dir f) in
        match cmt.cmt_annots with
        | Implementation str ->
            let src = Option.value cmt.cmt_sourcefile ~default:f in
            Some (src, str)
        | _ -> None)
      files
  in
  List.iter (fun (_, str) -> collect_toplevel str) loaded;
  let rec loop round =
    findings := [];
    new_wrappers := [];
    List.iter
      (fun (src, str) ->
        let it = analyse src (ref None) in
        it.structure it str)
      loaded;
    if !new_wrappers <> [] && round < 6 then (
      List.iter (fun w -> Hashtbl.replace wrappers w ()) !new_wrappers;
      loop (round + 1))
  in
  loop 0;
  Hashtbl.iter (fun w () -> Printf.printf "wrapper\t%s\n" w) wrappers;
  List.iter
    (fun (file, line, encl, callee, kind, name, ty) ->
      Printf.printf "%s:%d\t%s\t%s\t%s\t%s\t%s\n" file line encl callee kind
        name ty)
    (List.sort compare !findings)
