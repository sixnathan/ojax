open Types

let rebuild opts vals_count flat =
  let rec take_opts opts flat =
    match (opts, flat) with
    | [], rest -> ([], rest)
    | None :: tl, rest ->
        let built, rest = take_opts tl rest in
        (None :: built, rest)
    | Some _ :: tl, v :: rest ->
        let built, rest = take_opts tl rest in
        (Some v :: built, rest)
    | Some _ :: _, [] -> invalid_arg "Eager_jit: argument underflow"
  in
  let built, rest = take_opts opts flat in
  if List.length rest <> vals_count then invalid_arg "Eager_jit: argument count";
  (built, rest)

let many name statics opts vals f =
  let present = List.filter_map Fun.id opts in
  Backend.jitted_keyed
    (name, Obj.repr (statics, List.map Option.is_some opts))
    (fun flat ->
      let opts, vals = rebuild opts (List.length vals) flat in
      f opts vals)
    (present @ vals)

let one name statics opts vals f =
  match many name statics opts vals (fun o v -> [ f o v ]) with
  | [ out ] -> out
  | _ -> failwith "Eager_jit: expected one result"

let pair name statics opts vals f =
  match
    many name statics opts vals (fun o v ->
        let a, b = f o v in
        [ a; b ])
  with
  | [ a; b ] -> (a, b)
  | _ -> failwith "Eager_jit: expected two results"
