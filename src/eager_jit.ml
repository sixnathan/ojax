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

let static_key statics opts =
  Digest.to_hex
    (Digest.string
       (Marshal.to_string
          (statics, List.map Option.is_some opts)
          [ Marshal.Closures ]))

let many ?jit ?(vals_first = false) name statics opts vals f =
  let present = List.filter_map Fun.id opts in
  let split flat =
    if vals_first then
      let n = List.length vals in
      let front = List.filteri (fun i _ -> i < n) flat in
      let back = List.filteri (fun i _ -> i >= n) flat in
      rebuild opts n (back @ front)
    else rebuild opts (List.length vals) flat
  in
  let args = if vals_first then vals @ present else present @ vals in
  match jit with
  | Some jname when Core.top_is_staging args ->
      Jaxpr.named_jit ~name:jname ~key:(static_key statics opts)
        (fun flat ->
          let opts, vals = split flat in
          f opts vals)
        args
  | _ ->
      Backend.jitted_keyed
        (name, Obj.repr (statics, List.map Option.is_some opts))
        (fun flat ->
          let opts, vals = split flat in
          f opts vals)
        args

let one ?jit ?vals_first name statics opts vals f =
  match many ?jit ?vals_first name statics opts vals (fun o v -> [ f o v ]) with
  | [ out ] -> out
  | _ -> failwith "Eager_jit: expected one result"

let pair ?jit ?vals_first name statics opts vals f =
  match
    many ?jit ?vals_first name statics opts vals (fun o v ->
        let a, b = f o v in
        [ a; b ])
  with
  | [ a; b ] -> (a, b)
  | _ -> failwith "Eager_jit: expected two results"
