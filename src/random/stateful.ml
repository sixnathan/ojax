open Types
module Nd = Ndarray
module RC = Core

type t = { base_key : value; counter : int ref }

let stateful_rng ?(impl = "threefry2x32") seed =
  if impl <> "threefry2x32" then
    failwith
      "stateful: only the default threefry2x32 impl is wired into Random.Core \
       (M5)";
  let dt = if Config.x64_enabled () then Dtype.I64 else Dtype.I32 in
  let base = RC.key (Concrete (Nd.of_floats dt [||] [| float_of_int seed |])) in
  { base_key = base; counter = ref 0 }

let key ?(shape = [||]) t =
  let cnt =
    Concrete (Nd.of_floats Dtype.I32 [||] [| float_of_int !(t.counter) |])
  in
  let k = RC.fold_in t.base_key cnt in
  incr t.counter;
  if Array.length shape = 0 then k
  else if Array.length shape = 1 then RC.split k shape.(0)
  else failwith "stateful: multi-dimensional key shape not supported (M5)"

let random ?(size = [||]) t =
  RC.uniform (key t) ~shape:size ~minval:0. ~maxval:1.

let uniform ?(low = 0.) ?(high = 1.) ?(size = [||]) t =
  RC.uniform (key t) ~shape:size ~minval:low ~maxval:high

let normal ?(loc = 0.) ?(scale = 1.) ?(size = [||]) t =
  let n = RC.normal (key t) ~shape:size in
  if loc = 0. && scale = 1. then n
  else
    match n with
    | Concrete nd ->
        Concrete (Nd.map Dtype.F32 (fun x -> loc +. (scale *. x)) nd)
    | _ -> failwith "stateful: normal is concrete-only (M5)"

let integers ?high ?(size = [||]) ~low t =
  let minval, maxval =
    match high with None -> (0, low) | Some h -> (low, h)
  in
  RC.randint (key t) ~shape:size ~minval ~maxval

let split num t =
  let bk = key ~shape:[| num |] t in
  { base_key = bk; counter = ref 0 }

let base_key t = t.base_key

let spawn _ _ =
  failwith
    "stateful: spawn requires per-child slicing of the split key array \
     (deferred, M5); use split for the vmap-compatible form"
