type 'a flag = { name : string; mutable value : 'a }

let make name value = { name; value }
let value flag = flag.value
let name flag = flag.name
let set flag v = flag.value <- v

let with_value flag v thunk =
  let prev = flag.value in
  flag.value <- v;
  Fun.protect ~finally:(fun () -> flag.value <- prev) thunk

let enable_x64 = make "jax_enable_x64" false
let x64_enabled () = enable_x64.value
