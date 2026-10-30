let initialized = ref false

let initialize ?coordinator_address ?(num_processes = 1) ?(process_id = 0) () =
  ignore coordinator_address;
  if num_processes <> 1 || process_id <> 0 then
    invalid_arg
      "ojax runs a single process; multi-host setups are not supported";
  if !initialized then
    invalid_arg "jax.distributed.initialize should only be called once.";
  initialized := true

let is_initialized () = !initialized
let shutdown () = initialized := false
