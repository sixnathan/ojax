module Api = Ojax.Api
module Core = Ojax.Core
module T = Ojax.Types
module Tree = Ojax.Tree_util
module Dt = Ojax.Dtype
module Nd = Ojax.Ndarray
module D = Ojax.Pjrt.Discover

let () = Ojax.Lax.install ()

let plugin_present =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> true
  | _ -> false

let () = if plugin_present then Unix.putenv "OJAX_BACKEND" "xla"

let dot x y =
  Core.bind1
    (T.Dot_general
       {
         lhs_contract = [| 1 |];
         rhs_contract = [| 0 |];
         lhs_batch = [||];
         rhs_batch = [||];
       })
    [ x; y ]

let chain (args : T.value Tree.t list) : T.value Tree.t =
  match args with
  | [ Tree.Leaf x ] ->
      let y = dot x x in
      let y = dot y x in
      let y = dot y x in
      Tree.Leaf y
  | _ -> assert false

let x0 =
  let n = 16 in
  Nd.canonicalize Dt.F32
    (Nd.of_floats Dt.F32 [| 4; 4 |]
       (Array.init n (fun i -> (float_of_int i *. 0.1) +. 0.3)))

let time_loop name iters f =
  let t0 = Unix.gettimeofday () in
  for i = 1 to iters do
    f ();
    if i mod 200 = 0 then Gc.full_major ()
  done;
  let t1 = Unix.gettimeofday () in
  let us = (t1 -. t0) /. float_of_int iters *. 1e6 in
  Printf.printf "%-24s %8.3f us/call (%d iters)\n%!" name us iters

let () =
  if not plugin_present then
    print_endline "matmul_chain: OJAX_PJRT_PLUGIN not set, skipping benchmark"
  else begin
    let jf = Api.jit chain in
    let iters =
      match Sys.getenv_opt "OJAX_BENCH_ITERS" with
      | Some s -> int_of_string s
      | None -> 1000
    in
    let host_arg = Tree.Leaf (T.Concrete x0) in
    ignore (jf [ host_arg ]);
    time_loop "host-in/host-out" iters (fun () -> ignore (jf [ host_arg ]));
    let dev_arg = Api.device_put (Tree.Leaf (T.Concrete x0)) in
    ignore (jf [ dev_arg ]);
    time_loop "device-resident" iters (fun () -> ignore (jf [ dev_arg ]))
  end
