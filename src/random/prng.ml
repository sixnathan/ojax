open Types
module Nd = Ndarray

let mask = 0xFFFF_FFFF
let prod a = Array.fold_left ( * ) 1 a

let unravel shape i =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let r = ref i in
  for k = n - 1 downto 0 do
    idx.(k) <- !r mod shape.(k);
    r := !r / shape.(k)
  done;
  idx

let geti nd i = Int64.to_int (Nd.get_i64 nd (unravel (Nd.shape nd) i))

let of_u32 shape ints =
  Nd.of_floats Dtype.Uint32 shape (Array.map float_of_int ints)

let broadcast_shapes shapes =
  let r = List.fold_left (fun acc s -> max acc (Array.length s)) 0 shapes in
  let out = Array.make r 1 in
  List.iter
    (fun s ->
      let d = Array.length s in
      Array.iteri
        (fun j sz ->
          let k = r - d + j in
          if sz <> 1 then
            if out.(k) = 1 then out.(k) <- sz
            else if out.(k) <> sz then
              invalid_arg "prng: incompatible broadcast shapes")
        s)
    shapes;
  out

let bcast_flat src_shape out_shape out_idx =
  let d = Array.length src_shape and r = Array.length out_shape in
  let flat = ref 0 in
  for j = 0 to d - 1 do
    let coord = if src_shape.(j) = 1 then 0 else out_idx.(r - d + j) in
    flat := (!flat * src_shape.(j)) + coord
  done;
  !flat

let iota_2x32_nd shape =
  let n = prod shape in
  let hi = Array.make n 0 and lo = Array.make n 0 in
  for i = 0 to n - 1 do
    lo.(i) <- i land mask;
    hi.(i) <- (i lsr 32) land mask
  done;
  (of_u32 shape hi, of_u32 shape lo)

type prng_impl = {
  key_shape : int array;
  seed : Nd.t -> Nd.t;
  split : Nd.t -> int array -> Nd.t;
  random_bits : Nd.t -> int -> int array -> Nd.t;
  fold_in : Nd.t -> Nd.t -> Nd.t;
  name : string;
  tag : string;
}

let registry : (string, prng_impl) Hashtbl.t = Hashtbl.create 4
let registered : prng_impl option ref = ref None

let register_prng impl =
  if Hashtbl.mem registry impl.name then
    invalid_arg ("prng: PRNG with name " ^ impl.name ^ " already registered");
  Hashtbl.replace registry impl.name impl;
  registered := Some impl

let current_impl () =
  match !registered with
  | Some i -> i
  | None -> failwith "prng: no PRNG implementation registered"

let seed_with_impl impl seed = impl.seed seed

let drop_last a =
  let n = Array.length a in
  if n = 0 then invalid_arg "prng: key array must have a trailing dimension";
  Array.sub a 0 (n - 1)

let subkey key b =
  let base = 2 * b in
  of_u32 [| 2 |] [| geti key base; geti key (base + 1) |]

let random_seed_impl seeds =
  let impl = current_impl () in
  let sshape = Nd.shape seeds in
  let n = prod sshape in
  let out = Array.make (n * 2) 0 in
  for j = 0 to n - 1 do
    let sc =
      Nd.of_floats (Nd.dtype seeds) [||] [| Nd.get_f seeds (unravel sshape j) |]
    in
    let k = impl.seed sc in
    out.(2 * j) <- geti k 0;
    out.((2 * j) + 1) <- geti k 1
  done;
  of_u32 (Array.append sshape [| 2 |]) out

let random_split_impl shape key =
  let impl = current_impl () in
  let b_shape = drop_last (Nd.shape key) in
  let prod_b = prod b_shape in
  let out_shape = Array.append (Array.append b_shape shape) [| 2 |] in
  let block = prod shape * 2 in
  let out = Array.make (prod_b * block) 0 in
  for b = 0 to prod_b - 1 do
    let sk = impl.split (subkey key b) shape in
    for t = 0 to block - 1 do
      out.((b * block) + t) <- geti sk t
    done
  done;
  of_u32 out_shape out

let random_fold_in_impl key msg =
  let impl = current_impl () in
  let bk = drop_last (Nd.shape key) in
  let bm = Nd.shape msg in
  let ob = broadcast_shapes [ bk; bm ] in
  let n = prod ob in
  let out = Array.make (n * 2) 0 in
  for j = 0 to n - 1 do
    let idx = unravel ob j in
    let kb = bcast_flat bk ob idx in
    let mb = bcast_flat bm ob idx in
    let msc =
      Nd.of_floats (Nd.dtype msg) [||] [| Nd.get_f msg (unravel bm mb) |]
    in
    let r = impl.fold_in (subkey key kb) msc in
    out.(2 * j) <- geti r 0;
    out.((2 * j) + 1) <- geti r 1
  done;
  of_u32 (Array.append ob [| 2 |]) out

let random_bits_impl bit_width shape key =
  let impl = current_impl () in
  let bk = drop_last (Nd.shape key) in
  let prod_b = prod bk in
  let out_shape = Array.append bk shape in
  let block = prod shape in
  let out = Array.make (prod_b * block) 0 in
  for b = 0 to prod_b - 1 do
    let rb = impl.random_bits (subkey key b) bit_width shape in
    for t = 0 to block - 1 do
      out.((b * block) + t) <- geti rb t
    done
  done;
  of_u32 out_shape out

let impl_rule prim inputs =
  match (prim, inputs) with
  | Iota_2x32_shape shape, [] ->
      let hi, lo = iota_2x32_nd shape in
      [ hi; lo ]
  | Random_seed, [ seeds ] -> [ random_seed_impl seeds ]
  | Random_split shape, [ key ] -> [ random_split_impl shape key ]
  | Random_fold_in, [ key; msg ] -> [ random_fold_in_impl key msg ]
  | Random_bits { bit_width; shape }, [ key ] ->
      [ random_bits_impl bit_width shape key ]
  | Random_wrap, [ arr ] -> [ arr ]
  | Random_unwrap, [ key ] -> [ key ]
  | _ -> failwith "prng: impl arity mismatch"

let shaped shape = { shape; dtype = Dtype.Uint32; weak_type = false }

let abstract_eval_rule prim avals =
  match (prim, avals) with
  | Iota_2x32_shape shape, [] -> [ shaped shape; shaped shape ]
  | Random_seed, [ a ] -> [ shaped (Array.append a.shape [| 2 |]) ]
  | Random_split shape, [ a ] ->
      [ shaped (Array.append (Array.append (drop_last a.shape) shape) [| 2 |]) ]
  | Random_fold_in, [ ka; ma ] ->
      let ob = broadcast_shapes [ drop_last ka.shape; ma.shape ] in
      [ shaped (Array.append ob [| 2 |]) ]
  | Random_bits { bit_width; shape }, [ a ] ->
      if bit_width <> 32 then
        failwith "prng: only 32-bit random_bits supported (M3)";
      [ shaped (Array.append (drop_last a.shape) shape) ]
  | Random_wrap, [ a ] -> [ shaped a.shape ]
  | Random_unwrap, [ a ] -> [ shaped a.shape ]
  | _ -> failwith "prng: abstract_eval arity mismatch"

let is_prng_prim = function
  | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
  | Random_bits _ | Random_wrap | Random_unwrap ->
      true
  | _ -> false

let prev_impl = ref (fun (_ : primitive) (_ : Nd.t list) -> ([] : Nd.t list))
let prev_aeval = ref (fun (_ : primitive) (_ : aval list) -> ([] : aval list))

let impl prim inputs =
  if is_prng_prim prim then impl_rule prim inputs else !prev_impl prim inputs

let abstract_eval prim avals =
  if is_prng_prim prim then abstract_eval_rule prim avals
  else !prev_aeval prim avals

let install () =
  Lax.install ();
  prev_impl := Core.rules.impl;
  prev_aeval := Core.rules.abstract_eval;
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()

let iota_2x32_shape shape =
  if Array.length shape = 0 then
    let z () = Concrete (of_u32 [||] [| 0 |]) in
    (z (), z ())
  else
    match Core.bind (Iota_2x32_shape shape) [] with
    | [ hi; lo ] -> (hi, lo)
    | _ -> failwith "prng: iota_2x32_shape arity"

let random_seed seed = Core.bind1 Random_seed [ seed ]
let random_split key shape = Core.bind1 (Random_split shape) [ key ]
let random_fold_in key msg = Core.bind1 Random_fold_in [ key; msg ]

let random_bits key ~bit_width ~shape =
  Core.bind1 (Random_bits { bit_width; shape }) [ key ]

let random_wrap arr = Core.bind1 Random_wrap [ arr ]
let random_unwrap key = Core.bind1 Random_unwrap [ key ]
