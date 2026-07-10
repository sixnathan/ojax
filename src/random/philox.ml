open Types
module Nd = Ndarray
module Core = Ojax__Core

let mask = 0xFFFF_FFFF
let add32 a b = (a + b) land mask
let xor32 a b = a lxor b land mask
let philox_m2x32_0 = 0xD256_D193
let philox_m4x32_0 = 0xD251_1F53
let philox_m4x32_1 = 0xCD9E_8D57
let philox_w32_0 = 0x9E37_79B9
let philox_w32_1 = 0xBB67_AE85
let default_rounds = 10

let mulhilo m x =
  let full = Int64.mul (Int64.of_int m) (Int64.of_int x) in
  let lo = Int64.to_int (Int64.logand full 0xFFFF_FFFFL) in
  let hi =
    Int64.to_int (Int64.logand (Int64.shift_right_logical full 32) 0xFFFF_FFFFL)
  in
  (lo, hi)

let philox2x32_words k0 x0 x1 =
  let k0 = ref k0 and x0 = ref x0 and x1 = ref x1 in
  for rnd = 0 to default_rounds - 1 do
    if rnd > 0 then k0 := add32 !k0 philox_w32_0;
    let lo, hi = mulhilo philox_m2x32_0 !x0 in
    let nx0 = xor32 (xor32 hi !x1) !k0 in
    x0 := nx0;
    x1 := lo
  done;
  (!x0, !x1)

let philox4x32_words k0 k1 x0 x1 x2 x3 =
  let k0 = ref k0 and k1 = ref k1 in
  let x0 = ref x0 and x1 = ref x1 and x2 = ref x2 and x3 = ref x3 in
  for rnd = 0 to default_rounds - 1 do
    if rnd > 0 then (
      k0 := add32 !k0 philox_w32_0;
      k1 := add32 !k1 philox_w32_1);
    let lo0, hi0 = mulhilo philox_m4x32_0 !x0 in
    let lo1, hi1 = mulhilo philox_m4x32_1 !x2 in
    let nx0 = xor32 (xor32 hi1 !x1) !k0 in
    let nx1 = lo1 in
    let nx2 = xor32 (xor32 hi0 !x3) !k1 in
    let nx3 = lo0 in
    x0 := nx0;
    x1 := nx1;
    x2 := nx2;
    x3 := nx3
  done;
  (!x0, !x1, !x2, !x3)

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
              invalid_arg "philox: incompatible broadcast shapes")
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

let philox2x32_nd k0 x0 x1 =
  let os = broadcast_shapes [ Nd.shape k0; Nd.shape x0; Nd.shape x1 ] in
  let n = prod os in
  let o0 = Array.make n 0 and o1 = Array.make n 0 in
  for i = 0 to n - 1 do
    let idx = unravel os i in
    let g nd = geti nd (bcast_flat (Nd.shape nd) os idx) in
    let a, b = philox2x32_words (g k0) (g x0) (g x1) in
    o0.(i) <- a;
    o1.(i) <- b
  done;
  (of_u32 os o0, of_u32 os o1)

let philox4x32_nd k0 k1 x0 x1 x2 x3 =
  let os =
    broadcast_shapes
      [
        Nd.shape k0;
        Nd.shape k1;
        Nd.shape x0;
        Nd.shape x1;
        Nd.shape x2;
        Nd.shape x3;
      ]
  in
  let n = prod os in
  let o0 = Array.make n 0 and o1 = Array.make n 0 in
  let o2 = Array.make n 0 and o3 = Array.make n 0 in
  for i = 0 to n - 1 do
    let idx = unravel os i in
    let g nd = geti nd (bcast_flat (Nd.shape nd) os idx) in
    let a, b, c, d =
      philox4x32_words (g k0) (g k1) (g x0) (g x1) (g x2) (g x3)
    in
    o0.(i) <- a;
    o1.(i) <- b;
    o2.(i) <- c;
    o3.(i) <- d
  done;
  (of_u32 os o0, of_u32 os o1, of_u32 os o2, of_u32 os o3)

let seed_halves seed =
  let v = Nd.get_i64 seed [||] in
  let lo = Int64.to_int (Int64.logand v 0xFFFF_FFFFL) in
  let hi =
    match Nd.dtype seed with
    | Dtype.I64 ->
        Int64.to_int
          (Int64.logand (Int64.shift_right_logical v 32) 0xFFFF_FFFFL)
    | _ -> 0
  in
  (hi, lo)

let iota_2x32 shape =
  let n = prod shape in
  let hi = Array.make n 0 and lo = Array.make n 0 in
  for i = 0 to n - 1 do
    lo.(i) <- i land mask;
    hi.(i) <- (i lsr 32) land mask
  done;
  (of_u32 shape hi, of_u32 shape lo)

let philox2x32_seed_nd seed =
  let hi, lo = seed_halves seed in
  let o0, _ = philox2x32_words 0 hi lo in
  of_u32 [| 1 |] [| o0 |]

let philox2x32_split_nd key shape =
  let key0 = geti key 0 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    let a, _ = philox2x32_words key0 (geti hi i) (geti lo i) in
    out.(i) <- a
  done;
  of_u32 (Array.append shape [| 1 |]) out

let philox2x32_bits_nd key bit_width shape =
  if bit_width <> 32 then
    failwith "philox2x32: only 32-bit random_bits supported (M5)";
  let key0 = geti key 0 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    let a, b = philox2x32_words key0 (geti hi i) (geti lo i) in
    out.(i) <- xor32 a b
  done;
  of_u32 shape out

let philox2x32_fold_in_nd key data =
  let d = Int64.to_int (Int64.logand (Nd.get_i64 data [||]) 0xFFFF_FFFFL) in
  let o0, _ = philox2x32_words (geti key 0) 0 d in
  of_u32 [| 1 |] [| o0 |]

let philox4x32_seed_nd seed =
  let hi, lo = seed_halves seed in
  let o0, o1, _, _ = philox4x32_words 0 0 hi lo 0 0 in
  of_u32 [| 2 |] [| o0; o1 |]

let philox4x32_split_nd key shape =
  let k0 = geti key 0 and k1 = geti key 1 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make (n * 2) 0 in
  for i = 0 to n - 1 do
    let a, b, _, _ = philox4x32_words k0 k1 0 0 (geti hi i) (geti lo i) in
    out.(2 * i) <- a;
    out.((2 * i) + 1) <- b
  done;
  of_u32 (Array.append shape [| 2 |]) out

let philox4x32_bits_nd key bit_width shape =
  if bit_width <> 32 then
    failwith "philox4x32: only 32-bit random_bits supported (M5)";
  let k0 = geti key 0 and k1 = geti key 1 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    let a, b, c, d = philox4x32_words k0 k1 (geti hi i) (geti lo i) 0 0 in
    out.(i) <- xor32 (xor32 (xor32 a b) c) d
  done;
  of_u32 shape out

let philox4x32_fold_in_nd key data =
  let d = Int64.to_int (Int64.logand (Nd.get_i64 data [||]) 0xFFFF_FFFFL) in
  let o0, o1, _, _ = philox4x32_words (geti key 0) (geti key 1) 0 0 0 d in
  of_u32 [| 2 |] [| o0; o1 |]

let shaped shape = { shape; dtype = Dtype.Uint32; weak_type = false }

let philox2x32_impl inputs =
  match inputs with
  | [ k0; x0; x1 ] ->
      let a, b = philox2x32_nd k0 x0 x1 in
      [ a; b ]
  | _ -> failwith "philox: philox2x32 expects 3 operands"

let philox4x32_impl inputs =
  match inputs with
  | [ k0; k1; x0; x1; x2; x3 ] ->
      let a, b, c, d = philox4x32_nd k0 k1 x0 x1 x2 x3 in
      [ a; b; c; d ]
  | _ -> failwith "philox: philox4x32 expects 6 operands"

let philox2x32_aeval avals =
  match avals with
  | [ a; b; c ] ->
      let os = broadcast_shapes [ a.shape; b.shape; c.shape ] in
      [ shaped os; shaped os ]
  | _ -> failwith "philox: philox2x32 expects 3 avals"

let philox4x32_aeval avals =
  match avals with
  | [ a; b; c; d; e; f ] ->
      let os =
        broadcast_shapes
          [ a.shape; b.shape; c.shape; d.shape; e.shape; f.shape ]
      in
      [ shaped os; shaped os; shaped os; shaped os ]
  | _ -> failwith "philox: philox4x32 expects 6 avals"

let prev_impl = ref (fun (_ : primitive) (_ : Nd.t list) -> ([] : Nd.t list))
let prev_aeval = ref (fun (_ : primitive) (_ : aval list) -> ([] : aval list))

let impl prim inputs =
  match prim with
  | Philox2x32 -> philox2x32_impl inputs
  | Philox4x32 -> philox4x32_impl inputs
  | _ -> !prev_impl prim inputs

let abstract_eval prim avals =
  match prim with
  | Philox2x32 -> philox2x32_aeval avals
  | Philox4x32 -> philox4x32_aeval avals
  | _ -> !prev_aeval prim avals

let install () =
  prev_impl := Core.rules.impl;
  prev_aeval := Core.rules.abstract_eval;
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()

let philox2x32_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 1 |];
    seed = philox2x32_seed_nd;
    split = philox2x32_split_nd;
    random_bits = philox2x32_bits_nd;
    fold_in = philox2x32_fold_in_nd;
    name = "philox2x32";
    tag = "phx2";
  }

let philox4x32_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 2 |];
    seed = philox4x32_seed_nd;
    split = philox4x32_split_nd;
    random_bits = philox4x32_bits_nd;
    fold_in = philox4x32_fold_in_nd;
    name = "philox4x32";
    tag = "phx4";
  }

let concrete_nd = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "philox: alt-prng surface is concrete-only (M5)"
  | Device _ -> failwith "philox: alt-prng surface is concrete-only (M5)"

let philox2x32_seed s = Concrete (philox2x32_seed_nd (concrete_nd s))

let philox2x32_split k shape =
  Concrete (philox2x32_split_nd (concrete_nd k) shape)

let philox2x32_fold_in k d =
  Concrete (philox2x32_fold_in_nd (concrete_nd k) (concrete_nd d))

let philox2x32_random_bits k ~bit_width ~shape =
  Concrete (philox2x32_bits_nd (concrete_nd k) bit_width shape)

let philox4x32_seed s = Concrete (philox4x32_seed_nd (concrete_nd s))

let philox4x32_split k shape =
  Concrete (philox4x32_split_nd (concrete_nd k) shape)

let philox4x32_fold_in k d =
  Concrete (philox4x32_fold_in_nd (concrete_nd k) (concrete_nd d))

let philox4x32_random_bits k ~bit_width ~shape =
  Concrete (philox4x32_bits_nd (concrete_nd k) bit_width shape)
