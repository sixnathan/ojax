open Types
module Nd = Ndarray
module Core = Ojax__Core

let mask = 0xFFFF_FFFF
let add32 a b = (a + b) land mask
let xor32 a b = a lxor b land mask
let rotl d x = (x lsl d) lor (x lsr (32 - d)) land mask
let parity_const = 0x1BD1_1BDA
let default_rounds = 20

let rotations =
  [|
    (10, 26); (11, 21); (13, 27); (23, 5); (6, 20); (17, 11); (25, 10); (18, 20);
  |]

let threefry4x32_words k0 k1 k2 k3 y0 y1 y2 y3 =
  let ks =
    [| k0; k1; k2; k3; xor32 (xor32 (xor32 (xor32 k0 k1) k2) k3) parity_const |]
  in
  let x0 = ref (add32 y0 ks.(0)) in
  let x1 = ref (add32 y1 ks.(1)) in
  let x2 = ref (add32 y2 ks.(2)) in
  let x3 = ref (add32 y3 ks.(3)) in
  for rnd = 0 to default_rounds - 1 do
    let rot0, rot1 = rotations.(rnd mod 8) in
    if rnd mod 2 = 0 then (
      x0 := add32 !x0 !x1;
      x1 := xor32 !x0 (rotl rot0 !x1);
      x2 := add32 !x2 !x3;
      x3 := xor32 !x2 (rotl rot1 !x3))
    else (
      x0 := add32 !x0 !x3;
      x3 := xor32 !x0 (rotl rot0 !x3);
      x2 := add32 !x2 !x1;
      x1 := xor32 !x2 (rotl rot1 !x1));
    if rnd land 3 = 3 then (
      let inj = rnd / 4 in
      x0 := add32 !x0 ks.((1 + inj) mod 5);
      x1 := add32 !x1 ks.((2 + inj) mod 5);
      x2 := add32 !x2 ks.((3 + inj) mod 5);
      x3 := add32 !x3 ks.((4 + inj) mod 5);
      x3 := add32 !x3 (1 + inj))
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
              invalid_arg "threefry4x32: incompatible broadcast shapes")
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

let threefry4x32_nd k0 k1 k2 k3 x0 x1 x2 x3 =
  let shapes =
    [
      Nd.shape k0;
      Nd.shape k1;
      Nd.shape k2;
      Nd.shape k3;
      Nd.shape x0;
      Nd.shape x1;
      Nd.shape x2;
      Nd.shape x3;
    ]
  in
  let os = broadcast_shapes shapes in
  let n = prod os in
  let o0 = Array.make n 0 and o1 = Array.make n 0 in
  let o2 = Array.make n 0 and o3 = Array.make n 0 in
  for i = 0 to n - 1 do
    let idx = unravel os i in
    let g nd = geti nd (bcast_flat (Nd.shape nd) os idx) in
    let a, b, c, d =
      threefry4x32_words (g k0) (g k1) (g k2) (g k3) (g x0) (g x1) (g x2) (g x3)
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

let threefry4x32_seed_nd seed =
  let hi, lo = seed_halves seed in
  let a, b, c, d = threefry4x32_words hi lo 0 0 0 0 0 0 in
  of_u32 [| 4 |] [| a; b; c; d |]

let threefry4x32_split_nd key shape =
  let k0 = geti key 0 and k1 = geti key 1 in
  let k2 = geti key 2 and k3 = geti key 3 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make (n * 4) 0 in
  for i = 0 to n - 1 do
    let a, b, c, d =
      threefry4x32_words k0 k1 k2 k3 0 0 (geti hi i) (geti lo i)
    in
    out.(4 * i) <- a;
    out.((4 * i) + 1) <- b;
    out.((4 * i) + 2) <- c;
    out.((4 * i) + 3) <- d
  done;
  of_u32 (Array.append shape [| 4 |]) out

let threefry4x32_bits_nd key bit_width shape =
  if bit_width <> 32 then
    failwith "threefry4x32: only 32-bit random_bits supported (M5)";
  let k0 = geti key 0 and k1 = geti key 1 in
  let k2 = geti key 2 and k3 = geti key 3 in
  let hi, lo = iota_2x32 shape in
  let n = prod shape in
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    let a, b, c, d =
      threefry4x32_words k0 k1 k2 k3 (geti hi i) (geti lo i) 0 0
    in
    out.(i) <- xor32 (xor32 (xor32 a b) c) d
  done;
  of_u32 shape out

let threefry4x32_fold_in_nd key data =
  let d = Int64.to_int (Int64.logand (Nd.get_i64 data [||]) 0xFFFF_FFFFL) in
  let k0 = geti key 0 and k1 = geti key 1 in
  let k2 = geti key 2 and k3 = geti key 3 in
  let a, b, c, e = threefry4x32_words k0 k1 k2 k3 0 0 0 d in
  of_u32 [| 4 |] [| a; b; c; e |]

let shaped shape = { shape; dtype = Dtype.Uint32; weak_type = false }

let threefry4x32_impl inputs =
  match inputs with
  | [ k0; k1; k2; k3; x0; x1; x2; x3 ] ->
      let a, b, c, d = threefry4x32_nd k0 k1 k2 k3 x0 x1 x2 x3 in
      [ a; b; c; d ]
  | _ -> failwith "threefry4x32: expects 8 operands"

let threefry4x32_aeval avals =
  match avals with
  | [ a; b; c; d; e; f; g; h ] ->
      let os =
        broadcast_shapes
          [
            a.shape;
            b.shape;
            c.shape;
            d.shape;
            e.shape;
            f.shape;
            g.shape;
            h.shape;
          ]
      in
      [ shaped os; shaped os; shaped os; shaped os ]
  | _ -> failwith "threefry4x32: expects 8 avals"

let prev_impl = ref (fun (_ : primitive) (_ : Nd.t list) -> ([] : Nd.t list))
let prev_aeval = ref (fun (_ : primitive) (_ : aval list) -> ([] : aval list))

let impl prim inputs =
  match prim with
  | Threefry4x32 -> threefry4x32_impl inputs
  | _ -> !prev_impl prim inputs

let abstract_eval prim avals =
  match prim with
  | Threefry4x32 -> threefry4x32_aeval avals
  | _ -> !prev_aeval prim avals

let install () =
  prev_impl := Core.rules.impl;
  prev_aeval := Core.rules.abstract_eval;
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()

let threefry4x32_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 4 |];
    seed = threefry4x32_seed_nd;
    split = threefry4x32_split_nd;
    random_bits = threefry4x32_bits_nd;
    fold_in = threefry4x32_fold_in_nd;
    name = "threefry4x32";
    tag = "fry4";
  }

let concrete_nd = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "threefry4x32: alt-prng surface is concrete-only (M5)"
  | Device _ -> failwith "threefry4x32: alt-prng surface is concrete-only (M5)"

let threefry4x32_seed s = Concrete (threefry4x32_seed_nd (concrete_nd s))

let threefry4x32_split k shape =
  Concrete (threefry4x32_split_nd (concrete_nd k) shape)

let threefry4x32_fold_in k d =
  Concrete (threefry4x32_fold_in_nd (concrete_nd k) (concrete_nd d))

let threefry4x32_random_bits k ~bit_width ~shape =
  Concrete (threefry4x32_bits_nd (concrete_nd k) bit_width shape)
