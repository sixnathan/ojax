open Types
module Nd = Ndarray
module Core = Ojax__Core

let mask = 0xFFFF_FFFF
let add32 a b = (a + b) land mask
let xor32 a b = a lxor b land mask
let rotl d x = (x lsl d) lor (x lsr (32 - d)) land mask
let rotations0 = [ 13; 15; 26; 6 ]
let rotations1 = [ 17; 29; 16; 24 ]
let parity_const = 0x1BD1_1BDA

let hash2x32 k1 k2 x1 x2 =
  let ks2 = xor32 (xor32 k1 k2) parity_const in
  let a = ref (add32 x1 k1) and b = ref (add32 x2 k2) in
  let round rot =
    let na = add32 !a !b in
    let nb = xor32 na (rotl rot !b) in
    a := na;
    b := nb
  in
  List.iter round rotations0;
  a := add32 !a k2;
  b := add32 !b (add32 ks2 1);
  List.iter round rotations1;
  a := add32 !a ks2;
  b := add32 !b (add32 k1 2);
  List.iter round rotations0;
  a := add32 !a k1;
  b := add32 !b (add32 k2 3);
  List.iter round rotations1;
  a := add32 !a k2;
  b := add32 !b (add32 ks2 4);
  List.iter round rotations0;
  a := add32 !a ks2;
  b := add32 !b (add32 k1 5);
  (!a, !b)

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
              invalid_arg "threefry: incompatible broadcast shapes")
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

let threefry2x32_nd k1 k2 x1 x2 =
  let os =
    broadcast_shapes [ Nd.shape k1; Nd.shape k2; Nd.shape x1; Nd.shape x2 ]
  in
  let n = prod os in
  let o0 = Array.make n 0 and o1 = Array.make n 0 in
  for i = 0 to n - 1 do
    let idx = unravel os i in
    let g nd = geti nd (bcast_flat (Nd.shape nd) os idx) in
    let a, b = hash2x32 (g k1) (g k2) (g x1) (g x2) in
    o0.(i) <- a;
    o1.(i) <- b
  done;
  (of_u32 os o0, of_u32 os o1)

let seed_nd seed =
  let v = Nd.get_i64 seed [||] in
  let k2 = Int64.to_int (Int64.logand v 0xFFFF_FFFFL) in
  let k1 =
    match Nd.dtype seed with
    | Dtype.I64 ->
        Int64.to_int
          (Int64.logand (Int64.shift_right_logical v 32) 0xFFFF_FFFFL)
    | _ -> 0
  in
  of_u32 [| 2 |] [| k1; k2 |]

let split_nd key shape =
  let hi, lo = Prng.iota_2x32_nd shape in
  let k1 = geti key 0 and k2 = geti key 1 in
  let n = prod shape in
  let out = Array.make (n * 2) 0 in
  for i = 0 to n - 1 do
    let a, b = hash2x32 k1 k2 (geti hi i) (geti lo i) in
    out.(2 * i) <- a;
    out.((2 * i) + 1) <- b
  done;
  of_u32 (Array.append shape [| 2 |]) out

let bits_nd key bit_width shape =
  if bit_width <> 32 then
    failwith "threefry: only 32-bit random_bits supported (M3)";
  let hi, lo = Prng.iota_2x32_nd shape in
  let k1 = geti key 0 and k2 = geti key 1 in
  let n = prod shape in
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    let a, b = hash2x32 k1 k2 (geti hi i) (geti lo i) in
    out.(i) <- xor32 a b
  done;
  of_u32 shape out

let fold_in_nd key msg =
  let d = Int64.to_int (Int64.logand (Nd.get_i64 msg [||]) 0xFFFF_FFFFL) in
  let k1 = geti key 0 and k2 = geti key 1 in
  let a, b = hash2x32 k1 k2 0 d in
  of_u32 [| 2 |] [| a; b |]

let threefry2x32_impl inputs =
  match inputs with
  | [ k1; k2; x1; x2 ] ->
      let a, b = threefry2x32_nd k1 k2 x1 x2 in
      [ a; b ]
  | _ -> failwith "threefry: threefry2x32 expects 4 operands"

let shaped shape = { shape; dtype = Dtype.Uint32; weak_type = false }

let threefry2x32_aeval avals =
  match avals with
  | [ a; b; c; d ] ->
      let os = broadcast_shapes [ a.shape; b.shape; c.shape; d.shape ] in
      [ shaped os; shaped os ]
  | _ -> failwith "threefry: threefry2x32 expects 4 avals"

let prev_impl = ref (fun (_ : primitive) (_ : Nd.t list) -> ([] : Nd.t list))
let prev_aeval = ref (fun (_ : primitive) (_ : aval list) -> ([] : aval list))

let impl prim inputs =
  match prim with
  | Threefry2x32 -> threefry2x32_impl inputs
  | _ -> !prev_impl prim inputs

let abstract_eval prim avals =
  match prim with
  | Threefry2x32 -> threefry2x32_aeval avals
  | _ -> !prev_aeval prim avals

let install () =
  prev_impl := Core.rules.impl;
  prev_aeval := Core.rules.abstract_eval;
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()

let threefry_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 2 |];
    seed = seed_nd;
    split = split_nd;
    random_bits = bits_nd;
    fold_in = fold_in_nd;
    name = "threefry2x32";
    tag = "fry";
  }

let () = Prng.register_prng threefry_prng_impl

let scalar_at v i =
  let sliced =
    Core.bind1
      (Slice
         {
           start_indices = [| i |];
           limit_indices = [| i + 1 |];
           strides = None;
         })
      [ v ]
  in
  Core.bind1 (Reshape [||]) [ sliced ]

let threefry_seed seed =
  match seed with
  | Concrete nd -> Concrete (seed_nd nd)
  | Tracer _ -> failwith "threefry: seed of a tracer not supported in M3"
  | Device _ -> failwith "threefry: seed of a tracer not supported in M3"

let threefry_2x32 keypair count =
  let key1 = scalar_at keypair 0 and key2 = scalar_at keypair 1 in
  let cshape = (Core.get_aval count).shape in
  let n = prod cshape in
  let flat = Core.bind1 (Reshape [| n |]) [ count ] in
  let odd = n mod 2 in
  let padded, m =
    if odd = 1 then
      let z =
        Core.bind1
          (Iota { dtype = Dtype.Uint32; shape = [| 1 |]; dimension = 0 })
          []
      in
      (Core.bind1 (Concatenate 0) [ flat; z ], n + 1)
    else (flat, n)
  in
  let half = m / 2 in
  let x0 =
    Core.bind1
      (Slice
         { start_indices = [| 0 |]; limit_indices = [| half |]; strides = None })
      [ padded ]
  in
  let x1 =
    Core.bind1
      (Slice
         { start_indices = [| half |]; limit_indices = [| m |]; strides = None })
      [ padded ]
  in
  let o0, o1 =
    match Core.bind Threefry2x32 [ key1; key2; x0; x1 ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "threefry: 2x32 arity"
  in
  let cat = Core.bind1 (Concatenate 0) [ o0; o1 ] in
  let trimmed =
    if odd = 1 then
      Core.bind1
        (Slice
           { start_indices = [| 0 |]; limit_indices = [| n |]; strides = None })
        [ cat ]
    else cat
  in
  Core.bind1 (Reshape cshape) [ trimmed ]

let threefry_split key shape =
  let key1 = scalar_at key 0 and key2 = scalar_at key 1 in
  let c1, c2 = Prng.iota_2x32_shape shape in
  match Core.bind Threefry2x32 [ key1; key2; c1; c2 ] with
  | [ b1; b2 ] -> Core.bind1 (Stack (Array.length shape)) [ b1; b2 ]
  | _ -> failwith "threefry: split arity"

let threefry_fold_in key data =
  let data_u32 = Core.bind1 (Convert_element_type Dtype.Uint32) [ data ] in
  threefry_2x32 key (threefry_seed data_u32)

let threefry_random_bits key bit_width shape =
  if bit_width <> 32 then
    failwith "threefry: only 32-bit random_bits supported (M3)";
  let key1 = scalar_at key 0 and key2 = scalar_at key 1 in
  let c1, c2 = Prng.iota_2x32_shape shape in
  match Core.bind Threefry2x32 [ key1; key2; c1; c2 ] with
  | [ b1; b2 ] -> Core.bind1 Xor [ b1; b2 ]
  | _ -> failwith "threefry: random_bits arity"
