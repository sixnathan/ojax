open Types

let b1 = Core.bind1
let u32 = Dtype.Uint32

let fill shape x =
  Concrete
    (Ndarray.of_floats u32 shape (Array.make (Array.fold_left ( * ) 1 shape) x))

let like v x = fill (Core.get_aval v).shape x
let add a b = b1 Add [ a; b ]
let xor a b = b1 Xor [ a; b ]

let rotate_left v d =
  b1 Or
    [
      b1 Shift_left [ v; like v (float_of_int d) ];
      b1 Shift_right_logical [ v; like v (float_of_int (32 - d)) ];
    ]

let apply_round (x0, x1) r =
  let x0' = add x0 x1 in
  let x1' = xor (rotate_left x1 r) x0' in
  (x0', x1')

let rounds_a = [ 13; 15; 26; 6 ]
let rounds_b = [ 17; 29; 16; 24 ]

let broadcast_to_shape v shape =
  let a = Core.get_aval v in
  if a.shape = shape then v
  else if Array.length a.shape = 0 then
    b1 (Broadcast_in_dim { shape; dims = [||] }) [ v ]
  else invalid_arg "Prng_lowering: incompatible threefry operands"

let threefry2x32 key1 key2 c1 c2 =
  let shape = (Core.get_aval c1).shape in
  let k0 = broadcast_to_shape key1 shape in
  let k1 = broadcast_to_shape key2 shape in
  let k2 = xor (xor k0 k1) (fill shape 466688986.0) in
  let inc v n = add v (fill shape (float_of_int n)) in
  let stage (x0, x1) ka kb rotations =
    let x0 = add x0 ka and x1 = add x1 kb in
    List.fold_left apply_round (x0, x1) rotations
  in
  let s = stage (c1, c2) k0 k1 rounds_a in
  let s = stage s k1 (inc k2 1) rounds_b in
  let s = stage s k2 (inc k0 2) rounds_a in
  let s = stage s k0 (inc k1 3) rounds_b in
  let s = stage s k1 (inc k2 4) rounds_a in
  let x0, x1 = s in
  [ add x0 k2; add x1 (inc k0 5) ]

let iota_counters shape =
  let total = Array.fold_left ( * ) 1 shape in
  let high = float_of_int (total lsr 32) in
  ignore high;
  ( Concrete
      (Ndarray.of_floats u32 shape
         (Array.init total (fun i -> float_of_int (i lsr 32)))),
    Concrete
      (Ndarray.of_floats u32 shape
         (Array.init total (fun i -> float_of_int (i land 0xFFFFFFFF)))) )

let key_words key =
  let piece i =
    b1 (Reshape [||])
      [
        b1
          (Slice
             {
               start_indices = [| i |];
               limit_indices = [| i + 1 |];
               strides = None;
             })
          [ key ];
      ]
  in
  (piece 0, piece 1)

let rec over_key_batch f key =
  let a = Core.get_aval key in
  if Array.length a.shape = 1 then f key
  else
    let rows = Core.bind (Unstack 0) [ key ] in
    b1 (Stack 0) (List.map (over_key_batch f) rows)

let random_bits ~shape key =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let c1, c2 = iota_counters shape in
      match threefry2x32 k1 k2 c1 c2 with
      | [ x; y ] -> xor x y
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key

let random_split ~shape key =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let c1, c2 = iota_counters shape in
      match threefry2x32 k1 k2 c1 c2 with
      | [ x; y ] -> b1 (Stack (Array.length shape)) [ x; y ]
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key

let seed_words seed =
  let a = Core.get_aval seed in
  let convert v = b1 (Convert_element_type u32) [ v ] in
  let low =
    match a.dtype with
    | Dtype.I64 | Dtype.Uint64 ->
        convert
          (b1 And
             [
               seed;
               Concrete
                 (Ndarray.of_int64s a.dtype a.shape
                    (Array.make (Array.fold_left ( * ) 1 a.shape) 0xFFFFFFFFL));
             ])
    | _ -> convert seed
  in
  let high =
    match a.dtype with
    | Dtype.I64 | Dtype.Uint64 ->
        convert
          (b1 Shift_right_logical
             [
               seed;
               Concrete
                 (Ndarray.of_int64s a.dtype a.shape
                    (Array.make (Array.fold_left ( * ) 1 a.shape) 32L));
             ])
    | _ -> fill a.shape 0.0
  in
  (high, low)

let random_seed seed =
  let high, low = seed_words seed in
  b1 (Stack (Array.length (Core.get_aval seed).shape)) [ high; low ]

let fold_in key data =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let high, low = seed_words data in
      let shape = (Core.get_aval data).shape in
      ignore shape;
      match threefry2x32 k1 k2 high low with
      | [ x; y ] -> b1 (Stack (Array.length (Core.get_aval x).shape)) [ x; y ]
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key

let random_bits64 ~shape key =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let c1, c2 = iota_counters shape in
      match threefry2x32 k1 k2 c1 c2 with
      | [ x; y ] ->
          let wide v = b1 (Convert_element_type Dtype.Uint64) [ v ] in
          let hi = wide x and lo = wide y in
          let thirty_two =
            Concrete
              (Ndarray.of_int64s Dtype.Uint64 shape
                 (Array.make (Array.fold_left ( * ) 1 shape) 32L))
          in
          b1 Or [ b1 Shift_left [ hi; thirty_two ]; lo ]
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key
