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
  if Array.length a.shape = Array.length shape then v
  else if Array.length a.shape = 0 then
    b1 (Broadcast_in_dim { shape; dims = [||] }) [ v ]
  else invalid_arg "Prng_lowering: incompatible threefry operands"

let i32_scalar x =
  Concrete (Ndarray.of_floats (Dtypes.default_int_dtype ()) [||] [| x |])

let u32_scalar x = Concrete (Ndarray.of_floats u32 [||] [| x |])

let rotation_array values =
  Concrete
    (Ndarray.of_floats u32 [| 4 |]
       (Array.of_list (List.map float_of_int values)))

let rotate_scalar_left v rotation =
  let shape = (Core.get_aval v).shape in
  let spread r = broadcast_to_shape r shape in
  b1 Or
    [
      b1 Shift_left [ v; spread rotation ];
      b1 Shift_right_logical
        [ v; spread (b1 Sub [ u32_scalar 32.0; rotation ]) ];
    ]

let rotation_at rotations i =
  b1 (Reshape [||])
    [
      b1
        (Slice
           {
             start_indices = [| i |];
             limit_indices = [| i + 1 |];
             strides = None;
           })
        [ rotations ];
    ]

let threefry_step = function
  | [ i; x0; x1; k0; k1; k2; rotations; next_rotations ] ->
      let one = i32_scalar 1.0 in
      let next_i = b1 Add [ i; one ] in
      let shape = (Core.get_aval x0).shape in
      let spread v = broadcast_to_shape v shape in
      let x0, x1 =
        List.fold_left
          (fun (x0, x1) index ->
            let rotation = rotation_at rotations index in
            let x0' = add x0 x1 in
            (x0', xor x0' (rotate_scalar_left x1 rotation)))
          (x0, x1) [ 0; 1; 2; 3 ]
      in
      let x0' = add x0 (spread k0) in
      let x1' = add x1 (spread k1) in
      let counter =
        spread (b1 (Convert_element_type u32) [ b1 Add [ i; one ] ])
      in
      [ next_i; x0'; add x1' counter; k1; k2; k0; next_rotations; rotations ]
  | _ -> invalid_arg "Prng_lowering: threefry state"

let threefry2x32_body key1 key2 c1 c2 =
  let shape =
    Core.elementwise_shape
      [ (Core.get_aval c1).shape; (Core.get_aval key1).shape ]
  in
  let spread v = broadcast_to_shape v shape in
  let key3 = xor (xor key1 key2) (u32_scalar 466688986.0) in
  let x0 = add (spread c1) (spread key1) in
  let x1 = add (spread c2) (spread key2) in
  match
    Lax.scan ~length:5 threefry_step
      [
        i32_scalar 0.0;
        x0;
        x1;
        key2;
        key3;
        key1;
        rotation_array rounds_a;
        rotation_array rounds_b;
      ]
      []
  with
  | _ :: x0 :: x1 :: _ -> [ x0; x1 ]
  | _ -> invalid_arg "Prng_lowering: threefry state"

let threefry2x32 key1 key2 c1 c2 =
  Jaxpr.named_jit ~name:"threefry2x32"
    (function
      | [ k1; k2; x1; x2 ] -> threefry2x32_body k1 k2 x1 x2
      | _ -> invalid_arg "Prng_lowering: threefry state")
    [ key1; key2; c1; c2 ]

let iota_counters shape =
  if Array.length shape = 0 then (u32_scalar 0.0, u32_scalar 0.0)
  else begin
    let u64 = Dtype.Uint64 in
    let u64_scalar x = Concrete (Ndarray.of_int64s u64 [||] [| x |]) in
    let spread v = b1 (Broadcast_in_dim { shape; dims = [||] }) [ v ] in
    let n = Array.length shape in
    let strides =
      Array.init n (fun d ->
          let product = ref 1 in
          for k = d + 1 to n - 1 do
            product := !product * shape.(k)
          done;
          !product)
    in
    let iotas =
      List.init n (fun dimension ->
          b1 (Iota { dtype = u64; shape; dimension }) [])
    in
    let scaled =
      List.mapi
        (fun d iota ->
          b1 Mul [ spread (u64_scalar (Int64.of_int strides.(d))); iota ])
        iotas
    in
    let counts =
      match scaled with
      | [] -> invalid_arg "Prng_lowering: iota rank"
      | first :: rest -> List.fold_left add first rest
    in
    let shifted = b1 Shift_right_logical [ counts; spread (u64_scalar 32L) ] in
    let lo = b1 (Convert_element_type u32) [ counts ] in
    let hi = b1 (Convert_element_type u32) [ shifted ] in
    (hi, lo)
  end

let key_words key =
  let sh = (Core.get_aval key).shape in
  let r = Array.length sh in
  let piece i =
    let start_indices = Array.make r 0 and limit_indices = Array.copy sh in
    start_indices.(r - 1) <- i;
    limit_indices.(r - 1) <- i + 1;
    b1
      (Reshape (Array.sub sh 0 (r - 1)))
      [ b1 (Slice { start_indices; limit_indices; strides = None }) [ key ] ]
  in
  (piece 0, piece 1)

let batched_threefry k1 k2 c1 c2 =
  let ka = (Core.get_aval k1).shape in
  if Array.length ka = 0 then threefry2x32 k1 k2 c1 c2
  else
    let lift_count c =
      let sh = (Core.get_aval c).shape in
      let n = Array.length sh in
      if n = 0 then c
      else
        b1
          (Broadcast_in_dim
             {
               shape = Array.append [| 1 |] sh;
               dims = Array.init n (fun d -> d + 1);
             })
          [ c ]
    in
    let c1 = lift_count c1 in
    let c2 = lift_count c2 in
    let nd =
      Stdlib.max (Array.length ka) (Array.length (Core.get_aval c1).shape)
    in
    let lift_key k =
      if Array.length ka = nd then k
      else
        b1
          (Broadcast_in_dim
             {
               shape = Array.append ka (Array.make (nd - Array.length ka) 1);
               dims = Array.init (Array.length ka) Fun.id;
             })
          [ k ]
    in
    let k1 = lift_key k1 in
    let k2 = lift_key k2 in
    threefry2x32 k1 k2 c1 c2

let rec over_key_batch f key =
  let a = Core.get_aval key in
  if Array.length a.shape <= 2 then f key
  else
    let rows = Core.bind (Unstack 0) [ key ] in
    b1 (Stack 0) (List.map (over_key_batch f) rows)

let random_bits ~shape key =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let c1, c2 = iota_counters shape in
      match batched_threefry k1 k2 c1 c2 with
      | [ x; y ] -> xor x y
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key

let random_split ~shape key =
  over_key_batch
    (fun k ->
      List.hd
        (Jaxpr.named_jit ~name:"_threefry_split"
           ~key:
             (String.concat "," (List.map string_of_int (Array.to_list shape)))
           (function
             | [ k ] -> (
                 let k1, k2 = key_words k in
                 let c1, c2 = iota_counters shape in
                 match batched_threefry k1 k2 c1 c2 with
                 | [ x; y ] ->
                     let n = Array.length (Core.get_aval x).shape in
                     [ b1 (Stack n) [ x; y ] ]
                 | _ -> invalid_arg "Prng_lowering: threefry arity")
             | _ -> invalid_arg "Prng_lowering: split arity")
           [ k ]))
    key

let seed_const dt (a : aval) v =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete
    (match dt with
    | Dtype.I64 | Dtype.Uint64 -> Ndarray.of_int64s dt a.shape (Array.make n v)
    | Dtype.Uint32 ->
        Ndarray.of_floats dt a.shape (Array.make n (Int64.to_float v))
    | _ ->
        let wrapped = Int32.to_float (Int64.to_int32 v) in
        Ndarray.of_floats dt a.shape (Array.make n wrapped))

let seed_wide_dtype dt =
  Dtypes.canonicalize_dtype (Dtypes.promote_types dt Dtype.Uint32)

let seed_words seed =
  let a = Core.get_aval seed in
  let convert v = b1 (Convert_element_type u32) [ v ] in
  let high =
    convert (b1 Shift_right_logical [ seed; seed_const a.dtype a 32L ])
  in
  let wide = seed_wide_dtype a.dtype in
  let widened =
    if wide = a.dtype then seed else b1 (Convert_element_type wide) [ seed ]
  in
  let low = convert (b1 And [ widened; seed_const wide a 0xFFFFFFFFL ]) in
  (high, low)

let seed_scalar seed =
  let a = Core.get_aval seed in
  let word v =
    b1
      (Broadcast_in_dim { shape = [| 1 |]; dims = [||] })
      [ b1 (Convert_element_type u32) [ v ] ]
  in
  let high = word (b1 Shift_right_logical [ seed; seed_const a.dtype a 32L ]) in
  let wide = seed_wide_dtype a.dtype in
  let widened =
    if wide = a.dtype then seed else b1 (Convert_element_type wide) [ seed ]
  in
  let low = word (b1 And [ widened; seed_const wide a 0xFFFFFFFFL ]) in
  b1 (Concatenate 0) [ high; low ]

let random_seed seed =
  if Array.length (Core.get_aval seed).shape = 0 then seed_scalar seed
  else
    let high, low = seed_words seed in
    b1 (Stack (Array.length (Core.get_aval seed).shape)) [ high; low ]

let threefry_2x32 key count =
  let k1, k2 = key_words key in
  let n = (Core.get_aval count).shape.(0) in
  let half = n / 2 in
  let x = Core.bind (Split { sizes = [| half; half |]; axis = 0 }) [ count ] in
  match x with
  | [ x0; x1 ] -> (
      match threefry2x32 k1 k2 x0 x1 with
      | [ a; b ] -> b1 (Concatenate 0) [ a; b ]
      | _ -> invalid_arg "Prng_lowering: threefry arity")
  | _ -> invalid_arg "Prng_lowering: split arity"

let rec fold_in key data =
  let ka = Core.get_aval key and da = Core.get_aval data in
  if Array.length ka.shape = 1 && Array.length da.shape = 0 then
    List.hd
      (Jaxpr.named_jit ~name:"_threefry_fold_in"
         (function
           | [ key; data ] -> [ threefry_2x32 key (seed_scalar data) ]
           | _ -> invalid_arg "Prng_lowering: fold_in arity")
         [ key; data ])
  else if Array.length ka.shape = 1 then
    let k1, k2 = key_words key in
    let high, low = seed_words data in
    match threefry2x32 k1 k2 high low with
    | [ x; y ] -> b1 (Stack (Array.length (Core.get_aval x).shape)) [ x; y ]
    | _ -> invalid_arg "Prng_lowering: threefry arity"
  else
    let keys = Core.bind (Unstack 0) [ key ] in
    let datas =
      if Array.length da.shape = 0 then List.map (fun _ -> data) keys
      else Core.bind (Unstack 0) [ data ]
    in
    b1 (Stack 0) (List.map2 fold_in keys datas)

let random_bits64 ~shape key =
  over_key_batch
    (fun k ->
      let k1, k2 = key_words k in
      let c1, c2 = iota_counters shape in
      match batched_threefry k1 k2 c1 c2 with
      | [ x; y ] ->
          let wide v = b1 (Convert_element_type Dtype.Uint64) [ v ] in
          let hi = wide x and lo = wide y in
          let thirty_two =
            Concrete (Ndarray.of_int64s Dtype.Uint64 [||] [| 32L |])
          in
          b1 Or [ b1 Shift_left [ hi; thirty_two ]; lo ]
      | _ -> invalid_arg "Prng_lowering: threefry arity")
    key
