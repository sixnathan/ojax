open Types
module Nd = Ndarray
module C = Ojax__Core
module D = Dtype
module PR = Prng
module TF = Threefry

let prod a = Array.fold_left ( * ) 1 a
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let concrete = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "random/core: sampler on a tracer not supported in M3"

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

let const dt shape x =
  Concrete
    (Nd.canonicalize dt (Nd.of_floats dt shape (Array.make (prod shape) x)))

let wrap k = PR.random_wrap k
let unwrap k = PR.random_unwrap k
let mask32 = 0xFFFFFFFFL
let default_prng_impl () = TF.threefry_prng_impl

let resolve_prng_impl = function
  | None -> TF.threefry_prng_impl
  | Some "threefry2x32" -> TF.threefry_prng_impl
  | Some name ->
      invalid_arg
        (Printf.sprintf "unrecognized PRNG implementation \"%s\"" name)

let key_impl _keys = TF.threefry_prng_impl.PR.name
let key_dtype _impl_spec = D.Uint32
let key seed = PR.random_seed seed
let key_data keys = unwrap keys
let wrap_key_data data = wrap data
let clone k = k

let fold_in k data =
  unwrap
    (PR.random_fold_in (wrap k)
       (C.bind1 (Convert_element_type D.Uint32) [ data ]))

let split k num = unwrap (PR.random_split (wrap k) [| num |])
let bits k ~shape = PR.random_bits (wrap k) ~bit_width:32 ~shape

let uniform k ~shape ~minval ~maxval =
  let dt = D.F32 in
  let raw = concrete (PR.random_bits (wrap k) ~bit_width:32 ~shape) in
  let n = prod shape in
  let floats =
    Array.init n (fun i ->
        let b = geti raw i in
        let fb = (b lsr 9) lor 0x3F800000 in
        Int32.float_of_bits (Int32.of_int fb))
  in
  let floats_v = Concrete (Nd.of_floats dt shape floats) in
  let minv = f32 minval and maxv = f32 maxval in
  let span = f32 (maxv -. minv) in
  let u = C.bind1 Sub [ floats_v; const dt shape 1.0 ] in
  let scaled = C.bind1 Mul [ u; const dt shape span ] in
  let y = C.bind1 Add [ scaled; const dt shape minv ] in
  C.bind1 Max [ const dt shape minv; y ]

let scalar_f v = Nd.get_f (concrete v) [||]

let normal k ~shape =
  let dt = D.F32 in
  let lo =
    scalar_f (C.bind1 Nextafter [ const dt [||] (-1.0); const dt [||] 0.0 ])
  in
  let u = uniform k ~shape ~minval:lo ~maxval:1.0 in
  let sqrt2 = f32 (sqrt 2.0) in
  C.bind1 Mul [ const dt shape sqrt2; C.bind1 Erf_inv [ u ] ]

let truncated_normal k ~lower ~upper ~shape =
  let dt = D.F32 in
  let sqrt2 = f32 (sqrt 2.0) in
  let lower_f = f32 lower and upper_f = f32 upper in
  let erf_scaled b =
    scalar_f
      (C.bind1 Erf [ C.bind1 Div [ const dt [||] b; const dt [||] sqrt2 ] ])
  in
  let a = erf_scaled lower_f and b = erf_scaled upper_f in
  let u = uniform k ~shape ~minval:a ~maxval:b in
  let out = C.bind1 Mul [ const dt shape sqrt2; C.bind1 Erf_inv [ u ] ] in
  let lo =
    scalar_f
      (C.bind1 Nextafter [ const dt [||] lower_f; const dt [||] infinity ])
  in
  let hi =
    scalar_f
      (C.bind1 Nextafter [ const dt [||] upper_f; const dt [||] neg_infinity ])
  in
  C.bind1 Min [ const dt shape hi; C.bind1 Max [ const dt shape lo; out ] ]

let randint k ~shape ~minval ~maxval =
  let keys = concrete (PR.random_split (wrap k) [| 2 |]) in
  let subkey j =
    Concrete
      (Nd.of_floats D.Uint32 [| 2 |]
         [|
           Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
           Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
         |])
  in
  let hb = concrete (PR.random_bits (wrap (subkey 0)) ~bit_width:32 ~shape) in
  let lb = concrete (PR.random_bits (wrap (subkey 1)) ~bit_width:32 ~shape) in
  let span =
    if maxval <= minval then 1L
    else Int64.logand (Int64.of_int (maxval - minval)) mask32
  in
  let m1 = Int64.rem 0x10000L span in
  let mult = Int64.rem (Int64.mul m1 m1) span in
  let n = prod shape in
  let out =
    Array.init n (fun i ->
        let h = Int64.of_int (geti hb i) and l = Int64.of_int (geti lb i) in
        let t1 = Int64.rem h span in
        let t2 = Int64.logand (Int64.mul t1 mult) mask32 in
        let t3 = Int64.logand (Int64.add t2 (Int64.rem l span)) mask32 in
        let off = Int64.rem t3 span in
        float_of_int (minval + Int64.to_int off))
  in
  Concrete (Nd.of_floats D.I32 shape out)

let uint32max = 4294967295.0

let permutation k n =
  let dt = Dtypes.default_int_dtype () in
  let x0 = C.bind1 (Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [] in
  let rounds =
    int_of_float (ceil (3.0 *. log (float_of_int (max 1 n)) /. log uint32max))
  in
  let subkey keys j =
    Concrete
      (Nd.of_floats D.Uint32 [| 2 |]
         [|
           Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
           Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
         |])
  in
  let rec loop cur x r =
    if r = 0 then x
    else
      let keys = concrete (PR.random_split (wrap cur) [| 2 |]) in
      let nextk = subkey keys 0 and subk = subkey keys 1 in
      let sort_keys = PR.random_bits (wrap subk) ~bit_width:32 ~shape:[| n |] in
      let outs =
        C.bind
          (Sort { dimension = 0; is_stable = true; num_keys = 1 })
          [ sort_keys; x ]
      in
      loop nextk (List.nth outs 1) (r - 1)
  in
  loop k x0 rounds

let choice k ~n ~shape ~replace =
  let dt = Dtypes.default_int_dtype () in
  let n_draws = prod shape in
  if n_draws = 0 then Concrete (Nd.of_floats dt shape [||])
  else if replace then
    C.bind1 (Convert_element_type dt) [ randint k ~shape ~minval:0 ~maxval:n ]
  else
    let perm = permutation k n in
    let sliced =
      C.bind1
        (Slice
           {
             start_indices = [| 0 |];
             limit_indices = [| n_draws |];
             strides = None;
           })
        [ perm ]
    in
    C.bind1 (Reshape shape) [ sliced ]
