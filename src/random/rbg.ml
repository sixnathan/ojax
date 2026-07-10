open Types
module Nd = Ndarray
module TF = Threefry

let unravel shape i =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let r = ref i in
  for k = n - 1 downto 0 do
    idx.(k) <- !r mod shape.(k);
    r := !r / shape.(k)
  done;
  idx

let prod a = Array.fold_left ( * ) 1 a
let geti nd i = Int64.to_int (Nd.get_i64 nd (unravel (Nd.shape nd) i))

let of_u32 shape ints =
  Nd.of_floats Dtype.Uint32 shape (Array.map float_of_int ints)

let half key j = of_u32 [| 2 |] [| geti key (2 * j); geti key ((2 * j) + 1) |]

let rbg_seed_nd seed =
  let h = TF.threefry_prng_impl.seed seed in
  of_u32 [| 4 |] [| geti h 0; geti h 1; geti h 0; geti h 1 |]

let rbg_split_nd key shape =
  let s0 = TF.threefry_prng_impl.split (half key 0) shape in
  let s1 = TF.threefry_prng_impl.split (half key 1) shape in
  let n = prod shape in
  let out = Array.make (n * 4) 0 in
  for p = 0 to n - 1 do
    out.(4 * p) <- geti s0 (2 * p);
    out.((4 * p) + 1) <- geti s0 ((2 * p) + 1);
    out.((4 * p) + 2) <- geti s1 (2 * p);
    out.((4 * p) + 3) <- geti s1 ((2 * p) + 1)
  done;
  of_u32 (Array.append shape [| 4 |]) out

let rbg_fold_in_nd key data =
  let f0 = TF.threefry_prng_impl.fold_in (half key 0) data in
  let f1 = TF.threefry_prng_impl.fold_in (half key 1) data in
  of_u32 [| 4 |] [| geti f0 0; geti f0 1; geti f1 0; geti f1 1 |]

let rng_bit_generator_unavailable _ _ _ =
  failwith
    "rbg: random_bits lowers to XLA lax.rng_bit_generator, whose output is not \
     guaranteed stable across backends/compiler versions (per jax) and is not \
     host-reproducible bit-exactly; device-only, not host-goldenable (M5)"

let unsafe_split_unavailable _ _ =
  failwith
    "unsafe_rbg: split lowers to XLA lax.rng_bit_generator (non-portable, \
     device-only); not host-reproducible bit-exactly (M5)"

let unsafe_fold_in_unavailable _ _ =
  failwith
    "unsafe_rbg: fold_in lowers to XLA lax.rng_bit_generator (non-portable, \
     device-only); not host-reproducible bit-exactly (M5)"

let rbg_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 4 |];
    seed = rbg_seed_nd;
    split = rbg_split_nd;
    random_bits = rng_bit_generator_unavailable;
    fold_in = rbg_fold_in_nd;
    name = "rbg";
    tag = "rbg";
  }

let unsafe_rbg_prng_impl : Prng.prng_impl =
  {
    key_shape = [| 4 |];
    seed = rbg_seed_nd;
    split = unsafe_split_unavailable;
    random_bits = rng_bit_generator_unavailable;
    fold_in = unsafe_fold_in_unavailable;
    name = "unsafe_rbg";
    tag = "urbg";
  }

let concrete_nd = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "rbg: alt-prng surface is concrete-only (M5)"
  | Device _ -> failwith "rbg: alt-prng surface is concrete-only (M5)"

let rbg_seed s = Concrete (rbg_seed_nd (concrete_nd s))
let rbg_split k shape = Concrete (rbg_split_nd (concrete_nd k) shape)
let rbg_fold_in k d = Concrete (rbg_fold_in_nd (concrete_nd k) (concrete_nd d))
