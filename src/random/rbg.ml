open Types
module Core = Ojax__Core
module TF = Threefry
module LX = Lax_extras
module B = Interpreters.Batching

let dtype v = (Core.get_aval v).dtype
let shape v = (Core.get_aval v).shape
let is_unsafe k = Prng.is_rbg_key k && dtype k = Dtype.I32
let bitcast dt v = Core.bind1 (Bitcast_convert_type dt) [ v ]
let key_data k = if is_unsafe k then bitcast Dtype.Uint32 k else k
let wrap ~unsafe k = if unsafe then bitcast Dtype.I32 k else k
let reshape dims v = Core.bind1 (Reshape dims) [ v ]
let prod a = Array.fold_left ( * ) 1 a
let halves key = reshape [| 2; 2 |] key

let rec iterated f key =
  if Array.length (shape key) = 1 then f key
  else
    List.hd
      (B.vmap_flat
         (function [ k ] -> [ iterated f k ] | _ -> failwith "rbg: arity")
         [ Some 0 ] [ key ])

let half_seed seed = TF.threefry_seed seed

let rbg_seed seed =
  let half = half_seed seed in
  Core.bind1 (Concatenate 0) [ half; half ]

let seed ~unsafe s = wrap ~unsafe (rbg_seed s)

let bit_generator ?dtype:(element = Dtype.Uint32) key out_shape =
  snd (LX.rng_bit_generator ~dtype:element key out_shape)

let rbg_split shape key =
  let split =
    if Config.value Config.threefry_partitionable then TF.threefry_split
    else TF.threefry_split_original
  in
  let parts =
    List.hd
      (B.vmap_flat ~out_axis:(Array.length shape)
         (function
           | [ half ] -> [ split half shape ] | _ -> failwith "rbg: split arity")
         [ Some 0 ]
         [ halves key ])
  in
  reshape (Array.append shape [| 4 |]) parts

let unsafe_split shape key =
  let num = prod shape in
  let keys = bit_generator key [| 10 * num; 4 |] in
  let stepped =
    Core.bind1
      (Slice
         {
           start_indices = [| 0; 0 |];
           limit_indices = [| 10 * num; 4 |];
           strides = Some [| 10; 1 |];
         })
      [ keys ]
  in
  reshape (Array.append shape [| 4 |]) stepped

let rbg_fold_in msg key =
  let folded =
    List.hd
      (B.vmap_flat
         (function
           | [ half ] -> [ TF.threefry_fold_in half msg ]
           | _ -> failwith "rbg: fold_in arity")
         [ Some 0 ]
         [ halves key ])
  in
  reshape [| 4 |] folded

let unsafe_fold_in msg key =
  let bits = bit_generator (rbg_seed msg) [| 10; 4 |] in
  let last =
    Core.bind1
      (Slice
         {
           start_indices = [| 9; 0 |];
           limit_indices = [| 10; 4 |];
           strides = None;
         })
      [ bits ]
  in
  Core.bind1 Xor [ key; reshape [| 4 |] last ]

let split key out_shape =
  let unsafe = is_unsafe key in
  let body = if unsafe then unsafe_split else rbg_split in
  wrap ~unsafe (iterated (body out_shape) (key_data key))

let fold_in key msg =
  let unsafe = is_unsafe key in
  let body = if unsafe then unsafe_fold_in else rbg_fold_in in
  wrap ~unsafe (iterated (body msg) (key_data key))

let random_bits key bit_width out_shape =
  let element =
    match bit_width with
    | 8 -> Dtype.Uint8
    | 16 -> Dtype.Uint16
    | 32 -> Dtype.Uint32
    | 64 -> Dtype.Uint64
    | _ ->
        raise
          (Errors.jax_type_error "requires 8-, 16-, 32- or 64-bit field width.")
  in
  iterated (fun k -> bit_generator ~dtype:element k out_shape) (key_data key)

let () =
  Prng.rbg_split_hook := split;
  Prng.rbg_fold_in_hook := fold_in;
  Prng.rbg_bits_hook := random_bits
