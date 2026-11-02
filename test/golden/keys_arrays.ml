module U = Yojson.Safe.Util
module RC = Ojax.Random.Core
module NL = Ojax.Numpy.Lax_numpy
module AM = Ojax.Numpy.Array_methods
module AC = Ojax.Numpy.Array_constructors
module JU = Ojax.Numpy.Jnp_util
module UF = Ojax.Numpy.Ufuncs
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module L = Ojax.Lax_api

type outcome = Text of string | Value of Ojax.Types.value

let name = "keys_arrays"

let arr dt shape xs =
  AC.asarray ~dtype:dt (Ojax.Types.Concrete (Nd.of_floats dt shape xs))

let carr dt xs =
  AC.asarray ~dtype:dt
    (Ojax.Types.Concrete
       (Nd.of_complex dt
          [| Array.length xs |]
          (Array.map (fun (re, im) -> { Complex.re; im }) xs)))

let u32 n = NL.arange ~dtype:D.Uint32 (float_of_int n)
let f32_2x3 () = NL.reshape (NL.arange ~dtype:D.F32 6.0) [| 2; 3 |]
let seed n = JU.py_int n
let key ?impl n = RC.key ?impl (seed n)
let kd = RC.key_dtype
let kname = RC.key_dtype_name
let bool b = Text (if b then "True" else "False")
let wrap ?impl data = RC.wrap_key_data ?impl data
let fweak x = AC.asarray (JU.py_float x)
let iweak x = AC.asarray (JU.py_int x)
let repr v = Text (Ojax.Numpy.Extras.repr v)
let krepr v = Text (RC.key_repr v)
let ktype v = Text (RC.key_typeof v)
let typeof v = Text (Ojax.Api.typeof_text v)
let item ?args v = Text (AM.py_scalar_repr (AM.item ?args v))
let tolist v = Text (AM.py_list_repr (AM.tolist v))

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let sub a b = bool (NL.issubdtype a b)
let kcls d = RC.key_dtype_class d
let split_shape k shape = Ojax.Random.Prng.random_split k shape

let cases : (string * (unit -> outcome)) list =
  [
    ("key_dtype_default", fun () -> Text (kname (kd None)));
    ("key_dtype_threefry2x32", fun () -> Text (kname (kd (Some "threefry2x32"))));
    ("key_dtype_rbg", fun () -> Text (kname (kd (Some "rbg"))));
    ("key_dtype_unsafe_rbg", fun () -> Text (kname (kd (Some "unsafe_rbg"))));
    ("key_dtype_bad", fun () -> Text (kname (kd (Some "bad"))));
    ( "key_dtype_itemsize_fry",
      fun () -> Text (string_of_int (RC.key_dtype_itemsize (kd None))) );
    ( "key_dtype_itemsize_rbg",
      fun () -> Text (string_of_int (RC.key_dtype_itemsize (kd (Some "rbg"))))
    );
    ("key_dtype_of_key", fun () -> Text (kname (RC.dtype_of_key (key 0))));
    ( "key_dtype_of_rbg_key",
      fun () -> Text (kname (RC.dtype_of_key (key ~impl:"rbg" 0))) );
    ("key_dtype_eq_fry", fun () -> bool (RC.dtype_of_key (key 0) = kd None));
    ( "key_dtype_eq_rbg",
      fun () -> bool (RC.dtype_of_key (key ~impl:"rbg" 0) = kd (Some "rbg")) );
    ( "key_dtype_eq_mixed",
      fun () -> bool (RC.dtype_of_key (key ~impl:"rbg" 0) = kd None) );
    ("key_repr_0", fun () -> krepr (key 0));
    ("key_repr_42", fun () -> krepr (key 42));
    ("key_repr_12345", fun () -> krepr (key 12345));
    ("key_repr_split3", fun () -> krepr (RC.split (key 0) 3));
    ("key_repr_split_2x2", fun () -> krepr (split_shape (key 0) [| 2; 2 |]));
    ("key_str_split2", fun () -> krepr (RC.split (key 0) 2));
    ("key_repr_fold_in", fun () -> krepr (RC.fold_in (key 0) (seed 7)));
    ("key_repr_rbg1", fun () -> krepr (key ~impl:"rbg" 1));
    ("key_repr_urbg1", fun () -> krepr (key ~impl:"unsafe_rbg" 1));
    ("key_repr_split_rbg", fun () -> krepr (RC.split (key ~impl:"rbg" 3) 2));
    ( "key_repr_split_urbg",
      fun () -> krepr (RC.split (key ~impl:"unsafe_rbg" 3) 2) );
    ( "key_shape_split3",
      fun () ->
        Text
          (match RC.key_shape (RC.split (key 0) 3) with
          | [| d |] -> Printf.sprintf "(%d,)" d
          | _ -> "?") );
    ( "key_itemsize_fry",
      fun () -> Text (string_of_int (RC.key_itemsize (key 0))) );
    ( "key_itemsize_rbg",
      fun () -> Text (string_of_int (RC.key_itemsize (key ~impl:"rbg" 0))) );
    ( "key_nbytes_split3",
      fun () -> Text (string_of_int (RC.key_nbytes (RC.split (key 0) 3))) );
    ( "key_nbytes_urbg_split_2x2",
      fun () ->
        Text
          (string_of_int
             (RC.key_nbytes (split_shape (key ~impl:"unsafe_rbg" 0) [| 2; 2 |])))
    );
    ( "key_size_split_2x2",
      fun () ->
        Text (string_of_int (RC.key_size (split_shape (key 0) [| 2; 2 |]))) );
    ( "key_ndim_split_2x2",
      fun () ->
        Text (string_of_int (RC.key_ndim (split_shape (key 0) [| 2; 2 |]))) );
    ("key_ndim_scalar", fun () -> Text (string_of_int (RC.key_ndim (key 0))));
    ("typeof_key", fun () -> ktype (key 0));
    ("typeof_split3", fun () -> ktype (RC.split (key 0) 3));
    ("typeof_split_2x2", fun () -> ktype (split_shape (key 0) [| 2; 2 |]));
    ("typeof_rbg", fun () -> ktype (key ~impl:"rbg" 0));
    ("typeof_urbg", fun () -> ktype (key ~impl:"unsafe_rbg" 0));
    ("typeof_f32_2x3", fun () -> typeof (f32_2x3 ()));
    ("typeof_weak_float", fun () -> typeof (fweak 2.0));
    ("typeof_weak_int", fun () -> typeof (iweak 2));
    ( "typeof_weak_complex",
      fun () ->
        typeof (AC.asarray (JU.py_complex { Complex.re = 0.0; im = 1.0 })) );
    ( "typeof_bool",
      fun () -> typeof (AC.asarray (Ojax.Core.py_scalar D.Bool 1.0)) );
    ( "typeof_bf16_empty",
      fun () ->
        typeof (Ojax.Numpy.Array_creation.zeros ~dtype:D.Bf16 [| 0; 2 |]) );
    ( "typeof_zeros_2x0x3",
      fun () -> typeof (Ojax.Numpy.Array_creation.zeros [| 2; 0; 3 |]) );
    ( "typeof_arange5",
      fun () -> typeof (NL.arange ~dtype:(Ojax.Dtypes.default_int_dtype ()) 5.0)
    );
    ( "typeof_weak_product",
      fun () -> typeof (UF.multiply (fweak 2.0) (JU.py_int 2)) );
    ( "typeof_strong_sum",
      fun () ->
        typeof
          (UF.add (Ojax.Numpy.Array_creation.ones [| 3 |]) (JU.py_float 1.0)) );
    ("typeof_int8_vec", fun () -> typeof (NL.arange ~dtype:D.I8 4.0));
    ("typeof_key_data", fun () -> typeof (RC.key_data (key 0)));
    ( "issub_key_prng_key",
      fun () -> sub (kcls (RC.dtype_of_key (key 0))) NL.Prng_key_class );
    ( "issub_key_extended",
      fun () -> sub (kcls (RC.dtype_of_key (key 0))) NL.Extended );
    ( "issub_key_generic",
      fun () -> sub (kcls (RC.dtype_of_key (key 0))) NL.Generic );
    ( "issub_key_number",
      fun () -> sub (kcls (RC.dtype_of_key (key 0))) NL.Number );
    ( "issub_key_uint32",
      fun () -> sub (kcls (RC.dtype_of_key (key 0))) (NL.Cdtype D.Uint32) );
    ("issub_key_same", fun () -> sub (kcls (kd None)) (kcls (kd None)));
    ( "issub_key_rbg_fry",
      fun () -> sub (kcls (kd (Some "rbg"))) (kcls (kd None)) );
    ( "issub_key_rbg_prng_key",
      fun () -> sub (kcls (kd (Some "rbg"))) NL.Prng_key_class );
    ( "issub_key_urbg_extended",
      fun () -> sub (kcls (kd (Some "unsafe_rbg"))) NL.Extended );
    ("issub_float32_prng_key", fun () -> sub (NL.Cdtype D.F32) NL.Prng_key_class);
    ( "issub_uint32_prng_key",
      fun () -> sub (NL.Cdtype D.Uint32) NL.Prng_key_class );
    ("issub_prng_key_extended", fun () -> sub NL.Prng_key_class NL.Extended);
    ("issub_prng_key_key", fun () -> sub NL.Prng_key_class (kcls (kd None)));
    ("wrap_repr_rbg", fun () -> krepr (wrap ~impl:"rbg" (u32 4)));
    ( "wrap_repr_urbg_2x4",
      fun () -> krepr (wrap ~impl:"unsafe_rbg" (NL.reshape (u32 8) [| 2; 4 |]))
    );
    ("wrap_repr_fry", fun () -> krepr (wrap (u32 2)));
    ( "wrap_repr_fry_3x2",
      fun () ->
        krepr (wrap ~impl:"threefry2x32" (NL.reshape (u32 6) [| 3; 2 |])) );
    ("wrap_impl_rbg", fun () -> Text (RC.key_impl (wrap ~impl:"rbg" (u32 4))));
    ( "wrap_impl_urbg",
      fun () -> Text (RC.key_impl (wrap ~impl:"unsafe_rbg" (u32 4))) );
    ( "wrap_dtype_urbg",
      fun () -> Text (kname (RC.dtype_of_key (wrap ~impl:"unsafe_rbg" (u32 4))))
    );
    ( "wrap_typeof_rbg",
      fun () -> ktype (wrap ~impl:"rbg" (NL.reshape (u32 8) [| 2; 4 |])) );
    ( "wrap_split_urbg",
      fun () -> krepr (RC.split (wrap ~impl:"unsafe_rbg" (u32 4)) 2) );
    ("wrap_split_rbg", fun () -> krepr (RC.split (wrap ~impl:"rbg" (u32 4)) 2));
    ( "wrap_err_ndim",
      fun () -> krepr (wrap ~impl:"rbg" (arr D.Uint32 [||] [| 3.0 |])) );
    ( "wrap_err_int32",
      fun () -> krepr (wrap ~impl:"rbg" (NL.arange ~dtype:D.I32 4.0)) );
    ( "wrap_err_float32",
      fun () -> krepr (wrap ~impl:"unsafe_rbg" (NL.arange ~dtype:D.F32 4.0)) );
    ("wrap_err_shape_rbg", fun () -> krepr (wrap ~impl:"rbg" (u32 2)));
    ("wrap_err_shape_fry", fun () -> krepr (wrap (u32 4)));
    ("repr_weak_float", fun () -> repr (fweak 2.0));
    ("repr_weak_int", fun () -> repr (iweak 3));
    ("repr_weak_neg_zero", fun () -> repr (fweak (-0.0)));
    ( "repr_weak_complex",
      fun () -> repr (AC.asarray (JU.py_complex { Complex.re = 1.5; im = 2.0 }))
    );
    ("repr_bool", fun () -> repr (AC.asarray (Ojax.Core.py_scalar D.Bool 1.0)));
    ("repr_strong_float", fun () -> repr (arr D.F32 [||] [| 2.0 |]));
    ("repr_weak_product", fun () -> repr (UF.multiply (fweak 2.0) (JU.py_int 3)));
    ("repr_weak_bcast12", fun () -> repr (L.broadcast (fweak 1.5) [| 12 |]));
    ("repr_weak_bcast30", fun () -> repr (L.broadcast (fweak 1.5) [| 30 |]));
    ( "repr_weak_bcast_int_2x3",
      fun () -> repr (L.broadcast (iweak 2) [| 2; 3 |]) );
    ( "repr_weak_bcast_int17",
      fun () -> repr (L.broadcast (iweak 123456) [| 17 |]) );
    ("repr_weak_empty", fun () -> repr (L.broadcast (fweak 2.0) [| 0 |]));
    ( "repr_strong_vec",
      fun () ->
        repr
          (AC.asarray
             (arr (Ojax.Dtypes.default_float_dtype ()) [| 2 |] [| 1.0; 2.0 |]))
    );
    ( "repr_bf16_special",
      fun () ->
        repr
          (arr D.Bf16 [| 7 |]
             [|
               1.0; -0.0; Float.nan; Float.infinity; 1234567.0; 1e-7; 3.140625;
             |]) );
    ( "repr_bf16_neg_nan",
      fun () -> repr (arr D.Bf16 [| 2 |] [| -.Float.nan; Float.neg_infinity |])
    );
    ( "repr_bf16_wrap",
      fun () ->
        repr
          (UF.divide
             (NL.reshape (NL.arange ~dtype:D.Bf16 40.0) [| 2; 20 |])
             (JU.py_int 3)) );
    ( "repr_f8_e4m3fn",
      fun () -> repr (arr D.F8_e4m3fn [| 3 |] [| 1.0; 0.5; -448.0 |]) );
    ("repr_f8_e5m2", fun () -> repr (arr D.F8_e5m2 [| 3 |] [| 1.0; 0.1; -3.0 |]));
    ("repr_int4", fun () -> repr (arr D.I4 [| 3 |] [| 1.0; -3.0; 7.0 |]));
    ( "repr_uint4_2x2",
      fun () -> repr (arr D.Uint4 [| 2; 2 |] [| 1.0; 15.0; 10.0; 0.0 |]) );
    ("repr_bf16_scalar", fun () -> repr (arr D.Bf16 [||] [| 2.5 |]));
    ( "repr_bf16_third",
      fun () -> repr (UF.divide (arr D.Bf16 [||] [| 1.0 |]) (JU.py_int 3)) );
    ( "str_bf16_vec",
      fun () ->
        Text
          (Ojax.Numpy.Extras.array_str_jax (arr D.Bf16 [| 2 |] [| 2.5; 1.0 |]))
    );
    ( "str_bf16_scalar_third",
      fun () ->
        Text
          (Ojax.Numpy.Extras.array_str_jax
             (UF.divide (arr D.Bf16 [||] [| 1.0 |]) (JU.py_int 3))) );
    ("tolist_bf16", fun () -> tolist (arr D.Bf16 [| 2 |] [| 2.5; 1.0 /. 3.0 |]));
    ("item_f32_flat1", fun () -> item ~args:[ 1 ] (f32_2x3 ()));
    ("item_f32_neg1", fun () -> item ~args:[ -1 ] (f32_2x3 ()));
    ("item_f32_1_2", fun () -> item ~args:[ 1; 2 ] (f32_2x3 ()));
    ("item_f32_0_neg1", fun () -> item ~args:[ 0; -1 ] (f32_2x3 ()));
    ("item_err_size", fun () -> item (f32_2x3 ()));
    ("item_err_flat_oob", fun () -> item ~args:[ 10 ] (f32_2x3 ()));
    ("item_err_axis_oob", fun () -> item ~args:[ 2; 0 ] (f32_2x3 ()));
    ("item_err_count", fun () -> item ~args:[ 1; 2; 0 ] (f32_2x3 ()));
    ("item_f32_tenth", fun () -> item (arr D.F32 [||] [| 0.1 |]));
    ("item_weak_float", fun () -> item (fweak 0.1));
    ("item_int", fun () -> item (iweak 3));
    ("item_bool", fun () -> item (AC.asarray (Ojax.Core.py_scalar D.Bool 0.0)));
    ( "item_complex",
      fun () -> item (AC.asarray (JU.py_complex { Complex.re = 1.0; im = 2.0 }))
    );
    ( "item_complex_neg_zero",
      fun () -> item (L.reshape (carr D.Complex64 [| (1.5, -0.0) |]) [||]) );
    ( "item_complex_pure_imag",
      fun () -> item (L.reshape (carr D.Complex64 [| (0.0, 2.5) |]) [||]) );
    ("item_uint32_max", fun () -> item (arr D.Uint32 [||] [| 4294967295.0 |]));
    ("item_int8_neg", fun () -> item (arr D.I8 [||] [| -7.0 |]));
    ("item_f32_big", fun () -> item (arr D.F32 [||] [| 1e20 |]));
    ("item_f32_small", fun () -> item (arr D.F32 [||] [| 1e-5 |]));
    ("item_f32_1e16", fun () -> item (arr D.F32 [||] [| 1e16 |]));
    ("item_f32_123456789", fun () -> item (arr D.F32 [||] [| 123456789.0 |]));
    ("item_f32_inf", fun () -> item (arr D.F32 [||] [| Float.neg_infinity |]));
    ("item_f32_nan", fun () -> item (arr D.F32 [||] [| Float.nan |]));
    ("item_f32_neg_zero", fun () -> item (arr D.F32 [||] [| -0.0 |]));
    ("item_f16", fun () -> item (arr D.F16 [||] [| 0.1 |]));
    ("item_bf16", fun () -> item (arr D.Bf16 [||] [| 0.1 |]));
    ("item_f64", fun () -> item (arr D.F64 [||] [| 0.1 |]));
    ( "item_int64_big",
      fun () -> item (AC.asarray ~dtype:D.I64 (JU.py_int (1 lsl 40))) );
    ("tolist_f32_2x3", fun () -> tolist (f32_2x3 ()));
    ( "tolist_int_vec",
      fun () -> tolist (NL.arange ~dtype:(Ojax.Dtypes.default_int_dtype ()) 4.0)
    );
    ("tolist_bool", fun () -> tolist (arr D.Bool [| 2 |] [| 1.0; 0.0 |]));
    ( "tolist_complex",
      fun () ->
        tolist
          (carr
             (if Ojax.Config.value Ojax.Config.enable_x64 then D.Complex128
              else D.Complex64)
             [| (1.0, 2.0); (-0.0, -3.0); (0.5, 0.0) |]) );
    ("tolist_scalar", fun () -> tolist (fweak 2.5));
    ("tolist_empty", fun () -> tolist (Ojax.Numpy.Array_creation.zeros [| 0 |]));
    ( "tolist_empty_2x0",
      fun () -> tolist (Ojax.Numpy.Array_creation.zeros [| 2; 0 |]) );
    ( "tolist_f32_tenths",
      fun () -> tolist (UF.divide (NL.arange ~dtype:D.F32 3.0) (JU.py_int 10))
    );
    ("tolist_f16", fun () -> tolist (arr D.F16 [| 2 |] [| 0.1; 2.0 |]));
    ("tolist_uint8", fun () -> tolist (arr D.Uint8 [| 2 |] [| 0.0; 255.0 |]));
    ("tobytes_f32_2x3", fun () -> Text (hex (AM.tobytes (f32_2x3 ()))));
    ( "tobytes_int8",
      fun () ->
        Text (hex (AM.tobytes (arr D.I8 [| 3 |] [| -1.0; 2.0; 127.0 |]))) );
    ( "tobytes_bool",
      fun () -> Text (hex (AM.tobytes (arr D.Bool [| 3 |] [| 1.0; 0.0; 1.0 |])))
    );
    ( "tobytes_complex64",
      fun () ->
        Text
          (hex (AM.tobytes (carr D.Complex64 [| (1.0, 2.0); (-0.0, -0.5) |])))
    );
    ( "tobytes_uint16",
      fun () -> Text (hex (AM.tobytes (NL.arange ~dtype:D.Uint16 3.0))) );
    ("tobytes_scalar", fun () -> Text (hex (AM.tobytes (fweak 1.5))));
    ( "tobytes_empty",
      fun () ->
        Text (hex (AM.tobytes (Ojax.Numpy.Array_creation.zeros [| 0; 3 |]))) );
    ("itemsize_f32", fun () -> Text (string_of_int (AM.itemsize (f32_2x3 ()))));
    ( "itemsize_bool",
      fun () ->
        Text
          (string_of_int
             (AM.itemsize (AC.asarray (Ojax.Core.py_scalar D.Bool 1.0)))) );
    ( "itemsize_int8",
      fun () -> Text (string_of_int (AM.itemsize (NL.arange ~dtype:D.I8 3.0)))
    );
    ( "itemsize_complex64",
      fun () ->
        Text (string_of_int (AM.itemsize (carr D.Complex64 [| (0.0, 1.0) |])))
    );
    ( "itemsize_f16",
      fun () ->
        Text
          (string_of_int
             (AM.itemsize
                (Ojax.Numpy.Array_creation.zeros ~dtype:D.F16 [| 2 |]))) );
    ( "itemsize_bf16",
      fun () ->
        Text
          (string_of_int
             (AM.itemsize
                (Ojax.Numpy.Array_creation.zeros ~dtype:D.Bf16 [| 2 |]))) );
    ( "itemsize_default_int",
      fun () ->
        Text
          (string_of_int
             (AM.itemsize
                (NL.arange ~dtype:(Ojax.Dtypes.default_int_dtype ()) 3.0))) );
    ( "itemsize_weak_float",
      fun () -> Text (string_of_int (AM.itemsize (fweak 1.0))) );
    ("nbytes_f32", fun () -> Text (string_of_int (AM.nbytes (f32_2x3 ()))));
    ( "nbytes_bool",
      fun () ->
        Text
          (string_of_int (AM.nbytes (arr D.Bool [| 3 |] [| 1.0; 0.0; 1.0 |])))
    );
    ( "nbytes_complex64",
      fun () ->
        Text
          (string_of_int
             (AM.nbytes
                (Ojax.Numpy.Array_creation.zeros ~dtype:D.Complex64 [| 2; 2 |])))
    );
    ( "nbytes_empty",
      fun () ->
        Text
          (string_of_int
             (AM.nbytes (Ojax.Numpy.Array_creation.zeros [| 0; 3 |]))) );
    ("nbytes_scalar", fun () -> Text (string_of_int (AM.nbytes (fweak 1.0))));
    ( "nbytes_default_int",
      fun () ->
        Text
          (string_of_int
             (AM.nbytes
                (NL.arange ~dtype:(Ojax.Dtypes.default_int_dtype ()) 5.0))) );
    ("flat_err", fun () -> AM.flat (f32_2x3 ()));
    ("view_err_0d", fun () -> Value (AM.view (arr D.F32 [||] [| 2.0 |]) D.I16));
    ( "view_err_divisor",
      fun () -> Value (AM.view (NL.arange ~dtype:D.I8 3.0) D.I32) );
    ("view_err_complex", fun () -> Value (AM.view (f32_2x3 ()) D.Complex64));
    ("view_f32_int32", fun () -> Value (AM.view (f32_2x3 ()) D.I32));
    ("view_f32_uint32", fun () -> Value (AM.view (f32_2x3 ()) D.Uint32));
    ("view_f32_uint8", fun () -> Value (AM.view (f32_2x3 ()) D.Uint8));
    ("view_f32_uint16", fun () -> Value (AM.view (f32_2x3 ()) D.Uint16));
    ("view_f32_float16", fun () -> Value (AM.view (f32_2x3 ()) D.F16));
    ("view_f32_int16", fun () -> Value (AM.view (f32_2x3 ()) D.I16));
    ("view_f32_bool", fun () -> Value (AM.view (f32_2x3 ()) D.Bool));
    ("view_f32_same", fun () -> Value (AM.view (f32_2x3 ()) D.F32));
    ( "view_int8_bool",
      fun () -> Value (AM.view (NL.arange ~dtype:D.I8 4.0) D.Bool) );
    ( "view_int8_int16",
      fun () -> Value (AM.view (NL.arange ~dtype:D.I8 4.0) D.I16) );
    ( "view_int8_int32",
      fun () -> Value (AM.view (NL.arange ~dtype:D.I8 4.0) D.I32) );
    ( "view_int8_float32",
      fun () ->
        Value (AM.view (arr D.I8 [| 4 |] [| 0.; 0.; -128.; 63. |]) D.F32) );
    ( "view_uint8_float32",
      fun () ->
        Value
          (AM.view
             (arr D.Uint8 [| 8 |] [| 0.; 0.; 128.; 63.; 0.; 0.; 0.; 64. |])
             D.F32) );
    ( "view_bool_uint8",
      fun () ->
        Value (AM.view (arr D.Bool [| 4 |] [| 1.; 0.; 1.; 0. |]) D.Uint8) );
    ( "view_bool_int16",
      fun () -> Value (AM.view (arr D.Bool [| 4 |] [| 1.; 0.; 1.; 0. |]) D.I16)
    );
    ( "view_c64_float32",
      fun () ->
        Value (AM.view (carr D.Complex64 [| (1., 2.); (3., -4.) |]) D.F32) );
    ( "view_c64_int32",
      fun () ->
        Value (AM.view (carr D.Complex64 [| (1., 2.); (3., -4.) |]) D.I32) );
    ( "view_c64_uint8",
      fun () ->
        Value (AM.view (carr D.Complex64 [| (1., 2.); (3., -4.) |]) D.Uint8) );
    ( "view_f32_c64",
      fun () -> Value (AM.view (NL.arange ~dtype:D.F32 4.0) D.Complex64) );
    ( "view_int32_c64",
      fun () -> Value (AM.view (NL.arange ~dtype:D.I32 4.0) D.Complex64) );
    ( "view_f32_c64_special",
      fun () ->
        Value
          (AM.view
             (arr D.F32 [| 4 |] [| 1.0; Float.infinity; -0.0; Float.nan |])
             D.Complex64) );
    ( "view_scalar_int32",
      fun () -> Value (AM.view (arr D.F32 [||] [| 2.0 |]) D.I32) );
    ("view_scalar_uint32_weak", fun () -> Value (AM.view (fweak 2.0) D.Uint32));
    ( "view_f64_int64",
      fun () -> Value (AM.view (NL.arange ~dtype:D.F64 3.0) D.I64) );
    ( "view_bf16_uint16",
      fun () -> Value (AM.view (NL.arange ~dtype:D.Bf16 4.0) D.Uint16) );
    ( "view_bf16_int8",
      fun () -> Value (AM.view (NL.arange ~dtype:D.Bf16 4.0) D.I8) );
    ( "view_f16_bf16",
      fun () -> Value (AM.view (arr D.F16 [| 2 |] [| 1.0; -2.5 |]) D.Bf16) );
    ( "view_3d_int16_int8",
      fun () ->
        Value
          (AM.view
             (NL.reshape (NL.arange ~dtype:D.I16 12.0) [| 1; 2; 6 |])
             D.I8) );
    ( "view_3d_int8_int16",
      fun () ->
        Value
          (AM.view (NL.reshape (NL.arange ~dtype:D.I8 8.0) [| 1; 2; 4 |]) D.I16)
    );
    ("byteswap_f32", fun () -> Value (AM.byteswap (f32_2x3 ())));
    ( "byteswap_uint16",
      fun () -> Value (AM.byteswap (NL.arange ~dtype:D.Uint16 4.0)) );
    ( "byteswap_int32",
      fun () -> Value (AM.byteswap (NL.arange ~dtype:D.I32 5.0)) );
    ( "byteswap_bool",
      fun () -> Value (AM.byteswap (arr D.Bool [| 2 |] [| 1.; 0. |])) );
    ( "byteswap_int8",
      fun () -> Value (AM.byteswap (arr D.I8 [| 3 |] [| -1.; 2.; 3. |])) );
    ( "byteswap_c64",
      fun () -> Value (AM.byteswap (carr D.Complex64 [| (1., 2.); (3., -4.) |]))
    );
    ( "byteswap_f16",
      fun () -> Value (AM.byteswap (arr D.F16 [| 3 |] [| 1.0; -2.0; 0.5 |])) );
    ( "byteswap_bf16",
      fun () -> Value (AM.byteswap (arr D.Bf16 [| 3 |] [| 1.0; -2.0; 0.5 |])) );
    ( "byteswap_f64",
      fun () -> Value (AM.byteswap (arr D.F64 [| 2 |] [| 1.0; -2.0 |])) );
    ("byteswap_scalar", fun () -> Value (AM.byteswap (arr D.I32 [||] [| 7.0 |])));
    ( "byteswap_twice",
      fun () -> Value (AM.byteswap (AM.byteswap (NL.arange ~dtype:D.I32 5.0)))
    );
    ( "wrap_bits_rbg",
      fun () -> Value (RC.bits (wrap ~impl:"rbg" (u32 4)) ~shape:[| 3 |]) );
    ( "wrap_bits_urbg",
      fun () -> Value (RC.bits (wrap ~impl:"unsafe_rbg" (u32 4)) ~shape:[| 3 |])
    );
    ( "wrap_key_data_rbg",
      fun () -> Value (RC.key_data (wrap ~impl:"rbg" (u32 4))) );
    ( "wrap_uniform_urbg",
      fun () ->
        Value
          (RC.uniform
             (wrap ~impl:"unsafe_rbg" (UF.add (u32 4) (JU.py_int 9)))
             ~shape:[| 4 |] ~minval:0.0 ~maxval:1.0) );
    ("key_data_split3", fun () -> Value (RC.key_data (RC.split (key 0) 3)));
  ]

let traced : (string * (Ojax.Types.value -> Ojax.Types.value)) list =
  [
    ("jit_view_f32_uint8", fun x -> AM.view x D.Uint8);
    ("jit_view_f32_int32", fun x -> AM.view x D.I32);
    ("jit_view_f32_float16", fun x -> AM.view x D.F16);
    ("jit_view_int8_int32", fun x -> AM.view x D.I32);
    ("jit_view_bool_int16", fun x -> AM.view x D.I16);
    ("jit_view_f32_bool", fun x -> AM.view x D.Bool);
    ("jit_view_f32_c64", fun x -> AM.view x D.Complex64);
    ("jit_view_scalar", fun x -> AM.view x D.I32);
    ("jit_byteswap_f32", AM.byteswap);
    ("jit_byteswap_uint16", AM.byteswap);
    ("jit_byteswap_int8", AM.byteswap);
    ("jit_byteswap_c64", AM.byteswap);
    ("jit_bitcast_f32_uint8", fun x -> L.bitcast_convert_type x D.Uint8);
    ("jit_bitcast_uint8_f32", fun x -> L.bitcast_convert_type x D.F32);
    ("jit_bitcast_err", fun x -> L.bitcast_convert_type x D.F32);
    ("jit_key_split", fun k -> RC.split (Ojax.Key_reuse.mark_typed k) 2);
    ( "jit_key_fold_in",
      fun k -> RC.fold_in (Ojax.Key_reuse.mark_typed k) (seed 3) );
    ( "jit_key_bits",
      fun k -> RC.bits (Ojax.Key_reuse.mark_typed k) ~shape:[| 3 |] );
    ("jit_key_data", fun k -> RC.key_data (Ojax.Key_reuse.mark_typed k));
    ( "jit_key_uniform",
      fun k ->
        RC.uniform
          (Ojax.Key_reuse.mark_typed k)
          ~shape:[| 2 |] ~minval:0.0 ~maxval:1.0 );
    ("jit_wrap_key_data", fun x -> RC.wrap_key_data x);
    ("jit_wrap_key_data_3x2", fun x -> RC.wrap_key_data x);
    ("jit_key_from_seed", fun s -> RC.key s);
    ("jit_key_split3_data", fun k -> RC.key_data (Ojax.Key_reuse.mark_typed k));
    ( "jit_key_split3_bits",
      fun k -> RC.bits (Ojax.Key_reuse.mark_typed k) ~shape:[| 2 |] );
    ( "jit_key_normal",
      fun k -> RC.normal (Ojax.Key_reuse.mark_typed k) ~shape:[| 3 |] );
    ( "jit_key_split_split",
      fun k ->
        RC.split
          (L.squeeze
             (L.slice
                (RC.split (Ojax.Key_reuse.mark_typed k) 2)
                [| 0; 0 |] [| 1; 2 |])
             [| 0 |])
          2 );
    ("jit_rbg_bits", fun k -> RC.bits k ~shape:[| 2 |]);
    ("jit_rbg_split", fun k -> RC.split k 2);
    ("jit_urbg_split", fun k -> RC.split k 2);
    ("jit_urbg_data", fun k -> RC.key_data k);
  ]

let aval_of_arg a : Ojax.Types.aval =
  match U.to_list a with
  | [ `String (("key<fry>" | "key<rbg>" | "key<urbg>") as k); shape ] ->
      {
        dtype = D.Uint32;
        shape =
          Array.append
            (Array.of_list (List.map U.to_int (U.to_list shape)))
            [| (if k = "key<fry>" then 2 else 4) |];
        weak_type = k = "key<urbg>";
      }
  | [ dt; shape ] ->
      {
        dtype =
          (match D.of_numpy_name (U.to_string dt) with
          | Some d -> d
          | None -> failwith "keys_arrays: dtype");
        shape = Array.of_list (List.map U.to_int (U.to_list shape));
        weak_type = false;
      }
  | _ -> failwith "keys_arrays: arg"

let read_rows path =
  let ic = open_in path in
  let rec loop acc =
    match input_line ic with
    | line -> loop (Yojson.Safe.from_string line :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  loop []

let str_opt row field =
  match U.member field row with `String s -> Some s | _ -> None

let host = function
  | Ojax.Types.Concrete nd -> nd
  | Ojax.Types.Device b -> Ojax.Pjrt.Buffer.to_host b
  | Ojax.Types.Tracer _ -> failwith "keys_arrays: tracer result"

let esc = String.escaped

let address_proof =
  "(a) nondeterministic and (c) Python only: JAX ends the text with \
   impl=PRNGImpl(...), whose repr prints every impl function as <function \
   _rbg_seed at 0x...>, a CPython object address; two runs of the same call \
   printed 0xe4e74b6d56c0 and 0xe2a6c46616c0 for _rbg_random_bits. ojax prints \
   the same text without the addresses."

let defects =
  List.concat_map
    (fun mode ->
      List.map
        (fun row -> (("keys_arrays:" ^ mode, row), address_proof))
        [ "wrap_err_shape_rbg"; "wrap_err_shape_fry" ])
    [ "x64_off"; "x64_on" ]

let error_detail e =
  match Ojax.Py_exn.class_name e with
  | Some c -> c ^ ": " ^ Error_types.message_of e
  | None -> Error_types.ojax_class e ^ ": " ^ Error_types.message_of e

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log ?(name = id) jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(esc jax ^ "\t" ^ esc ojax)
      ~suite name 1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical
       else if List.mem_assoc (suite, name) defects then "jax_defect"
       else "differ")
  in
  let expected_error =
    match (str_opt row "error_class", str_opt row "message") with
    | Some c, Some m -> Some (c ^ ": " ^ m)
    | _ -> None
  in
  let got =
    match List.assoc_opt id cases with
    | None -> Error "no_binding"
    | Some f -> ( try Ok (f ()) with e -> Error (error_detail e))
  in
  match U.member "args" row with
  | `List args -> (
      let avals = List.map aval_of_arg args in
      let closed =
        match List.assoc_opt id traced with
        | None -> Error "no_binding"
        | Some f -> (
            try
              Ok
                (Ojax.Jaxpr.make_jaxpr avals (function
                  | [ x ] -> [ f x ]
                  | _ -> failwith "keys_arrays: arity"))
            with e -> Error (error_detail e))
      in
      match (expected_error, closed) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            ojax;
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            ojax
      | None, Ok c ->
          let text f = try f c with e -> error_detail e in
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            (text Ojax.Pretty_printer.closed_jaxpr_to_jax_string);
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            (text (Ojax.Stablehlo.Emit.emit_jax_text ~name:"f")))
  | _ -> (
      match (expected_error, got) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax -> (
          log (Option.value (str_opt row "text") ~default:"array") ojax;
          match str_opt row "repr" with
          | Some r -> log ~name:(id ^ ":repr") r ojax
          | None -> ())
      | None, Ok (Text t) ->
          log (Option.value (str_opt row "text") ~default:"array") t
      | None, Ok (Value v) -> (
          match
            (str_opt row "dtype", str_opt row "hex", str_opt row "repr")
          with
          | Some dt, Some hx, Some r ->
              let nd = host v in
              let shape =
                List.map U.to_int (U.to_list (U.member "shape" row))
              in
              let shape_text s = String.concat "," (List.map string_of_int s) in
              let raw = Npz.raw_of_nd nd in
              log
                (dt ^ "[" ^ shape_text shape ^ "]" ^ hx)
                (raw.Npz.raw_dtype ^ "["
                ^ shape_text (Array.to_list (Nd.shape nd))
                ^ "]" ^ hex raw.Npz.bytes);
              log ~name:(id ^ ":repr") r (Ojax.Numpy.Extras.repr v)
          | _ -> log (Option.value (str_opt row "text") ~default:"text") "array"
          ))

let path ~goldens_root =
  Filename.concat (Filename.concat goldens_root name) "cases.jsonl"

let available ~goldens_root = Sys.file_exists (path ~goldens_root)

let suites ~goldens_root =
  if not (available ~goldens_root) then []
  else
    let rows = read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              let x64 = U.to_bool (U.member "x64" row) in
              if mode = "x64_on" = x64 then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
