module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RD = Numpy.Reductions
module AC = Numpy.Array_creation
module IX = Numpy.Indexing
open Dist_util

let invert_permutation i =
  let size = Array.fold_left ( * ) 1 (shape i) in
  let base = AC.zeros_like i in
  let vals = NL.arange ~dtype:(dtype i) (float_of_int size) in
  IX.put base i vals

let sem ?(axis = 0) ?(ddof = 1) a =
  let b = match promote_dtypes [ a ] with [ b ] -> b | _ -> assert false in
  let size = (shape b).(axis) in
  let denom = sc b (Float.sqrt (float_of_int size)) in
  UF.divide (RD.std ~axis:[| axis |] ~ddof b) denom

let mode ?axis:_ ?nan_policy:_ ?keepdims:_ _a =
  failwith "scipy.stats.mode: requires jnp.unique (deferred to M5)"

let rankdata ?method_:_ ?axis:_ ?nan_policy:_ _a =
  failwith "scipy.stats.rankdata: requires jnp.nonzero (deferred to M5)"
