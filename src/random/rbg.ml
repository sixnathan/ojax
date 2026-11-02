open Types

let is_unsafe k = Prng.is_rbg_key k && (Ojax__Core.get_aval k).weak_type
