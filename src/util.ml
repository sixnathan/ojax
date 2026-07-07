let safe_map f xs = List.map f xs
let safe_map2 f xs ys = List.map2 f xs ys
let foreach f xs = List.iter f xs
let safe_zip xs ys = List.map2 (fun a b -> (a, b)) xs ys
let unzip2 xys = List.split xys

let unzip3 xyzs =
  let rec go acc = function
    | [] ->
        let xs, ys, zs = acc in
        (List.rev xs, List.rev ys, List.rev zs)
    | (x, y, z) :: rest ->
        let xs, ys, zs = acc in
        go (x :: xs, y :: ys, z :: zs) rest
  in
  go ([], [], []) xyzs

let subvals lst replace =
  let a = Array.of_list lst in
  List.iter (fun (i, v) -> a.(i) <- v) replace;
  Array.to_list a

let split_list args ns =
  let rec go args = function
    | [] -> [ args ]
    | n :: ns ->
        let head, tail =
          ( List.filteri (fun i _ -> i < n) args,
            List.filteri (fun i _ -> i >= n) args )
        in
        head :: go tail ns
  in
  go args ns

let split_half lst =
  let n = List.length lst in
  if n mod 2 <> 0 then invalid_arg "split_half: odd length";
  match split_list lst [ n / 2 ] with
  | [ first; second ] -> (first, second)
  | _ -> invalid_arg "split_half"

let partition_list bs l =
  if List.length bs <> List.length l then
    invalid_arg "partition_list: length mismatch";
  let rec go falses trues = function
    | [], [] -> (List.rev falses, List.rev trues)
    | b :: bs, x :: xs ->
        if b then go falses (x :: trues) (bs, xs)
        else go (x :: falses) trues (bs, xs)
    | _ -> invalid_arg "partition_list"
  in
  go [] [] (bs, l)

let merge_lists bs l0 l1 =
  let rec go acc bs l0 l1 =
    match (bs, l0, l1) with
    | [], [], [] -> List.rev acc
    | true :: bs, l0, x :: l1 -> go (x :: acc) bs l0 l1
    | false :: bs, x :: l0, l1 -> go (x :: acc) bs l0 l1
    | _ -> invalid_arg "merge_lists: length mismatch"
  in
  go [] bs l0 l1

let subs_list subs src base =
  let src = Array.of_list src in
  let rec go subs base =
    match subs with
    | [] -> if base = [] then [] else invalid_arg "subs_list: extra base"
    | Some i :: subs -> src.(i) :: go subs base
    | None :: subs -> (
        match base with
        | b :: base -> b :: go subs base
        | [] -> invalid_arg "subs_list: base exhausted")
  in
  go subs base

let subs_list2 subs1 subs2 src1 src2 base =
  if List.length subs1 <> List.length subs2 then
    invalid_arg "subs_list2: length mismatch";
  let src1 = Array.of_list src1 in
  let src2 = Array.of_list src2 in
  let rec go subs1 subs2 base =
    match (subs1, subs2) with
    | [], [] -> if base = [] then [] else invalid_arg "subs_list2: extra base"
    | Some i :: subs1, _ :: subs2 -> src1.(i) :: go subs1 subs2 base
    | None :: subs1, Some i :: subs2 -> src2.(i) :: go subs1 subs2 base
    | None :: subs1, None :: subs2 -> (
        match base with
        | b :: base -> b :: go subs1 subs2 base
        | [] -> invalid_arg "subs_list2: base exhausted")
    | _ -> invalid_arg "subs_list2"
  in
  go subs1 subs2 base

let concatenate xs = List.concat xs
let flatten = concatenate

let unflatten xs ns =
  let rec go xs = function
    | [] -> if xs = [] then [] else invalid_arg "unflatten: leftover elements"
    | n :: ns ->
        let head, tail =
          ( List.filteri (fun i _ -> i < n) xs,
            List.filteri (fun i _ -> i >= n) xs )
        in
        if List.length head <> n then invalid_arg "unflatten: too few elements";
        head :: go tail ns
  in
  go xs ns
