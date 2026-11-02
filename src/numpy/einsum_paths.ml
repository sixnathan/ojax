type optimize =
  | Auto
  | Optimal
  | Greedy
  | Branch_all
  | Branch_2
  | Branch_1
  | Unoptimized
  | Path of int list list

let optimize_of_string s =
  match String.lowercase_ascii s with
  | "auto" -> Auto
  | "optimal" -> Optimal
  | "greedy" | "eager" | "opportunistic" -> Greedy
  | "branch-all" -> Branch_all
  | "branch-2" -> Branch_2
  | "branch-1" -> Branch_1
  | other ->
      Py_exn.key_error (Printf.sprintf "Path optimizer '%s' not found." other)

type contraction = {
  positions : int list;
  einsum_str : string;
  remaining : string list option;
  blas : string option;
}

type path_info = {
  contraction_list : contraction list;
  input_subscripts : string;
  output_subscript : string;
  eq : string;
  path : int list list;
  indices : string;
  scale_list : int list;
  naive_cost : int;
  opt_cost : int;
  speedup : float;
  size_list : int list;
  size_dict : (char * int) list;
  shapes : int array list;
  largest_intermediate : int;
}

let chars s = List.init (String.length s) (String.get s)
let of_chars cs = String.init (List.length cs) (List.nth cs)
let canon s = of_chars (List.sort_uniq Char.compare (chars s))
let mem s c = String.contains s c
let union a b = canon (a ^ b)
let inter a b = of_chars (List.filter (mem b) (chars a))
let diff a b = of_chars (List.filter (fun c -> not (mem b c)) (chars a))
let is_disjoint a b = inter a b = ""
let count_char s c = List.length (List.filter (Char.equal c) (chars s))

let size_of sizes s =
  List.fold_left (fun acc c -> acc * List.assoc c sizes) 1 (chars s)

let flop_count idx inner num_terms sizes =
  let factor = Stdlib.max 1 (num_terms - 1) + if inner then 1 else 0 in
  size_of sizes idx * factor

let ssa_to_linear ssa_path =
  let n =
    List.fold_left (fun acc s -> acc + List.length s) 0 ssa_path
    - List.length ssa_path + 1
  in
  let ids = ref (List.init n Fun.id) in
  let ssa = ref n in
  List.map
    (fun scon ->
      let position s =
        let rec go i = function
          | [] -> i
          | x :: rest -> if x >= s then i else go (i + 1) rest
        in
        go 0 !ids
      in
      let con = List.sort compare (List.map position scon) in
      ids := List.filteri (fun i _ -> not (List.mem i con)) !ids;
      ids := !ids @ [ !ssa ];
      incr ssa;
      con)
    ssa_path

let calc_k12_flops inputs output remaining i j sizes =
  let k1 = inputs.(i) and k2 = inputs.(j) in
  let either = union k1 k2 in
  let shared = inter k1 k2 in
  let keep =
    List.fold_left
      (fun acc r -> if r = i || r = j then acc else union acc inputs.(r))
      output remaining
  in
  (inter either keep, flop_count either (diff shared keep <> "") 2 sizes)

let pairs remaining =
  let rec go = function
    | [] -> []
    | x :: rest -> List.map (fun y -> (x, y)) rest @ go rest
  in
  go remaining

let extend inputs k12 = Array.append inputs [| k12 |]

let next_remaining remaining i j len =
  List.filter (fun r -> r <> i && r <> j) remaining @ [ len ]

let optimal inputs output sizes =
  let n = List.length inputs in
  let best_flops = ref infinity in
  let best_path = ref [ List.init n Fun.id ] in
  let cache = Hashtbl.create 64 in
  let rec iterate path remaining inputs flops =
    match remaining with
    | [ _ ] ->
        best_flops := float_of_int flops;
        best_path := List.rev path
    | _ ->
        List.iter
          (fun (i, j) ->
            let key = (inputs.(i), inputs.(j)) in
            let k12, flops12 =
              match Hashtbl.find_opt cache key with
              | Some r -> r
              | None ->
                  let r = calc_k12_flops inputs output remaining i j sizes in
                  Hashtbl.add cache key r;
                  r
            in
            let new_flops = flops + flops12 in
            if float_of_int new_flops < !best_flops then
              iterate ([ i; j ] :: path)
                (next_remaining remaining i j (Array.length inputs))
                (extend inputs k12) new_flops)
          (pairs remaining)
  in
  iterate [] (List.init n Fun.id) (Array.of_list inputs) 0;
  ssa_to_linear !best_path

let branch nbranch inputs output sizes =
  let n = List.length inputs in
  let best_flops = ref infinity and best_size = ref infinity in
  let best_path = ref [] in
  let progress = Hashtbl.create 16 in
  let cache = Hashtbl.create 64 in
  let rec iterate path inputs remaining flops size =
    match remaining with
    | [ _ ] ->
        best_flops := float_of_int flops;
        best_size := float_of_int size;
        best_path := List.rev path
    | _ ->
        let assess i j =
          let k1 = inputs.(i) and k2 = inputs.(j) in
          let k12, flops12 =
            match Hashtbl.find_opt cache (k1, k2) with
            | Some r -> r
            | None ->
                let r = calc_k12_flops inputs output remaining i j sizes in
                Hashtbl.add cache (k1, k2) r;
                r
          in
          let size12 = size_of sizes k12 in
          let new_flops = flops + flops12 in
          let new_size = Stdlib.max size size12 in
          let nf = float_of_int new_flops in
          let better =
            compare (nf, float_of_int new_size) (!best_flops, !best_size) < 0
          in
          if not better then None
          else
            let len = Array.length inputs in
            let seen =
              Option.value (Hashtbl.find_opt progress len) ~default:infinity
            in
            if nf < seen then Hashtbl.replace progress len nf;
            if nf >= seen && nf > 4.0 *. seen then None
            else
              let cost = size12 - size_of sizes k1 - size_of sizes k2 in
              Some ((cost, flops12, new_flops, new_size, (i, j)), k12)
        in
        let connected =
          List.filter_map
            (fun (i, j) ->
              if is_disjoint inputs.(i) inputs.(j) then None else assess i j)
            (pairs remaining)
        in
        let candidates =
          if connected <> [] then connected
          else List.filter_map (fun (i, j) -> assess i j) (pairs remaining)
        in
        let sorted = List.sort (fun (a, _) (b, _) -> compare a b) candidates in
        List.iteri
          (fun bi ((_, _, new_flops, new_size, (i, j)), k12) ->
            let allowed =
              match nbranch with None -> true | Some limit -> bi < limit
            in
            if allowed then
              iterate ([ i; j ] :: path) (extend inputs k12)
                (next_remaining remaining i j (Array.length inputs))
                new_flops new_size)
          sorted
  in
  iterate [] (Array.of_list inputs) (List.init n Fun.id) 0 0;
  ssa_to_linear !best_path

let greedy_ssa inputs output sizes =
  match inputs with
  | [ _ ] -> [ [ 0 ] ]
  | first :: rest ->
      let common = List.fold_left inter first rest in
      let output = union output common in
      let next_id = ref (List.length inputs) in
      let fresh () =
        let id = !next_id in
        incr next_id;
        id
      in
      let remaining = Hashtbl.create 16 in
      let ssa_path = ref [] in
      List.iteri
        (fun ssa_id key ->
          match Hashtbl.find_opt remaining key with
          | Some prior ->
              ssa_path := [ prior; ssa_id ] :: !ssa_path;
              Hashtbl.replace remaining key (fresh ())
          | None -> Hashtbl.replace remaining key ssa_id)
        inputs;
      let dim_keys = Hashtbl.create 16 in
      let keys_of dim =
        Option.value (Hashtbl.find_opt dim_keys dim) ~default:[]
      in
      let add_key dim key =
        if not (List.mem key (keys_of dim)) then
          Hashtbl.replace dim_keys dim (key :: keys_of dim)
      in
      let remove_key dim key =
        Hashtbl.replace dim_keys dim (List.filter (( <> ) key) (keys_of dim))
      in
      Hashtbl.iter
        (fun key _ ->
          List.iter (fun d -> add_key d key) (chars (diff key output)))
        remaining;
      let refs2 = ref "" and refs3 = ref "" in
      let update_refs dims =
        List.iter
          (fun d ->
            let count = List.length (keys_of d) in
            let drop r = r := diff !r (String.make 1 d) in
            let add r = r := union !r (String.make 1 d) in
            if count <= 1 then (
              drop refs2;
              drop refs3)
            else if count = 2 then (
              add refs2;
              drop refs3)
            else (
              add refs2;
              add refs3))
          dims
      in
      Hashtbl.iter
        (fun d keys ->
          if (not (mem output d)) && List.length keys >= 2 then begin
            refs2 := union !refs2 (String.make 1 d);
            if List.length keys >= 3 then
              refs3 := union !refs3 (String.make 1 d)
          end)
        dim_keys;
      let candidate k1 k2 =
        let either = union k1 k2 in
        let two = inter k1 k2 in
        let one = diff either two in
        let k12 =
          union
            (union (inter either output) (inter two !refs3))
            (inter one !refs2)
        in
        let cost = size_of sizes k12 - size_of sizes k1 - size_of sizes k2 in
        let id1 = Hashtbl.find remaining k1
        and id2 = Hashtbl.find remaining k2 in
        if id1 > id2 then ((cost, id1, id2), k2, k1, k12)
        else ((cost, id2, id1), k1, k2, k12)
      in
      let queue = ref [] in
      let push k1 k2s =
        match List.map (candidate k1) k2s with
        | [] -> ()
        | c :: cs ->
            let best =
              List.fold_left (fun a b -> if compare b a < 0 then b else a) c cs
            in
            queue := best :: !queue
      in
      let dims =
        Hashtbl.fold (fun d _ acc -> d :: acc) dim_keys []
        |> List.sort Char.compare
      in
      List.iter
        (fun d ->
          let keys =
            List.sort
              (fun a b ->
                compare (Hashtbl.find remaining a) (Hashtbl.find remaining b))
              (keys_of d)
          in
          let rec go = function
            | [] | [ _ ] -> ()
            | k1 :: rest ->
                push k1 rest;
                go rest
          in
          go keys)
        dims;
      let pop () =
        match List.sort compare !queue with
        | [] -> None
        | head :: tail ->
            queue := tail;
            Some head
      in
      let rec drain () =
        match pop () with
        | None -> ()
        | Some (_, k1, k2, k12) ->
            if Hashtbl.mem remaining k1 && Hashtbl.mem remaining k2 then begin
              let id1 = Hashtbl.find remaining k1 in
              let id2 = Hashtbl.find remaining k2 in
              Hashtbl.remove remaining k1;
              Hashtbl.remove remaining k2;
              List.iter (fun d -> remove_key d k1) (chars (diff k1 output));
              List.iter (fun d -> remove_key d k2) (chars (diff k2 output));
              ssa_path := [ id1; id2 ] :: !ssa_path;
              (match Hashtbl.find_opt remaining k12 with
              | Some prior ->
                  let id = fresh () in
                  ssa_path := [ prior; id ] :: !ssa_path
              | None ->
                  List.iter (fun d -> add_key d k12) (chars (diff k12 output)));
              Hashtbl.replace remaining k12 (fresh ());
              update_refs (chars (union k1 (diff k2 output)));
              let k2s =
                List.sort_uniq compare (List.concat_map keys_of (chars k12))
                |> List.filter (( <> ) k12)
              in
              push k12 k2s
            end;
            drain ()
      in
      drain ();
      let final =
        ref
          (Hashtbl.fold
             (fun key id acc ->
               (size_of sizes (inter key output), id, key) :: acc)
             remaining [])
      in
      let pop_final () =
        match List.sort compare !final with
        | head :: tail ->
            final := tail;
            head
        | [] -> failwith "einsum_path: empty greedy queue"
      in
      let _, id0, k0 = pop_final () in
      let id1 = ref id0 and k1 = ref k0 in
      while !final <> [] do
        let _, id2, k2 = pop_final () in
        ssa_path := [ Stdlib.min !id1 id2; Stdlib.max !id1 id2 ] :: !ssa_path;
        let k12 = inter (union !k1 k2) output in
        let cost = size_of sizes k12 in
        let id12 = fresh () in
        final := (cost, id12, k12) :: !final;
        let _, nid, nk = pop_final () in
        id1 := nid;
        k1 := nk
      done;
      List.rev !ssa_path
  | [] -> invalid_arg "einsum_path: no operands"

let greedy inputs output sizes = ssa_to_linear (greedy_ssa inputs output sizes)
let symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

let unused_symbols used n =
  of_chars
    (List.filteri
       (fun i _ -> i < n)
       (List.filter (fun c -> not (mem used c)) (chars symbols)))

let find_output_str subscripts =
  let flat = diff subscripts "," in
  of_chars (List.filter (fun c -> count_char flat c = 1) (chars (canon flat)))

let split_arrow s =
  let n = String.length s in
  let rec find i =
    if i + 1 >= n then None
    else if s.[i] = '-' && s.[i + 1] = '>' then Some i
    else find (i + 1)
  in
  match find 0 with
  | Some i -> (String.sub s 0 i, Some (String.sub s (i + 2) (n - i - 2)))
  | None -> (s, None)

let replace_dots term by =
  let i = String.index term '.' in
  String.sub term 0 i ^ by ^ String.sub term (i + 3) (String.length term - i - 3)

let parse subscripts shapes =
  let s = diff subscripts " " in
  if String.contains s '-' || String.contains s '>' then
    begin if
      count_char s '-' <> 1
      || count_char s '>' <> 1
      || snd (split_arrow s) = None
    then Py_exn.value_error "Subscripts can only contain one '->'."
    end;
  let input_part, output_part = split_arrow s in
  let terms = String.split_on_char ',' input_part in
  let input_terms, output_term =
    if not (String.contains s '.') then
      ( input_part,
        match output_part with
        | Some o -> o
        | None -> find_output_str input_part )
    else begin
      let used = diff s ".,->" in
      let widest =
        List.fold_left (fun acc sh -> Stdlib.max acc (Array.length sh)) 0 shapes
      in
      let ellipse = unused_symbols used widest in
      let longest = ref 0 in
      let expanded =
        List.mapi
          (fun num term ->
            if not (String.contains term '.') then term
            else begin
              let width = Array.length (List.nth shapes num) in
              let count = width - (String.length term - 3) in
              if count > !longest then longest := count;
              if count < 0 then
                Py_exn.value_error "Ellipses lengths do not match.";
              let last = String.sub ellipse (widest - count) count in
              replace_dots term last
            end)
          terms
      in
      let joined = String.concat "," expanded in
      let out_ellipse = String.sub ellipse (widest - !longest) !longest in
      match output_part with
      | Some o ->
          ( joined,
            if String.contains o '.' then replace_dots o out_ellipse else o )
      | None ->
          let normal = diff (canon (find_output_str joined)) out_ellipse in
          (joined, out_ellipse ^ normal)
    end
  in
  List.iter
    (fun c ->
      if count_char output_term c <> 1 then
        Py_exn.value_error
          (Printf.sprintf
             "Output character '%c' appeared more than once in the output." c);
      if not (mem input_terms c) then
        Py_exn.value_error
          (Printf.sprintf "Output character '%c' did not appear in the input" c))
    (chars output_term);
  let count = List.length (String.split_on_char ',' input_terms) in
  if count <> List.length shapes then
    Py_exn.value_error
      (Printf.sprintf
         "Number of einsum subscripts, %d, must be equal to the number of \
          operands, %d."
         count (List.length shapes));
  (input_terms, output_term)

let starts_with s prefix =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let ends_with s suffix =
  let n = String.length s and m = String.length suffix in
  n >= m && String.sub s (n - m) m = suffix

let can_blas inputs result removed shapes =
  match (inputs, shapes) with
  | [ left; right ], [ sl; sr ] ->
      let bad_label c =
        let nl = count_char left c and nr = count_char right c in
        nl > 1 || nr > 1
        || nl + nr > 2
        || nl + nr - 1 = if mem result c then 1 else 0
      in
      if List.exists bad_label (chars (left ^ right)) then None
      else if
        List.exists
          (fun c -> sl.(String.index left c) <> sr.(String.index right c))
          (chars removed)
      then None
      else if removed = "" then Some "OUTER/EINSUM"
      else begin
        let rs = String.length removed in
        let tail s = String.sub s (String.length s - rs) rs in
        let head s = String.sub s 0 rs in
        let keep_left = diff (canon left) removed in
        let keep_right = diff (canon right) removed in
        if left = right then Some "DOT"
        else if canon left = canon right then Some "DOT/EINSUM"
        else if
          ends_with left (head right)
          || starts_with left (tail right)
          || tail left = tail right
          || head left = head right
        then Some "GEMM"
        else if keep_left = "" || keep_right = "" then Some "GEMV/EINSUM"
        else Some "TDOT"
      end
  | _ -> None

let auto_path inputs output sizes =
  let n = List.length inputs in
  if n <= 4 then optimal inputs output sizes
  else if n <= 6 then branch None inputs output sizes
  else if n <= 8 then branch (Some 2) inputs output sizes
  else if n <= 14 then branch (Some 1) inputs output sizes
  else greedy inputs output sizes

let collect_sizes terms shapes =
  let sizes = ref [] in
  List.iteri
    (fun tnum term ->
      let sh = List.nth shapes tnum in
      if Array.length sh <> String.length term then
        Py_exn.value_error
          (Printf.sprintf
             "Einstein sum subscript '%s' does not contain the correct number \
              of indices for operand %d."
             term tnum);
      String.iteri
        (fun cnum c ->
          let dim = sh.(cnum) in
          match List.assoc_opt c !sizes with
          | None -> sizes := !sizes @ [ (c, dim) ]
          | Some 1 ->
              sizes :=
                List.map (fun (k, v) -> (k, if k = c then dim else v)) !sizes
          | Some prior when dim <> 1 && dim <> prior ->
              Py_exn.value_error
                (Printf.sprintf
                   "Size of label '%c' for operand %d (%d) does not match \
                    previous terms (%d)."
                   c tnum prior dim)
          | Some _ -> ())
        term)
    terms;
  !sizes

let take_positions positions items =
  let picked = List.map (List.nth items) positions in
  let rest = List.filteri (fun i _ -> not (List.mem i positions)) items in
  (picked, rest)

let output_shape inputs shapes result =
  Array.of_list
    (List.map
       (fun c ->
         List.fold_left2
           (fun acc term sh ->
             match String.index_opt term c with
             | Some loc -> Stdlib.max acc sh.(loc)
             | None -> acc)
           0 inputs shapes)
       (chars result))

let contract_path ?(optimize = Auto) subscripts shapes =
  let input_subscripts, output_subscript = parse subscripts shapes in
  let terms = String.split_on_char ',' input_subscripts in
  let sets = List.map canon terms in
  let indices = canon (diff input_subscripts ",") in
  let sizes = collect_sizes terms shapes in
  let num_ops = List.length terms in
  let inner =
    List.fold_left (fun acc t -> acc + String.length t) 0 sets
    - String.length indices
    > 0
  in
  let naive_cost = flop_count indices inner num_ops sizes in
  let output_set = canon output_subscript in
  let path =
    match optimize with
    | Path p -> p
    | _ when num_ops <= 2 -> [ List.init num_ops Fun.id ]
    | Unoptimized -> List.init (num_ops - 1) (fun _ -> [ 0; 1 ])
    | Auto -> auto_path sets output_set sizes
    | Optimal -> optimal sets output_set sizes
    | Greedy -> greedy sets output_set sizes
    | Branch_all -> branch None sets output_set sizes
    | Branch_2 -> branch (Some 2) sets output_set sizes
    | Branch_1 -> branch (Some 1) sets output_set sizes
  in
  let num_steps = List.length path in
  let input_sets = ref sets in
  let input_list = ref terms in
  let input_shapes = ref shapes in
  let steps =
    List.mapi
      (fun cnum inds ->
        let positions = List.sort (fun a b -> compare b a) inds in
        let picked_sets, rest_sets = take_positions positions !input_sets in
        let contract = List.fold_left union "" picked_sets in
        let remain = List.fold_left union output_set rest_sets in
        let out_inds = inter contract remain in
        let removed = diff contract out_inds in
        input_sets := rest_sets @ [ out_inds ];
        let cost =
          flop_count contract (removed <> "") (List.length positions) sizes
        in
        let scale = String.length contract in
        let size = size_of sizes out_inds in
        let tmp_inputs, rest_inputs = take_positions positions !input_list in
        let tmp_shapes, rest_shapes = take_positions positions !input_shapes in
        let blas = can_blas tmp_inputs out_inds removed tmp_shapes in
        let idx_result =
          if cnum = num_steps - 1 then output_subscript
          else begin
            let all = String.concat "" tmp_inputs in
            of_chars
              (List.sort
                 (fun a b -> compare (String.index all a) (String.index all b))
                 (chars out_inds))
          end
        in
        let shp = output_shape tmp_inputs tmp_shapes idx_result in
        input_list := rest_inputs @ [ idx_result ];
        input_shapes := rest_shapes @ [ shp ];
        let einsum_str = String.concat "," tmp_inputs ^ "->" ^ idx_result in
        let remaining =
          if List.length !input_list <= 20 then Some !input_list else None
        in
        ({ positions; einsum_str; remaining; blas }, cost, scale, size))
      path
  in
  let opt_cost = List.fold_left (fun acc (_, c, _, _) -> acc + c) 0 steps in
  let size_list = List.map (fun (_, _, _, s) -> s) steps in
  let info =
    {
      contraction_list = List.map (fun (c, _, _, _) -> c) steps;
      input_subscripts;
      output_subscript;
      eq = input_subscripts ^ "->" ^ output_subscript;
      path;
      indices;
      scale_list = List.map (fun (_, _, s, _) -> s) steps;
      naive_cost;
      opt_cost;
      speedup = float_of_int naive_cost /. float_of_int (Stdlib.max opt_cost 1);
      size_list;
      size_dict = sizes;
      shapes;
      largest_intermediate = List.fold_left Stdlib.max 1 size_list;
    }
  in
  (path, info)

let format_e x =
  let s = Printf.sprintf "%.3e" x in
  let i = String.index s 'e' in
  let mantissa = String.sub s 0 i in
  let sign = s.[i + 1] in
  let exponent =
    int_of_string (String.sub s (i + 2) (String.length s - i - 2))
  in
  Printf.sprintf "%se%c%d" mantissa sign exponent

let pad_left width s =
  let n = String.length s in
  if n >= width then s else String.make (width - n) ' ' ^ s

let path_info_to_string info =
  let max_scale = List.fold_left Stdlib.max 0 info.scale_list in
  let buf = Buffer.create 256 in
  let add = Buffer.add_string buf in
  add (Printf.sprintf "  Complete contraction:  %s\n" info.eq);
  add
    (Printf.sprintf "         Naive scaling:  %d\n"
       (String.length info.indices));
  add (Printf.sprintf "     Optimized scaling:  %d\n" max_scale);
  add
    (Printf.sprintf "      Naive FLOP count:  %s\n"
       (format_e (float_of_int info.naive_cost)));
  add
    (Printf.sprintf "  Optimized FLOP count:  %s\n"
       (format_e (float_of_int info.opt_cost)));
  add (Printf.sprintf "   Theoretical speedup:  %s\n" (format_e info.speedup));
  add
    (Printf.sprintf "  Largest intermediate:  %s elements\n"
       (format_e (float_of_int info.largest_intermediate)));
  add (String.make 80 '-' ^ "\n");
  add
    (String.concat " "
       [
         pad_left 6 "scaling";
         pad_left 11 "BLAS";
         pad_left 22 "current";
         pad_left 37 "remaining";
       ]
    ^ "\n");
  add (String.make 80 '-');
  List.iteri
    (fun n c ->
      let remaining =
        match c.remaining with
        | Some r -> String.concat "," r ^ "->" ^ info.output_subscript
        | None -> "..."
      in
      let width =
        Stdlib.max 0 (56 - Stdlib.max 22 (String.length c.einsum_str))
      in
      add
        (Printf.sprintf "\n%s %s %s    %s"
           (pad_left 4 (string_of_int (List.nth info.scale_list n)))
           (pad_left 14 (Option.value c.blas ~default:"0"))
           (pad_left 22 c.einsum_str) (pad_left width remaining)))
    info.contraction_list;
  Buffer.contents buf
