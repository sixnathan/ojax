open Types

let split_at lst n =
  match Util.split_list lst [ n ] with
  | [ a; b ] -> (a, b)
  | _ -> failwith "loops: split_at"

let mapped_leading (a : aval) : aval =
  let n = Array.length a.shape in
  { a with shape = Array.sub a.shape 1 (n - 1) }

let unmapped_leading (length : int) (a : aval) : aval =
  { a with shape = Array.append [| length |] a.shape }

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar a -> a

let scan_out_avals ~length ~num_carry (jaxpr : closed_jaxpr) : aval list =
  let outs = List.map aval_of_atom jaxpr.jaxpr.outs in
  let carry_out, y_slices = split_at outs num_carry in
  carry_out @ List.map (unmapped_leading length) y_slices

let index_leading (i : int) (v : value) : value =
  let a = Core.get_aval v in
  let sh = a.shape in
  let n = Array.length sh in
  let start = Array.init n (fun k -> if k = 0 then i else 0) in
  let limit = Array.init n (fun k -> if k = 0 then i + 1 else sh.(k)) in
  let sliced =
    Core.bind1
      (Slice { start_indices = start; limit_indices = limit; strides = None })
      [ v ]
  in
  Core.bind1 (Squeeze [| 0 |]) [ sliced ]

let rec scan_impl ~length ~reverse ~num_consts ~num_carry (jaxpr : closed_jaxpr)
    (inputs : value list) : value list =
  if num_consts > 0 then
    snd
      (split_at
         (scan_impl ~length ~reverse ~num_consts:0
            ~num_carry:(num_consts + num_carry)
            (Core.scan_lift_consts ~num_consts jaxpr)
            inputs)
         num_consts)
  else scan_impl_plain ~length ~reverse ~num_carry jaxpr inputs

and scan_impl_plain ~length ~reverse ~num_carry (jaxpr : closed_jaxpr)
    (inputs : value list) : value list =
  let carry0, xs = split_at inputs num_carry in
  let num_ys = List.length jaxpr.jaxpr.outs - num_carry in
  let ys_by_iter = Array.make length [] in
  let carry = ref carry0 in
  let order =
    if reverse then List.init length (fun i -> length - 1 - i)
    else List.init length (fun i -> i)
  in
  List.iter
    (fun i ->
      let x_slice = List.map (index_leading i) xs in
      let outs = Jaxpr.eval_closed_jaxpr jaxpr (!carry @ x_slice) in
      let c', y = split_at outs num_carry in
      carry := c';
      ys_by_iter.(i) <- y)
    order;
  let stacked =
    List.init num_ys (fun j ->
        let column = List.init length (fun i -> List.nth ys_by_iter.(i) j) in
        Core.bind1 (Stack 0) column)
  in
  !carry @ stacked

let while_out_avals (body : closed_jaxpr) : aval list =
  List.map aval_of_atom body.jaxpr.outs

let while_impl (cond : closed_jaxpr) (body : closed_jaxpr) (inputs : value list)
    : value list =
  let carry = ref inputs in
  let pred () =
    match Jaxpr.eval_closed_jaxpr cond !carry with
    | [ Concrete nd ] -> Ndarray.get_f nd [||] <> 0.0
    | [ _ ] -> failwith "loops: while cond must be concrete"
    | _ -> failwith "loops: while cond must return one value"
  in
  while pred () do
    carry := Jaxpr.eval_closed_jaxpr body !carry
  done;
  !carry

let union_cells a b =
  a @ List.filter (fun (c : Ref_state.cell) -> not (List.memq c a)) b

let promote_weak_init init (body : closed_jaxpr) =
  let outs = while_out_avals body in
  if List.length outs <> List.length init then init
  else
    List.map2
      (fun v (out : aval) ->
        let a = Core.get_aval v in
        if a.weak_type && a.dtype <> out.dtype then
          let dt, _ =
            Dtypes.result_type [ (a.dtype, true); (out.dtype, out.weak_type) ]
          in
          Core.convert_like ~weak:false v dt
        else v)
      init outs

let body_forwarded (body : closed_jaxpr) carry_binders =
  let fwd = Hashtbl.create 16 in
  let resolve (v : var) =
    Option.value (Hashtbl.find_opt fwd v.vid) ~default:v.vid
  in
  List.iter
    (fun (e : eqn) ->
      match (e.prim, e.inputs, e.outs) with
      | Convert_element_type dt, [ A_var v ], [ o ]
        when v.vaval.dtype = dt && v.vaval.weak_type = o.vaval.weak_type ->
          Hashtbl.replace fwd o.vid (resolve v)
      | Broadcast_in_dim { shape; _ }, [ A_var v ], [ o ]
        when shape = v.vaval.shape ->
          Hashtbl.replace fwd o.vid (resolve v)
      | _ -> ())
    body.jaxpr.eqns;
  List.map2
    (fun (b : var) out ->
      match out with A_var v -> resolve v = b.vid | _ -> false)
    carry_binders body.jaxpr.outs

let rec while_loop (cond_fun : value list -> value)
    (body_fun : value list -> value list) (init : value list) : value list =
  let trace init =
    let avals = List.map Core.get_aval init in
    let cond, hoisted_c, cells_c =
      Jaxpr.make_jaxpr_stateful avals (fun vs -> [ cond_fun vs ])
    in
    let body, hoisted_b, cells_b = Jaxpr.make_jaxpr_stateful avals body_fun in
    (cond, hoisted_c, cells_c, body, hoisted_b, cells_b)
  in
  let first = trace init in
  let init, (cond, hoisted_c, cells_c, body, hoisted_b, cells_b) =
    let _, _, _, body, _, _ = first in
    let promoted = promote_weak_init init body in
    if List.for_all2 ( == ) promoted init then (init, first)
    else (promoted, trace promoted)
  in
  let cells = union_cells cells_c cells_b in
  if cells <> [] then begin
    let n = List.length init in
    let enter args =
      let carry, state = split_at args n in
      Ref_state.pin cells state;
      carry
    in
    let outs =
      while_loop
        (fun args -> cond_fun (enter args))
        (fun args ->
          let next = body_fun (enter args) in
          next @ Ref_state.contents cells)
        (init @ List.map Ref_state.read cells)
    in
    let results, finals = split_at outs n in
    List.iter2 Ref_state.write cells finals;
    results
  end
  else
    let binders_first (c : closed_jaxpr) hoisted =
      let nc = List.length c.consts in
      let nh = List.length hoisted in
      let binders = c.jaxpr.in_binders in
      let total = List.length binders in
      let const_part = List.filteri (fun i _ -> i < nc) binders in
      let hoist_part = List.filteri (fun i _ -> i >= total - nh) binders in
      let arg_part =
        List.filteri (fun i _ -> i >= nc && i < total - nh) binders
      in
      ( {
          c with
          consts = [];
          jaxpr =
            { c.jaxpr with in_binders = const_part @ hoist_part @ arg_part };
        },
        List.map (fun nd -> Concrete nd) c.consts @ hoisted )
    in
    let cond', cond_consts = binders_first cond hoisted_c in
    let body', body_consts = binders_first body hoisted_b in
    let n_body_consts = List.length body_consts in
    let carry_binders =
      List.filteri (fun i _ -> i >= n_body_consts) body'.jaxpr.in_binders
    in
    let forwarded = body_forwarded body' carry_binders in
    let cond_pruned, cond_used =
      Jaxpr.dce_closed cond' (List.map (fun _ -> true) cond'.jaxpr.outs)
    in
    let n_cond_consts = List.length cond_consts in
    let cond_carry_used =
      List.filteri (fun i _ -> i >= n_cond_consts) cond_used
    in
    let move =
      List.map2 (fun f used -> f && not used) forwarded cond_carry_used
    in
    if not (List.exists Fun.id move) then
      Core.bind
        (While
           {
             cond_nconsts = List.length cond_consts;
             body_nconsts = n_body_consts;
             cond = cond';
             body = body';
           })
        (cond_consts @ body_consts @ init)
    else
      let pick keep l =
        List.concat
          (List.map2 (fun k v -> if k then [ v ] else []) keep l)
      in
      let stay = List.map not move in
      let cond_binder_keep =
        List.init n_cond_consts (fun _ -> true) @ stay
      in
      let cond_final =
        {
          cond_pruned with
          jaxpr =
            {
              cond_pruned.jaxpr with
              in_binders = pick cond_binder_keep cond'.jaxpr.in_binders;
            };
        }
      in
      let consts_part =
        List.filteri (fun i _ -> i < n_body_consts) body'.jaxpr.in_binders
      in
      let body_final =
        {
          body' with
          jid = Core.fresh_id ();
          jaxpr =
            {
              body'.jaxpr with
              in_binders =
                pick move carry_binders @ consts_part @ pick stay carry_binders;
              outs = pick stay body'.jaxpr.outs;
            };
        }
      in
      let moved = pick move init in
      let outs =
        Core.bind
          (While
             {
               cond_nconsts = n_cond_consts;
               body_nconsts = List.length moved + n_body_consts;
               cond = cond_final;
               body = body_final;
             })
          (cond_consts @ moved @ body_consts @ pick stay init)
      in
      let rec merge flags outs init =
        match (flags, outs, init) with
        | [], _, _ -> []
        | true :: fs, _, v :: tl -> v :: merge fs outs tl
        | false :: fs, o :: os, _ :: tl -> o :: merge fs os tl
        | _ -> failwith "while_loop: merge"
      in
      merge move outs init

let logaddexp a b =
  let m = Float.max a b in
  if Float.is_finite m then m +. Float.log1p (Float.exp (Float.min a b -. m))
  else m

let cumred_impl ~axis ~reverse ~combine (nd : Ndarray.t) : Ndarray.t =
  let sh = Ndarray.shape nd in
  let ndim = Array.length sh in
  if axis < 0 || axis >= ndim then
    failwith "loops: cumulative axis out of bounds";
  let dtype = Ndarray.dtype nd in
  let total = Array.fold_left ( * ) 1 sh in
  let out = Ndarray.of_floats dtype sh (Array.make total 0.0) in
  let n = sh.(axis) in
  let outer_shape = Array.mapi (fun d s -> if d = axis then 1 else s) sh in
  let outer_total = Array.fold_left ( * ) 1 outer_shape in
  let idx = Array.make ndim 0 in
  let order =
    if reverse then List.init n (fun i -> n - 1 - i)
    else List.init n (fun i -> i)
  in
  for line = 0 to outer_total - 1 do
    let rem = ref line in
    for d = ndim - 1 downto 0 do
      idx.(d) <- !rem mod outer_shape.(d);
      rem := !rem / outer_shape.(d)
    done;
    let acc = ref None in
    List.iter
      (fun i ->
        idx.(axis) <- i;
        let x = Ndarray.get_f nd idx in
        let v = match !acc with None -> x | Some a -> combine a x in
        acc := Some v;
        Ndarray.set_f out idx v)
      order
  done;
  out

let rec scan ?length ?(reverse = false) ?(unroll = 1)
    (body : value list -> value list) (init : value list) (xs : value list) :
    value list =
  let num_carry = List.length init in
  let length =
    match (length, xs) with
    | Some n, _ -> n
    | None, x :: _ -> (Core.get_aval x).shape.(0)
    | None, [] ->
        invalid_arg "scan got no values to scan over and `length` not provided."
  in
  let carry_avals = List.map Core.get_aval init in
  let x_slice_avals = List.map (fun x -> mapped_leading (Core.get_aval x)) xs in
  let jaxpr, hoisted, cells =
    Jaxpr.make_jaxpr_stateful (carry_avals @ x_slice_avals) body
  in
  if cells <> [] then begin
    let n_state = List.length cells in
    let stateful args =
      let carry, rest = split_at args num_carry in
      let state, x = split_at rest n_state in
      Ref_state.pin cells state;
      let next, ys = split_at (body (carry @ x)) num_carry in
      next @ Ref_state.contents cells @ ys
    in
    let outs =
      scan ~length ~reverse ~unroll stateful
        (init @ List.map Ref_state.read cells)
        xs
    in
    let carry, rest = split_at outs num_carry in
    let finals, ys = split_at rest n_state in
    List.iter2 Ref_state.write cells finals;
    carry @ ys
  end
  else
    let nc = List.length jaxpr.consts in
    let nh = List.length hoisted in
    let binders = jaxpr.jaxpr.in_binders in
    let total = List.length binders in
    let const_part = List.filteri (fun i _ -> i < nc) binders in
    let hoist_part = List.filteri (fun i _ -> i >= total - nh) binders in
    let arg_part =
      List.filteri (fun i _ -> i >= nc && i < total - nh) binders
    in
    let consts = List.map (fun nd -> Concrete nd) jaxpr.consts @ hoisted in
    let jaxpr =
      {
        jaxpr with
        consts = [];
        jaxpr =
          { jaxpr.jaxpr with in_binders = const_part @ hoist_part @ arg_part };
      }
    in
    Core.bind
      (Scan
         {
           length;
           reverse;
           num_consts = List.length consts;
           num_carry;
           jaxpr;
           unroll;
         })
      (consts @ init @ xs)

let const_like v x =
  let a = Core.get_aval v in
  Concrete (Ndarray.of_floats a.dtype [||] [| x |])

let dynamic_index_in_dim ?(axis = 0) ?(keepdims = true)
    ?(allow_negative_indices = true) operand index =
  let key = Printf.sprintf "%d,%b,%b" axis keepdims allow_negative_indices in
  match
    Jaxpr.named_jit ~name:"dynamic_index_in_dim" ~key
      (function
        | [ operand; index ] ->
            let a = Core.get_aval operand in
            let rank = Array.length a.shape in
            let n = a.shape.(axis) in
            let index =
              if allow_negative_indices then
                let negative = Core.bind1 Lt [ index; const_like index 0.0 ] in
                let shifted =
                  Core.bind1 Add [ index; const_like index (float_of_int n) ]
                in
                Core.bind1 Select_n [ negative; index; shifted ]
              else index
            in
            let starts =
              List.init rank (fun d ->
                  if d = axis then index else const_like index 0.0)
            in
            let sizes =
              Array.mapi (fun d s -> if d = axis then 1 else s) a.shape
            in
            let sliced =
              Core.bind1
                (Dynamic_slice { slice_sizes = sizes })
                (operand :: starts)
            in
            [
              (if keepdims then sliced
               else Core.bind1 (Squeeze [| axis |]) [ sliced ]);
            ]
        | _ -> invalid_arg "dynamic_index_in_dim: arity")
      [ operand; index ]
  with
  | [ v ] -> v
  | _ -> invalid_arg "dynamic_index_in_dim: arity"

let dynamic_update_index_in_dim ?(allow_negative_indices = true) operand update
    index axis =
  let key = Printf.sprintf "%d,%b" axis allow_negative_indices in
  match
    Jaxpr.named_jit ~name:"dynamic_update_index_in_dim" ~key
      (function
        | [ operand; update; index ] ->
            let a = Core.get_aval operand in
            let rank = Array.length a.shape in
            let n = a.shape.(axis) in
            let u = Core.get_aval update in
            let update =
              if Array.length u.shape <> rank then
                let shape =
                  Array.init rank (fun d ->
                      if d < axis then u.shape.(d)
                      else if d = axis then 1
                      else u.shape.(d - 1))
                in
                let dims =
                  Array.init (rank - 1) (fun d -> if d < axis then d else d + 1)
                in
                Core.bind1 (Broadcast_in_dim { shape; dims }) [ update ]
              else update
            in
            let index =
              if allow_negative_indices then
                let negative = Core.bind1 Lt [ index; const_like index 0.0 ] in
                let shifted =
                  Core.bind1 Add [ index; const_like index (float_of_int n) ]
                in
                Core.bind1 Select_n [ negative; index; shifted ]
              else index
            in
            let starts =
              List.init rank (fun d ->
                  if d = axis then index else const_like index 0.0)
            in
            [ Core.bind1 Dynamic_update_slice (operand :: update :: starts) ]
        | _ -> invalid_arg "dynamic_update_index_in_dim: arity")
      [ operand; update; index ]
  with
  | [ v ] -> v
  | _ -> invalid_arg "dynamic_update_index_in_dim: arity"

let lower_scan ~length ~reverse ~num_consts ~num_carry ~unroll
    (body : closed_jaxpr) args =
  let b1 = Core.bind1 in
  let consts, r = split_at args num_consts in
  let carry, xs_ = split_at r num_carry in
  Pretty_printer.register_call_name body.jid "closed_call";
  let y_avals =
    snd (split_at (List.map aval_of_atom body.jaxpr.outs) num_carry)
  in
  let num_trips, remainder =
    if unroll = 0 then (0, length) else (length / unroll, length mod unroll)
  in
  let num_trips, remainder =
    if unroll <> 1 && num_trips = 1 && remainder = 0 then (0, length)
    else (num_trips, remainder)
  in
  let empty_array prefix (a : aval) =
    let zero = Concrete (Ndarray.of_floats a.dtype [||] [| 0.0 |]) in
    b1
      (Broadcast_in_dim { shape = Array.append prefix a.shape; dims = [||] })
      [ zero ]
  in
  let slice_leading x lo hi =
    let sh = (Core.get_aval x).shape in
    let n = Array.length sh in
    b1
      (Slice
         {
           start_indices = Array.init n (fun d -> if d = 0 then lo else 0);
           limit_indices = Array.init n (fun d -> if d = 0 then hi else sh.(d));
           strides = None;
         })
      [ x ]
  in
  let split_leading sz x =
    let hi = (Core.get_aval x).shape.(0) in
    let a = slice_leading x 0 sz in
    let b = slice_leading x sz hi in
    (a, b)
  in
  let xs_, xs_rem =
    if unroll = 1 || remainder = 0 then (xs_, [])
    else if not reverse then
      List.split (List.map (split_leading (num_trips * unroll)) xs_)
    else
      let rem, front = List.split (List.map (split_leading remainder) xs_) in
      (front, rem)
  in
  let xss, yss =
    if unroll = 1 then (xs_, List.map (empty_array [| length |]) y_avals)
    else if num_trips <> 0 then
      ( List.map
          (fun x ->
            let sh = (Core.get_aval x).shape in
            b1
              (Reshape
                 (Array.append [| num_trips; unroll |]
                    (Array.sub sh 1 (Array.length sh - 1))))
              [ x ])
          xs_,
        List.map (empty_array [| num_trips; unroll |]) y_avals )
    else ([], List.map (empty_array [| 0 |]) y_avals)
  in
  let eval_body operands = Core.bind (Xla_call body) operands in
  let inner n carry xs =
    if unroll = 1 then split_at (eval_body (consts @ carry @ xs)) num_carry
    else begin
      let carry = ref carry in
      let collected = ref [] in
      for i_ = 0 to n - 1 do
        let i = if reverse then n - i_ - 1 else i_ in
        let x =
          List.map
            (fun x -> b1 (Squeeze [| 0 |]) [ slice_leading x i (i + 1) ])
            xs
        in
        let c, y = split_at (eval_body (consts @ !carry @ x)) num_carry in
        carry := c;
        collected := y :: !collected
      done;
      let ys = if reverse then !collected else List.rev !collected in
      let stacked =
        List.mapi
          (fun j _ ->
            let parts =
              List.map
                (fun y ->
                  let v = List.nth y j in
                  let a = Core.get_aval v in
                  let rank = Array.length a.shape in
                  b1
                    (Broadcast_in_dim
                       {
                         shape = Array.append [| 1 |] a.shape;
                         dims = Array.init rank (fun d -> d + 1);
                       })
                    [ v ])
                ys
            in
            b1 (Concatenate 0) parts)
          y_avals
      in
      (!carry, stacked)
    end
  in
  let n_ys = List.length yss in
  let idt = Dtypes.default_int_dtype () in
  let int_lit v = Concrete (Ndarray.of_floats idt [||] [| float_of_int v |]) in
  let carry, yss =
    if num_trips = 0 then (carry, yss)
    else
      let init = (int_lit 0 :: carry) @ yss in
      let cond_fun state = b1 Lt [ List.hd state; int_lit num_trips ] in
      let body_fun state =
        match state with
        | i_ :: rest ->
            let c, ys = split_at rest num_carry in
            let i =
              if reverse then
                b1 Sub [ b1 Sub [ int_lit num_trips; i_ ]; int_lit 1 ]
              else i_
            in
            let xs =
              List.map
                (fun x ->
                  dynamic_index_in_dim ~keepdims:false
                    ~allow_negative_indices:false x i)
                xss
            in
            let c, outs = inner unroll c xs in
            let updated =
              List.map2
                (fun y upd ->
                  dynamic_update_index_in_dim ~allow_negative_indices:false y
                    upd i 0)
                ys outs
            in
            (b1 Add [ i_; int_lit 1 ] :: c) @ updated
        | [] -> invalid_arg "lower_scan: state"
      in
      let final = while_loop cond_fun body_fun init in
      let c, ys = split_at (List.tl final) num_carry in
      (c, ys)
  in
  let ys =
    if unroll <> 1 && num_trips <> 0 then
      List.map
        (fun y ->
          let sh = (Core.get_aval y).shape in
          b1
            (Reshape
               (Array.append
                  [| num_trips * unroll |]
                  (Array.sub sh 2 (Array.length sh - 2))))
            [ y ])
        yss
    else yss
  in
  let carry, ys =
    if remainder = 0 then (carry, ys)
    else
      let carry, ys_rem = inner remainder carry xs_rem in
      let ys =
        List.map2
          (fun a b ->
            b1 (Concatenate 0) (if reverse then [ b; a ] else [ a; b ]))
          ys ys_rem
      in
      (carry, ys)
  in
  ignore n_ys;
  carry @ ys

let cumsum ?(axis = 0) ?(reverse = false) (operand : value) : value =
  Core.bind1 (Cumsum { axis; reverse }) [ operand ]

let cumprod ?(axis = 0) ?(reverse = false) (operand : value) : value =
  Core.bind1 (Cumprod { axis; reverse }) [ operand ]

let cummax ?(axis = 0) ?(reverse = false) (operand : value) : value =
  Core.bind1 (Cummax { axis; reverse }) [ operand ]

let cummin ?(axis = 0) ?(reverse = false) (operand : value) : value =
  Core.bind1 (Cummin { axis; reverse }) [ operand ]

let cumlogsumexp ?(axis = 0) ?(reverse = false) (operand : value) : value =
  Core.bind1 (Cumlogsumexp { axis; reverse }) [ operand ]
