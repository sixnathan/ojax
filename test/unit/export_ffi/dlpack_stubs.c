#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(struct DLManagedTensor *);
} DLManagedTensor;

static int deleter_calls = 0;

typedef struct {
  DLManagedTensor tensor;
  int64_t shape[4];
  int64_t strides[4];
  void *allocation;
} Owned;

static void owned_delete(DLManagedTensor *t) {
  Owned *o = t->manager_ctx;
  deleter_calls += 1;
  free(o->allocation);
  free(o);
}

value ojax_test_make_tensor(value v_code, value v_bits, value v_lanes,
                            value v_shape, value v_strides, value v_misalign,
                            value v_offset) {
  CAMLparam5(v_code, v_bits, v_lanes, v_shape, v_strides);
  CAMLxparam2(v_misalign, v_offset);
  int nd = (int)Wosize_val(v_shape);
  int bytes = Int_val(v_bits) / 8;
  int64_t count = 1;
  for (int i = 0; i < nd; i++) count *= Long_val(Field(v_shape, i));
  int offset = Int_val(v_offset);
  int misalign = Int_val(v_misalign);
  void *mem = NULL;
  if (posix_memalign(&mem, 64, 64 + (size_t)(count + 16) * (size_t)bytes + offset))
    caml_failwith("posix_memalign");
  memset(mem, 0, 64 + (size_t)(count + 16) * (size_t)bytes + offset);
  char *base = (char *)mem + misalign;
  char *payload = base + offset;
  for (int64_t i = 0; i < count; i++) {
    switch (bytes) {
      case 1: ((int8_t *)payload)[i] = (int8_t)i; break;
      case 2: ((int16_t *)payload)[i] = (int16_t)i; break;
      case 4:
        if (Int_val(v_code) == 2) ((float *)payload)[i] = (float)i;
        else ((int32_t *)payload)[i] = (int32_t)i;
        break;
      case 8:
        if (Int_val(v_code) == 2) ((double *)payload)[i] = (double)i;
        else ((int64_t *)payload)[i] = i;
        break;
      default: break;
    }
  }
  Owned *o = calloc(1, sizeof *o);
  o->allocation = mem;
  for (int i = 0; i < nd; i++) o->shape[i] = Long_val(Field(v_shape, i));
  int has_strides = Wosize_val(v_strides) > 0;
  for (int i = 0; i < nd && has_strides; i++)
    o->strides[i] = Long_val(Field(v_strides, i));
  o->tensor.dl_tensor.data = base;
  o->tensor.dl_tensor.device.device_type = 1;
  o->tensor.dl_tensor.device.device_id = 0;
  o->tensor.dl_tensor.ndim = nd;
  o->tensor.dl_tensor.dtype.code = (uint8_t)Int_val(v_code);
  o->tensor.dl_tensor.dtype.bits = (uint8_t)Int_val(v_bits);
  o->tensor.dl_tensor.dtype.lanes = (uint16_t)Int_val(v_lanes);
  o->tensor.dl_tensor.shape = o->shape;
  o->tensor.dl_tensor.strides = has_strides ? o->strides : NULL;
  o->tensor.dl_tensor.byte_offset = (uint64_t)offset;
  o->tensor.manager_ctx = o;
  o->tensor.deleter = owned_delete;
  CAMLreturn(caml_copy_nativeint((intnat)&o->tensor));
}

value ojax_test_make_tensor_bytecode(value *argv, int argn) {
  (void)argn;
  return ojax_test_make_tensor(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5], argv[6]);
}

value ojax_test_deleter_calls(value unit) {
  (void)unit;
  return Val_int(deleter_calls);
}

value ojax_test_data_pointer(value v_ptr) {
  CAMLparam1(v_ptr);
  DLManagedTensor *t = (DLManagedTensor *)Nativeint_val(v_ptr);
  CAMLreturn(caml_copy_nativeint(
      (intnat)((char *)t->dl_tensor.data + t->dl_tensor.byte_offset)));
}

value ojax_test_write_i32(value v_addr, value v_index, value v_x) {
  CAMLparam3(v_addr, v_index, v_x);
  ((int32_t *)Nativeint_val(v_addr))[Int_val(v_index)] = (int32_t)Int_val(v_x);
  CAMLreturn(Val_unit);
}

value ojax_test_describe(value v_ptr) {
  CAMLparam1(v_ptr);
  CAMLlocal3(v_out, v_shape, v_strides);
  DLManagedTensor *t = (DLManagedTensor *)Nativeint_val(v_ptr);
  DLTensor *d = &t->dl_tensor;
  v_shape = caml_alloc(d->ndim, 0);
  for (int i = 0; i < d->ndim; i++)
    Store_field(v_shape, i, Val_long(d->shape[i]));
  if (d->strides != NULL) {
    v_strides = caml_alloc(d->ndim, 0);
    for (int i = 0; i < d->ndim; i++)
      Store_field(v_strides, i, Val_long(d->strides[i]));
  } else {
    v_strides = Atom(0);
  }
  v_out = caml_alloc_tuple(10);
  Store_field(v_out, 0, Val_int(d->dtype.code));
  Store_field(v_out, 1, Val_int(d->dtype.bits));
  Store_field(v_out, 2, Val_int(d->dtype.lanes));
  Store_field(v_out, 3, Val_int(d->ndim));
  Store_field(v_out, 4, v_shape);
  Store_field(v_out, 5, v_strides);
  Store_field(v_out, 6, Val_long(d->byte_offset));
  Store_field(v_out, 7, Val_int(d->device.device_type));
  Store_field(v_out, 8, Val_int(d->device.device_id));
  Store_field(v_out, 9, Val_bool(((uintptr_t)d->data % 64) == 0));
  CAMLreturn(v_out);
}

value ojax_test_sum_f32(value v_ptr) {
  CAMLparam1(v_ptr);
  DLManagedTensor *t = (DLManagedTensor *)Nativeint_val(v_ptr);
  DLTensor *d = &t->dl_tensor;
  int64_t n = 1;
  for (int i = 0; i < d->ndim; i++) n *= d->shape[i];
  double acc = 0;
  const float *p = (const float *)((char *)d->data + d->byte_offset);
  for (int64_t i = 0; i < n; i++) acc += p[i];
  CAMLreturn(caml_copy_double(acc));
}
