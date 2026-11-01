#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "lapack_shim.h"

typedef int32_t LI;

typedef struct Ext {
  size_t struct_size;
  int type;
  struct Ext *next;
} Ext;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  int dtype;
  void *data;
  int64_t rank;
  int64_t *dims;
} Buffer;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  int64_t size;
  int *types;
  void **items;
} List;

typedef struct {
  const char *ptr;
  size_t len;
} Span;

typedef struct {
  int dtype;
  void *value;
} Scalar;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  int64_t size;
  int *types;
  Span **names;
  void **attrs;
} Attrs;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  const void *api;
  void *ctx;
  int stage;
  List args;
  List rets;
  Attrs attrs;
  void *future;
} Frame;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  int major_version;
  int minor_version;
} Version;

typedef struct {
  size_t struct_size;
  Version api_version;
  uint32_t traits;
} Metadata;

typedef struct {
  Ext base;
  Metadata *metadata;
} MetadataExt;

_Static_assert(offsetof(Buffer, dims) + sizeof(void *) == 48,
               "XLA_FFI_Buffer size drift");
_Static_assert(offsetof(Frame, stage) == 32, "XLA_FFI_CallFrame stage drift");

#define EXT_METADATA 1
#define STAGE_EXECUTE 3
#define DT_F32 11
#define DT_F64 12
#define DT_C64 15
#define DT_C128 18

static int metadata_query(Frame *f) {
  for (Ext *e = f->extension_start; e; e = e->next)
    if (e->type == EXT_METADATA) {
      Metadata *m = ((MetadataExt *)e)->metadata;
      m->api_version.struct_size = sizeof(Version);
      m->api_version.extension_start = NULL;
      m->api_version.major_version = 0;
      m->api_version.minor_version = 1;
      m->traits = 0;
      return 1;
    }
  return 0;
}

#define ENTER(name)                                \
  static void *name(void *frame) {                 \
    Frame *f = frame;                              \
    if (metadata_query(f)) return NULL;            \
    if (f->stage != STAGE_EXECUTE) return NULL;

#define ARG(i) ((Buffer *)f->args.items[(i)])
#define RET(i) ((Buffer *)f->rets.items[(i)])

static char prec_of(int dtype) {
  switch (dtype) {
    case DT_F32:
      return 's';
    case DT_F64:
      return 'd';
    case DT_C64:
      return 'c';
    default:
      return 'z';
  }
}

static size_t elt_bytes(int dtype) {
  switch (dtype) {
    case DT_F32:
      return 4;
    case DT_F64:
    case DT_C64:
      return 8;
    case DT_C128:
      return 16;
    default:
      return 4;
  }
}

static const Scalar *attr(Frame *f, const char *name) {
  size_t len = strlen(name);
  for (int64_t i = 0; i < f->attrs.size; i++) {
    const Span *s = f->attrs.names[i];
    if (s->len == len && memcmp(s->ptr, name, len) == 0)
      return f->attrs.attrs[i];
  }
  return NULL;
}

static char attr_u8(Frame *f, const char *name) {
  const Scalar *s = attr(f, name);
  return s ? (char)*(uint8_t *)s->value : 0;
}

static int attr_bool(Frame *f, const char *name) {
  const Scalar *s = attr(f, name);
  return s ? *(uint8_t *)s->value != 0 : 0;
}

static LI attr_i32(Frame *f, const char *name) {
  const Scalar *s = attr(f, name);
  return s ? *(int32_t *)s->value : 0;
}

static int64_t batches(const Buffer *b, int core) {
  int64_t n = 1;
  for (int64_t i = 0; i < b->rank - core; i++) n *= b->dims[i];
  return n;
}

static int64_t core_size(const Buffer *b, int core) {
  int64_t n = 1;
  for (int64_t i = b->rank - core; i < b->rank; i++) n *= b->dims[i];
  return n;
}

static void *at(const Buffer *b, int core, int64_t batch) {
  return (char *)b->data +
         (size_t)(batch * core_size(b, core)) * elt_bytes(b->dtype);
}

static LI rows(const Buffer *b) { return (LI)b->dims[b->rank - 2]; }
static LI cols(const Buffer *b) { return (LI)b->dims[b->rank - 1]; }

static void copy_to(Buffer *out, const Buffer *in) {
  if (out->data == in->data) return;
  memcpy(out->data, in->data,
         (size_t)core_size(in, (int)in->rank) * elt_bytes(in->dtype));
}

static void set_info(Buffer *info, int64_t batch, int value) {
  ((int32_t *)info->data)[batch] = value;
}

ENTER(h_potrf)
  Buffer *a = RET(0), *info = RET(1);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype), uplo = attr_u8(f, "uplo");
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    set_info(info, b,
             ojax_lapack_call_potrf(prec, uplo, cols(a), at(a, 2, b), rows(a)));
  return NULL;
}

ENTER(h_getrf)
  Buffer *a = RET(0), *ipiv = RET(1), *info = RET(2);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype);
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    set_info(info, b,
             ojax_lapack_call_getrf(prec, rows(a), cols(a), at(a, 2, b), rows(a),
                                    at(ipiv, 1, b)));
  return NULL;
}

ENTER(h_geqrf)
  Buffer *a = RET(0), *tau = RET(1);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype);
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    ojax_lapack_call_geqrf(prec, rows(a), cols(a), at(a, 2, b), rows(a),
                           at(tau, 1, b));
  return NULL;
}

ENTER(h_geqp3)
  Buffer *a = RET(0), *jpvt = RET(1), *tau = RET(2);
  copy_to(a, ARG(0));
  copy_to(jpvt, ARG(1));
  char prec = prec_of(a->dtype);
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    ojax_lapack_call_geqp3(prec, rows(a), cols(a), at(a, 2, b), rows(a),
                           at(jpvt, 1, b), at(tau, 1, b));
  return NULL;
}

ENTER(h_orgqr)
  Buffer *a = RET(0), *tau = ARG(1);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype);
  LI k = (LI)tau->dims[tau->rank - 1];
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    ojax_lapack_call_orgqr(prec, rows(a), cols(a), k, at(a, 2, b), rows(a),
                           at(tau, 1, b));
  return NULL;
}

ENTER(h_ormqr)
  Buffer *a = ARG(0), *tau = ARG(1), *c = RET(0);
  copy_to(c, ARG(2));
  char prec = prec_of(c->dtype);
  int cplx = prec == 'c' || prec == 'z';
  char side = attr_bool(f, "left") ? 'L' : 'R';
  char trans = attr_bool(f, "transpose") ? (cplx ? 'C' : 'T') : 'N';
  LI k = (LI)tau->dims[tau->rank - 1];
  for (int64_t b = 0, n = batches(c, 2); b < n; b++)
    ojax_lapack_call_ormqr(prec, side, trans, rows(c), cols(c), k, at(a, 2, b),
                           rows(a), at(tau, 1, b), at(c, 2, b), rows(c));
  return NULL;
}

static void *svd_handler(Frame *f, int qr) {
  Buffer *a = RET(0), *s = RET(1), *u = RET(2), *vt = RET(3), *info = RET(4);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype), mode = attr_u8(f, "mode");
  for (int64_t b = 0, n = batches(a, 2); b < n; b++) {
    int r = qr ? ojax_lapack_call_gesvd(prec, mode, rows(a), cols(a), at(a, 2, b),
                                        rows(a), at(s, 1, b), at(u, 2, b), rows(u),
                                        at(vt, 2, b), rows(vt))
               : ojax_lapack_call_gesdd(prec, mode, rows(a), cols(a), at(a, 2, b),
                                        rows(a), at(s, 1, b), at(u, 2, b), rows(u),
                                        at(vt, 2, b), rows(vt));
    set_info(info, b, r);
  }
  return NULL;
}

ENTER(h_gesdd)
  return svd_handler(f, 0);
}

ENTER(h_gesvd)
  return svd_handler(f, 1);
}

ENTER(h_syevd)
  Buffer *a = RET(0), *w = RET(1), *info = RET(2);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype), mode = attr_u8(f, "mode"),
       uplo = attr_u8(f, "uplo");
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    set_info(info, b,
             ojax_lapack_call_syevd(prec, mode, uplo, cols(a), at(a, 2, b),
                                    rows(a), at(w, 1, b)));
  return NULL;
}

static void unpack_real_vectors(char prec, LI n, const void *real, const void *wi,
                                void *out) {
  for (LI j = 0; j < n;) {
    int pair = prec == 's' ? ((const float *)wi)[j] != 0.0f
                           : ((const double *)wi)[j] != 0.0;
    for (LI i = 0; i < n; i++) {
      size_t k = (size_t)i + (size_t)j * (size_t)n;
      if (prec == 's') {
        const float *r = real;
        float *o = out;
        float re = r[k], im = pair ? r[k + (size_t)n] : 0.0f;
        o[2 * k] = re;
        o[2 * k + 1] = im;
        if (pair) {
          o[2 * (k + (size_t)n)] = re;
          o[2 * (k + (size_t)n) + 1] = -im;
        }
      } else {
        const double *r = real;
        double *o = out;
        double re = r[k], im = pair ? r[k + (size_t)n] : 0.0;
        o[2 * k] = re;
        o[2 * k + 1] = im;
        if (pair) {
          o[2 * (k + (size_t)n)] = re;
          o[2 * (k + (size_t)n) + 1] = -im;
        }
      }
    }
    j += pair ? 2 : 1;
  }
}

ENTER(h_geev)
  Buffer *a = ARG(0);
  char prec = prec_of(a->dtype);
  int cplx = prec == 'c' || prec == 'z';
  Buffer *wr = RET(0), *vl = RET(cplx ? 1 : 2), *vr = RET(cplx ? 2 : 3),
         *info = RET(cplx ? 3 : 4);
  char jobvl = attr_u8(f, "compute_left"), jobvr = attr_u8(f, "compute_right");
  LI n = cols(a);
  size_t real_bytes = prec == 's' ? 4 : 8;
  size_t nn = (size_t)n * (size_t)n;
  void *copy = malloc(nn * elt_bytes(a->dtype) + 1);
  void *w = malloc(2 * (size_t)n * real_bytes + 1);
  void *rl = cplx ? NULL : malloc(nn * real_bytes + 1);
  void *rr = cplx ? NULL : malloc(nn * real_bytes + 1);
  int oom = !copy || !w || (!cplx && (!rl || !rr));
  for (int64_t b = 0, count = batches(a, 2); b < count; b++) {
    if (oom) {
      set_info(info, b, -1);
      continue;
    }
    memcpy(copy, at(a, 2, b), nn * elt_bytes(a->dtype));
    int r;
    if (cplx) {
      r = ojax_lapack_call_geev(prec, jobvl, jobvr, n, copy, n, at(wr, 1, b),
                                at(vl, 2, b), n, at(vr, 2, b), n);
    } else {
      r = ojax_lapack_call_geev(prec, jobvl, jobvr, n, copy, n, w, rl, n, rr, n);
      if (r == 0) {
        const char *wi = (const char *)w + (size_t)n * real_bytes;
        memcpy(at(wr, 1, b), w, (size_t)n * real_bytes);
        memcpy(at(RET(1), 1, b), wi, (size_t)n * real_bytes);
        if (jobvl == 'V') unpack_real_vectors(prec, n, rl, wi, at(vl, 2, b));
        if (jobvr == 'V') unpack_real_vectors(prec, n, rr, wi, at(vr, 2, b));
      }
    }
    set_info(info, b, r);
  }
  free(copy);
  free(w);
  free(rl);
  free(rr);
  return NULL;
}

ENTER(h_gees)
  Buffer *a = RET(0), *vs = RET(1);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype);
  int cplx = prec == 'c' || prec == 'z';
  Buffer *wr = RET(2), *sdim = RET(cplx ? 3 : 4), *info = RET(cplx ? 4 : 5);
  char mode = attr_u8(f, "mode");
  LI n = cols(a);
  size_t real_bytes = prec == 's' ? 4 : 8;
  void *w = cplx ? NULL : malloc(2 * (size_t)n * real_bytes + 1);
  for (int64_t b = 0, count = batches(a, 2); b < count; b++) {
    if (!cplx && !w) {
      set_info(info, b, -1);
      continue;
    }
    int r = ojax_lapack_call_gees(prec, mode, n, at(a, 2, b), n,
                                  (int32_t *)sdim->data + b,
                                  cplx ? at(wr, 1, b) : w, at(vs, 2, b), n);
    if (!cplx) {
      memcpy(at(wr, 1, b), w, (size_t)n * real_bytes);
      memcpy(at(RET(3), 1, b), (char *)w + (size_t)n * real_bytes,
             (size_t)n * real_bytes);
    }
    set_info(info, b, r);
  }
  free(w);
  return NULL;
}

ENTER(h_gehrd)
  Buffer *a = RET(0), *tau = RET(1), *info = RET(2);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype);
  LI low = attr_i32(f, "low"), high = attr_i32(f, "high");
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    set_info(info, b,
             ojax_lapack_call_gehrd(prec, cols(a), low, high, at(a, 2, b), rows(a),
                                    at(tau, 1, b)));
  return NULL;
}

ENTER(h_sytrd)
  Buffer *a = RET(0), *d = RET(1), *e = RET(2), *tau = RET(3), *info = RET(4);
  copy_to(a, ARG(0));
  char prec = prec_of(a->dtype), uplo = attr_u8(f, "uplo");
  for (int64_t b = 0, n = batches(a, 2); b < n; b++)
    set_info(info, b,
             ojax_lapack_call_sytrd(prec, uplo, cols(a), at(a, 2, b), rows(a),
                                    at(d, 1, b), at(e, 1, b), at(tau, 1, b)));
  return NULL;
}

ENTER(h_trsm)
  Buffer *a = ARG(0), *x = RET(0);
  copy_to(x, ARG(1));
  char prec = prec_of(x->dtype), side = attr_u8(f, "side"),
       uplo = attr_u8(f, "uplo"), trans = attr_u8(f, "trans_x"),
       diag = attr_u8(f, "diag");
  for (int64_t b = 0, n = batches(x, 2); b < n; b++)
    ojax_lapack_call_trsm(prec, side, uplo, trans, diag, rows(x), cols(x),
                          at(a, 2, b), rows(a), at(x, 2, b),
                          rows(x) < 1 ? 1 : rows(x));
  return NULL;
}

ENTER(h_gtsv)
  Buffer *dl = RET(0), *d = RET(1), *du = RET(2), *x = RET(3), *info = RET(4);
  copy_to(dl, ARG(0));
  copy_to(d, ARG(1));
  copy_to(du, ARG(2));
  copy_to(x, ARG(3));
  char prec = prec_of(x->dtype);
  LI n = (LI)d->dims[d->rank - 1];
  for (int64_t b = 0, count = batches(x, 2); b < count; b++)
    set_info(info, b,
             ojax_lapack_call_gtsv(prec, n, cols(x),
                                   (char *)at(dl, 1, b) + elt_bytes(dl->dtype),
                                   at(d, 1, b), at(du, 1, b), at(x, 2, b),
                                   n < 1 ? 1 : n));
  return NULL;
}

ENTER(h_tridiagonal_perturbed)
  Buffer *dl = ARG(0), *d = ARG(1), *du = ARG(2), *b = ARG(3), *x = RET(0);
  char prec = prec_of(b->dtype);
  int n = (int)rows(b), k_rhs = (int)cols(b);
  for (int64_t i = 0, count = batches(b, 2); i < count; i++)
    ojax_tridiagonal_perturbed(prec, n, k_rhs, at(dl, 1, i), at(d, 1, i),
                               at(du, 1, i), at(b, 2, i), at(x, 2, i));
  return NULL;
}

static size_t ffi_elt_bytes(int dtype) {
  switch (dtype) {
    case 1:
    case 2:
    case 6:
      return 1;
    case 3:
    case 7:
    case 10:
    case 16:
      return 2;
    case 4:
    case 8:
    case 11:
      return 4;
    case 15:
    case 5:
    case 9:
    case 12:
      return 8;
    case 18:
      return 16;
    default:
      return 0;
  }
}

static value host_view(const Buffer *b) {
  CAMLparam0();
  CAMLlocal3(item, dims, data);
  int64_t count = 1;
  dims = caml_alloc(b->rank, 0);
  for (int64_t d = 0; d < b->rank; d++) {
    Store_field(dims, d, Val_long(b->dims[d]));
    count *= b->dims[d];
  }
  intnat size = (intnat)(count * (int64_t)ffi_elt_bytes(b->dtype));
  data = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                            1, b->data, size);
  item = caml_alloc_tuple(3);
  Store_field(item, 0, Val_int(b->dtype));
  Store_field(item, 1, dims);
  Store_field(item, 2, data);
  CAMLreturn(item);
}

static void dispatch_host_callback(Frame *f, uint64_t index) {
  CAMLparam0();
  CAMLlocal2(args, rets);
  const value *fn = caml_named_value("ojax_host_callback");
  args = caml_alloc(f->args.size, 0);
  for (int64_t i = 0; i < f->args.size; i++)
    Store_field(args, i, host_view(ARG(i)));
  rets = caml_alloc(f->rets.size, 0);
  for (int64_t i = 0; i < f->rets.size; i++)
    Store_field(rets, i, host_view(RET(i)));
  if (fn != NULL)
    (void)caml_callback3_exn(*fn, Val_long((intnat)index), args, rets);
  CAMLreturn0;
}

ENTER(h_host_callback)
  const Scalar *s = attr(f, "index");
  uint64_t index = s ? *(uint64_t *)s->value : 0;
  int registered = caml_c_thread_register();
  caml_acquire_runtime_system();
  dispatch_host_callback(f, index);
  caml_release_runtime_system();
  if (registered) caml_c_thread_unregister();
  return NULL;
}

CAMLprim value ojax_lapack_ffi_handler(value v_name) {
  CAMLparam1(v_name);
  CAMLlocal1(result);
  const char *name = String_val(v_name);
  static const struct {
    const char *name;
    void *(*fn)(void *);
  } table[] = {{"potrf", h_potrf}, {"getrf", h_getrf}, {"geqrf", h_geqrf},
               {"geqp3", h_geqp3}, {"orgqr", h_orgqr}, {"ormqr", h_ormqr},
               {"gesdd", h_gesdd}, {"gesvd", h_gesvd}, {"syevd", h_syevd},
               {"geev", h_geev},   {"gees", h_gees},   {"gehrd", h_gehrd},
               {"sytrd", h_sytrd}, {"trsm", h_trsm},   {"gtsv", h_gtsv},
               {"tridiagonal_perturbed", h_tridiagonal_perturbed},
               {"host_callback", h_host_callback}};
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
    if (strcmp(table[i].name, name) == 0) {
      result = caml_copy_nativeint((intnat)table[i].fn);
      CAMLreturn(result);
    }
  caml_failwith("ojax linalg: unknown LAPACK handler");
}
