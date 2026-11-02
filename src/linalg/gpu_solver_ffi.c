#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "lapack_shim.h"

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
  const char *api;
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

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  const char *message;
  int errc;
} ErrorArgs;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  void *ctx;
  void *stream;
} StreamArgs;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  void *ctx;
  size_t size;
  size_t alignment;
  void *data;
} AllocArgs;

typedef struct {
  size_t struct_size;
  Ext *extension_start;
  void *ctx;
  size_t size;
  void *data;
} FreeArgs;

_Static_assert(offsetof(Frame, stage) == 32, "XLA_FFI_CallFrame stage drift");

#define API_ERROR_CREATE 48
#define API_STREAM_GET 80
#define API_ALLOCATE 120
#define API_FREE 128

#define EXT_METADATA 1
#define STAGE_EXECUTE 3
#define DT_F32 11
#define DT_F64 12
#define DT_C64 15
#define DT_C128 18

#define ERRC_INTERNAL 13
#define ERRC_RESOURCE_EXHAUSTED 8

#define SIDE_LEFT 0
#define SIDE_RIGHT 1
#define OP_N 0
#define OP_T 1
#define OP_C 2
#define FILL_LOWER 0
#define FILL_UPPER 1
#define EIG_NOVECTOR 0
#define EIG_VECTOR 1

#define COPY_HOST_TO_DEVICE 1
#define COPY_DEVICE_TO_HOST 2
#define COPY_DEVICE_TO_DEVICE 3

static void *lib_solver;
static void *lib_blas;
static void *lib_runtime;

static void *open_first(const char *const *names) {
  const char *root = getenv("CUDA_ROOT");
  for (; *names; names++) {
    void *h = dlopen(*names, RTLD_NOW | RTLD_GLOBAL);
    if (h) return h;
    if (root) {
      char path[PATH_MAX];
      snprintf(path, sizeof(path), "%s/lib/%s", root, *names);
      h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
      if (h) return h;
    }
  }
  return NULL;
}

static int load_libraries(void) {
  static const char *const runtime[] = {"libcudart.so.13", "libcudart.so.12",
                                        NULL};
  static const char *const blas[] = {"libcublas.so.13", "libcublas.so.12",
                                     NULL};
  static const char *const solver[] = {"libcusolver.so.12", "libcusolver.so.11",
                                       NULL};
  if (!lib_runtime) lib_runtime = open_first(runtime);
  if (!lib_blas) lib_blas = open_first(blas);
  if (!lib_solver) lib_solver = open_first(solver);
  return lib_runtime && lib_blas && lib_solver;
}

static void *need(void *lib, const char *name) {
  void *fn = dlsym(lib, name);
  if (!fn) {
    fprintf(stderr, "ojax gpu solver: missing symbol %s\n", name);
    abort();
  }
  return fn;
}

static void *solver_sym(char prec, const char *real, const char *cplx,
                        const char *suffix) {
  char name[96];
  int is_complex = prec == 'C' || prec == 'Z';
  snprintf(name, sizeof(name), "cusolverDn%c%s%s", prec, is_complex ? cplx : real,
           suffix);
  return need(lib_solver, name);
}

static void *blas_sym(char prec, const char *base) {
  char name[96];
  snprintf(name, sizeof(name), "cublas%c%s", prec, base);
  return need(lib_blas, name);
}

typedef struct {
  Frame *f;
  void *stream;
  void *error;
  void *alloc[8];
  size_t alloc_size[8];
  int allocs;
} Kernel;

static void *make_error(Frame *f, int errc, const char *message) {
  void *(*create)(ErrorArgs *) = *(void **)(f->api + API_ERROR_CREATE);
  ErrorArgs args = {sizeof(ErrorArgs), NULL, message, errc};
  return create(&args);
}

static void fail_call(Kernel *k, const char *what, int status) {
  char message[160];
  snprintf(message, sizeof(message), "%s failed with status %d", what, status);
  if (!k->error) k->error = make_error(k->f, ERRC_INTERNAL, message);
}

static void *scratch(Kernel *k, size_t size, size_t alignment) {
  void *(*allocate)(AllocArgs *) = *(void **)(k->f->api + API_ALLOCATE);
  AllocArgs args = {sizeof(AllocArgs), NULL, k->f->ctx, size, alignment, NULL};
  void *err = allocate(&args);
  if (err) {
    if (!k->error) k->error = err;
    return NULL;
  }
  k->alloc[k->allocs] = args.data;
  k->alloc_size[k->allocs] = size;
  k->allocs++;
  return args.data;
}

static void *finish(Kernel *k) {
  void *(*release)(FreeArgs *) = *(void **)(k->f->api + API_FREE);
  for (int i = 0; i < k->allocs; i++) {
    FreeArgs args = {sizeof(FreeArgs), NULL, k->f->ctx, k->alloc_size[i],
                     k->alloc[i]};
    release(&args);
  }
  return k->error;
}

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

static void *start(Kernel *k, Frame *f) {
  k->f = f;
  k->error = NULL;
  k->allocs = 0;
  void *(*stream_get)(StreamArgs *) = *(void **)(f->api + API_STREAM_GET);
  StreamArgs args = {sizeof(StreamArgs), NULL, f->ctx, NULL};
  void *err = stream_get(&args);
  k->stream = args.stream;
  if (err) return err;
  if (!load_libraries())
    return make_error(f, ERRC_INTERNAL, "ojax gpu solver: CUDA libraries missing");
  return NULL;
}

#define ENTER(name)                                   \
  static void *name(void *frame) {                    \
    Frame *f = frame;                                 \
    if (metadata_query(f)) return NULL;               \
    if (f->stage != STAGE_EXECUTE) return NULL;       \
    Kernel kernel;                                    \
    Kernel *k = &kernel;                              \
    {                                                 \
      void *start_error = start(k, f);                \
      if (start_error) return start_error;            \
    }

#define ARG(i) ((Buffer *)f->args.items[(i)])
#define RET(i) ((Buffer *)f->rets.items[(i)])

#define CHECK(what, call)                \
  do {                                   \
    int status_ = (call);                \
    if (status_ != 0) {                  \
      fail_call(k, what, status_);       \
      return finish(k);                  \
    }                                    \
  } while (0)

#define CHECK_PTR(p)             \
  do {                           \
    if (!(p)) return finish(k);  \
  } while (0)

static char prec_of(int dtype) {
  switch (dtype) {
    case DT_F32:
      return 'S';
    case DT_F64:
      return 'D';
    case DT_C64:
      return 'C';
    default:
      return 'Z';
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

static size_t real_bytes(int dtype) {
  return dtype == DT_F32 || dtype == DT_C64 ? 4 : 8;
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

static int attr_flag(Frame *f, const char *name) {
  const Scalar *s = attr(f, name);
  return s ? *(uint8_t *)s->value != 0 : 0;
}

static int attr_u8(Frame *f, const char *name) {
  const Scalar *s = attr(f, name);
  return s ? *(uint8_t *)s->value : 0;
}

static int64_t batch_of(const Buffer *b, int core) {
  int64_t n = 1;
  for (int64_t i = 0; i < b->rank - core; i++) n *= b->dims[i];
  return n;
}

static size_t byte_size(const Buffer *b) {
  size_t n = elt_bytes(b->dtype);
  for (int64_t i = 0; i < b->rank; i++) n *= (size_t)b->dims[i];
  return n;
}

typedef int (*copy_fn)(void *, const void *, size_t, int, void *);
typedef int (*sync_fn)(void *);

static int device_copy(Kernel *k, void *dst, const void *src, size_t n, int kind) {
  copy_fn copy = (copy_fn)need(lib_runtime, "cudaMemcpyAsync");
  return copy(dst, src, n, kind, k->stream);
}

static int stream_sync(Kernel *k) {
  sync_fn sync = (sync_fn)need(lib_runtime, "cudaStreamSynchronize");
  return sync(k->stream);
}

static int copy_input(Kernel *k, void *out, const Buffer *a) {
  if (a->data == out) return 0;
  return device_copy(k, out, a->data, byte_size(a), COPY_DEVICE_TO_DEVICE);
}

typedef struct PoolEntry {
  void *stream;
  void *handle;
  int busy;
  struct PoolEntry *next;
} PoolEntry;

typedef struct {
  PoolEntry *entries;
  pthread_mutex_t lock;
} Pool;

static Pool solver_pool = {NULL, PTHREAD_MUTEX_INITIALIZER};
static Pool blas_pool = {NULL, PTHREAD_MUTEX_INITIALIZER};

static PoolEntry *pool_take(Pool *pool, void *stream) {
  pthread_mutex_lock(&pool->lock);
  PoolEntry *e = pool->entries;
  while (e && (e->busy || e->stream != stream)) e = e->next;
  if (e) {
    e->busy = 1;
  } else {
    e = calloc(1, sizeof(PoolEntry));
    e->stream = stream;
    e->busy = 1;
    e->next = pool->entries;
    pool->entries = e;
  }
  pthread_mutex_unlock(&pool->lock);
  return e;
}

static void pool_give(Pool *pool, void *handle) {
  pthread_mutex_lock(&pool->lock);
  for (PoolEntry *e = pool->entries; e; e = e->next)
    if (e->handle == handle) e->busy = 0;
  pthread_mutex_unlock(&pool->lock);
}

static void *pooled_handle(Kernel *k, Pool *pool, void *lib, const char *create_name,
                           const char *set_name) {
  int (*create)(void **) = (int (*)(void **))need(lib, create_name);
  int (*set_stream)(void *, void *) = (int (*)(void *, void *))need(lib, set_name);
  PoolEntry *e = pool_take(pool, k->stream);
  int status = 0;
  if (!e->handle) status = create(&e->handle);
  if (status == 0 && k->stream) status = set_stream(e->handle, k->stream);
  if (status != 0) {
    fail_call(k, create_name, status);
    return NULL;
  }
  return e->handle;
}

static void *solver_handle(Kernel *k) {
  return pooled_handle(k, &solver_pool, lib_solver, "cusolverDnCreate",
                       "cusolverDnSetStream");
}

static void solver_release(void *h) { pool_give(&solver_pool, h); }

static void *blas_handle(Kernel *k) {
  return pooled_handle(k, &blas_pool, lib_blas, "cublasCreate_v2",
                       "cublasSetStream_v2");
}

static void blas_release(void *h) { pool_give(&blas_pool, h); }

static void **batch_pointers(Kernel *k, char *base, int64_t batch, size_t step) {
  void **device = scratch(k, sizeof(void *) * (size_t)batch, sizeof(void *));
  if (!device) return NULL;
  void **host = malloc(sizeof(void *) * (size_t)batch);
  for (int64_t i = 0; i < batch; i++) host[i] = base + (size_t)i * step;
  int status = device_copy(k, device, host, sizeof(void *) * (size_t)batch,
                           COPY_HOST_TO_DEVICE);
  if (status == 0) status = stream_sync(k);
  free(host);
  if (status != 0) {
    fail_call(k, "batch pointers", status);
    return NULL;
  }
  return device;
}

ENTER(h_getrf)
  Buffer *a = ARG(0), *out = RET(0), *ipiv = RET(1), *info = RET(2);
  int64_t batch = batch_of(a, 2);
  int m = (int)a->dims[a->rank - 2], n = (int)a->dims[a->rank - 1];
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  if (batch > 1 && m == n && m / batch <= 128) {
    void *handle = blas_handle(k);
    CHECK_PTR(handle);
    CHECK("copy", copy_input(k, out->data, a));
    void **ptrs = batch_pointers(k, out->data, batch, elt * (size_t)n * (size_t)n);
    CHECK_PTR(ptrs);
    int (*batched)(void *, int, void **, int, int *, int *, int) =
        (int (*)(void *, int, void **, int, int *, int *, int))blas_sym(
            p, "getrfBatched");
    int status = batched(handle, n, ptrs, n, ipiv->data, info->data, (int)batch);
    blas_release(handle);
    CHECK("getrfBatched", status);
    return finish(k);
  }
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  int lwork = 0;
  int (*size)(void *, int, int, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, int *))solver_sym(p, "getrf", "getrf",
                                                                "_bufferSize");
  CHECK("getrf_bufferSize", size(handle, m, n, NULL, m, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, int, int, void *, int, void *, int *, int *) =
      (int (*)(void *, int, int, void *, int, void *, int *, int *))solver_sym(
          p, "getrf", "getrf", "");
  char *o = out->data;
  int *pv = ipiv->data, *in = info->data;
  int step = m < n ? m : n;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("getrf", run(handle, m, n, o, m, work, pv, in));
    o += elt * (size_t)m * (size_t)n;
    pv += step;
    in++;
  }
  solver_release(handle);
  return finish(k);
}

ENTER(h_geqrf)
  Buffer *a = ARG(0), *out = RET(0), *tau = RET(1);
  int64_t batch = batch_of(a, 2);
  int m = (int)a->dims[a->rank - 2], n = (int)a->dims[a->rank - 1];
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  int mn = m < n ? m : n;
  if (batch > 1 && m / batch <= 128 && n / batch <= 128) {
    void *handle = blas_handle(k);
    CHECK_PTR(handle);
    CHECK("copy", copy_input(k, out->data, a));
    void **op = batch_pointers(k, out->data, batch, elt * (size_t)m * (size_t)n);
    CHECK_PTR(op);
    void **tp = batch_pointers(k, tau->data, batch, elt * (size_t)mn);
    CHECK_PTR(tp);
    int host_info = 0;
    int (*batched)(void *, int, int, void **, int, void **, int *, int) =
        (int (*)(void *, int, int, void **, int, void **, int *, int))blas_sym(
            p, "geqrfBatched");
    int status = batched(handle, m, n, op, m, tp, &host_info, (int)batch);
    blas_release(handle);
    CHECK("geqrfBatched", status);
    return finish(k);
  }
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  int lwork = 0;
  int (*size)(void *, int, int, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, int *))solver_sym(p, "geqrf", "geqrf",
                                                                "_bufferSize");
  CHECK("geqrf_bufferSize", size(handle, m, n, NULL, m, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  int *info = scratch(k, sizeof(int), sizeof(int));
  CHECK_PTR(info);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, int, int, void *, int, void *, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, void *, void *, int, int *))solver_sym(
          p, "geqrf", "geqrf", "");
  char *o = out->data, *t = tau->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("geqrf", run(handle, m, n, o, m, t, work, lwork, info));
    o += elt * (size_t)m * (size_t)n;
    t += elt * (size_t)mn;
  }
  solver_release(handle);
  return finish(k);
}

typedef struct {
  double re, im;
} cx;

static double fused(double a, double b, double c, int single) {
  if (single) return (double)fmaf((float)a, (float)b, (float)c);
  return fma(a, b, c);
}

static double round_to(double v, int single) {
  return single ? (double)(float)v : v;
}

static cx load_elt(const char *base, int64_t i, int dtype) {
  cx v = {0.0, 0.0};
  switch (dtype) {
    case DT_F32:
      v.re = ((const float *)base)[i];
      break;
    case DT_F64:
      v.re = ((const double *)base)[i];
      break;
    case DT_C64:
      v.re = ((const float *)base)[2 * i];
      v.im = ((const float *)base)[2 * i + 1];
      break;
    default:
      v.re = ((const double *)base)[2 * i];
      v.im = ((const double *)base)[2 * i + 1];
      break;
  }
  return v;
}

static void store_elt(char *base, int64_t i, int dtype, cx v) {
  switch (dtype) {
    case DT_F32:
      ((float *)base)[i] = (float)v.re;
      break;
    case DT_F64:
      ((double *)base)[i] = v.re;
      break;
    case DT_C64:
      ((float *)base)[2 * i] = (float)v.re;
      ((float *)base)[2 * i + 1] = (float)v.im;
      break;
    default:
      ((double *)base)[2 * i] = v.re;
      ((double *)base)[2 * i + 1] = v.im;
      break;
  }
}

static cx mul_add(cx a, cx b, cx c, int single, int cplx) {
  cx r;
  if (!cplx) {
    r.re = fused(a.re, b.re, c.re, single);
    r.im = 0.0;
    return r;
  }
  double re = round_to(fused(a.re, b.re, -round_to(a.im * b.im, single), single), single);
  double im = round_to(fused(a.re, b.im, round_to(a.im * b.re, single), single), single);
  r.re = round_to(re + c.re, single);
  r.im = round_to(im + c.im, single);
  return r;
}

static cx neg(cx a) {
  cx r = {-a.re, -a.im};
  return r;
}

static cx conjugate(cx a) {
  cx r = {a.re, -a.im};
  return r;
}

static void reflect_left(char *a, char *tau, char *c, int dtype, int m, int n, int kk,
                         int lda, int transpose) {
  int single = dtype == DT_F32 || dtype == DT_C64;
  int cplx = dtype == DT_C64 || dtype == DT_C128;
  cx zero = {0.0, 0.0};
  for (int col = 0; col < n; col++) {
    for (int step = 0; step < kk; step++) {
      int i = transpose ? step : kk - 1 - step;
      cx t = load_elt(tau, i, dtype);
      if (transpose) t = conjugate(t);
      cx w = load_elt(c, i + (int64_t)col * m, dtype);
      for (int r = i + 1; r < m; r++)
        w = mul_add(conjugate(load_elt(a, r + (int64_t)i * lda, dtype)),
                    load_elt(c, r + (int64_t)col * m, dtype), w, single, cplx);
      cx tw = mul_add(t, w, zero, single, cplx);
      cx head = load_elt(c, i + (int64_t)col * m, dtype);
      head.re = round_to(head.re - tw.re, single);
      head.im = round_to(head.im - tw.im, single);
      store_elt(c, i + (int64_t)col * m, dtype, head);
      for (int r = i + 1; r < m; r++) {
        cx u = mul_add(neg(load_elt(a, r + (int64_t)i * lda, dtype)), tw,
                       load_elt(c, r + (int64_t)col * m, dtype), single, cplx);
        store_elt(c, r + (int64_t)col * m, dtype, u);
      }
    }
  }
}

static void reflect_right(char *a, char *tau, char *c, int dtype, int m, int n, int kk,
                          int lda, int transpose) {
  int single = dtype == DT_F32 || dtype == DT_C64;
  int cplx = dtype == DT_C64 || dtype == DT_C128;
  cx zero = {0.0, 0.0};
  for (int row = 0; row < m; row++) {
    for (int step = 0; step < kk; step++) {
      int i = transpose ? kk - 1 - step : step;
      cx t = load_elt(tau, i, dtype);
      if (transpose) t = conjugate(t);
      cx u = load_elt(c, row + (int64_t)i * m, dtype);
      for (int col = i + 1; col < n; col++)
        u = mul_add(load_elt(c, row + (int64_t)col * m, dtype),
                    load_elt(a, col + (int64_t)i * lda, dtype), u, single, cplx);
      cx ut = mul_add(u, t, zero, single, cplx);
      cx head = load_elt(c, row + (int64_t)i * m, dtype);
      head.re = round_to(head.re - ut.re, single);
      head.im = round_to(head.im - ut.im, single);
      store_elt(c, row + (int64_t)i * m, dtype, head);
      for (int col = i + 1; col < n; col++) {
        cx v = mul_add(neg(ut), conjugate(load_elt(a, col + (int64_t)i * lda, dtype)),
                       load_elt(c, row + (int64_t)col * m, dtype), single, cplx);
        store_elt(c, row + (int64_t)col * m, dtype, v);
      }
    }
  }
}

static void *small_batched(Kernel *k, Buffer *a, Buffer *tau, char *out, int dtype,
                           int64_t batch, int m, int n, int kk, int lda,
                           int64_t a_stride, int left, int transpose, int orgqr) {
  size_t elt = elt_bytes(dtype);
  size_t a_bytes = byte_size(a), tau_bytes = byte_size(tau);
  size_t c_bytes = elt * (size_t)batch * (size_t)m * (size_t)n;
  char *ha = malloc(a_bytes), *ht = malloc(tau_bytes), *hc = calloc(c_bytes, 1);
  CHECK("sync", stream_sync(k));
  CHECK("copy", device_copy(k, ha, a->data, a_bytes, COPY_DEVICE_TO_HOST));
  CHECK("copy", device_copy(k, ht, tau->data, tau_bytes, COPY_DEVICE_TO_HOST));
  if (!orgqr) CHECK("copy", device_copy(k, hc, out, c_bytes, COPY_DEVICE_TO_HOST));
  CHECK("sync", stream_sync(k));
  for (int64_t b = 0; b < batch; b++) {
    char *ca = ha + (size_t)(b * a_stride) * elt;
    char *ct = ht + (size_t)(b * kk) * elt;
    char *cc = hc + (size_t)b * (size_t)m * (size_t)n * elt;
    if (orgqr) {
      cx one = {1.0, 0.0};
      for (int col = 0; col < n && col < m; col++)
        store_elt(cc, col + (int64_t)col * m, dtype, one);
    }
    if (left)
      reflect_left(ca, ct, cc, dtype, m, n, kk, lda, transpose);
    else
      reflect_right(ca, ct, cc, dtype, m, n, kk, lda, transpose);
  }
  CHECK("copy", device_copy(k, out, hc, c_bytes, COPY_HOST_TO_DEVICE));
  CHECK("sync", stream_sync(k));
  free(ha);
  free(ht);
  free(hc);
  return finish(k);
}

ENTER(h_orgqr)
  Buffer *a = ARG(0), *tau = ARG(1), *out = RET(0);
  int64_t batch = batch_of(a, 2);
  int m = (int)a->dims[a->rank - 2], n = (int)a->dims[a->rank - 1];
  int kk = (int)tau->dims[tau->rank - 1];
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  if (batch > 1 && m <= 128 && n <= 128)
    return small_batched(k, a, tau, out->data, a->dtype, batch, m, n, kk, m,
                         (int64_t)m * n, 1, 0, 1);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  int lwork = 0;
  int (*size)(void *, int, int, int, void *, int, void *, int *) =
      (int (*)(void *, int, int, int, void *, int, void *, int *))solver_sym(
          p, "orgqr", "ungqr", "_bufferSize");
  CHECK("orgqr_bufferSize", size(handle, m, n, kk, NULL, m, NULL, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  int *info = scratch(k, sizeof(int), sizeof(int));
  CHECK_PTR(info);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, int, int, int, void *, int, void *, void *, int, int *) =
      (int (*)(void *, int, int, int, void *, int, void *, void *, int, int *))
          solver_sym(p, "orgqr", "ungqr", "");
  char *o = out->data, *t = tau->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("orgqr", run(handle, m, n, kk, o, m, t, work, lwork, info));
    o += elt * (size_t)m * (size_t)n;
    t += elt * (size_t)kk;
  }
  solver_release(handle);
  return finish(k);
}

ENTER(h_ormqr)
  Buffer *a = ARG(0), *tau = ARG(1), *c = ARG(2), *out = RET(0);
  int64_t batch = batch_of(c, 2);
  int left = attr_flag(f, "left"), transpose = attr_flag(f, "transpose");
  int a_rows = (int)a->dims[a->rank - 2], a_cols = (int)a->dims[a->rank - 1];
  int m = (int)c->dims[c->rank - 2], n = (int)c->dims[c->rank - 1];
  int kk = (int)tau->dims[tau->rank - 1];
  int lda = a_rows;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  int cplx = a->dtype == DT_C64 || a->dtype == DT_C128;
  CHECK("copy", copy_input(k, out->data, c));
  if (batch > 1 && m <= 128 && n <= 128)
    return small_batched(k, a, tau, out->data, a->dtype, batch, m, n, kk, lda,
                         (int64_t)lda * a_cols, left, transpose, 0);
  int side = left ? SIDE_LEFT : SIDE_RIGHT;
  int trans = transpose ? (cplx ? OP_C : OP_T) : OP_N;
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  int lwork = 0;
  int (*size)(void *, int, int, int, int, int, void *, int, void *, void *, int, int *) =
      (int (*)(void *, int, int, int, int, int, void *, int, void *, void *, int,
               int *))solver_sym(p, "ormqr", "unmqr", "_bufferSize");
  CHECK("ormqr_bufferSize", size(handle, side, trans, m, n, kk, NULL,
                                 left ? m : n, NULL, NULL, m, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  int *info = scratch(k, sizeof(int), sizeof(int));
  CHECK_PTR(info);
  int (*run)(void *, int, int, int, int, int, void *, int, void *, void *, int, void *,
             int, int *) =
      (int (*)(void *, int, int, int, int, int, void *, int, void *, void *, int, void *,
               int, int *))solver_sym(p, "ormqr", "unmqr", "");
  char *ap = a->data, *t = tau->data, *o = out->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("ormqr", run(handle, side, trans, m, n, kk, ap, lda, t, o, m, work, lwork, info));
    o += elt * (size_t)m * (size_t)n;
    ap += elt * (size_t)a_rows * (size_t)a_cols;
    t += elt * (size_t)kk;
  }
  solver_release(handle);
  return finish(k);
}

ENTER(h_potrf)
  Buffer *a = ARG(0), *out = RET(0), *info = RET(1);
  int64_t batch = batch_of(a, 2);
  int n = (int)a->dims[a->rank - 1];
  int uplo = attr_flag(f, "lower") ? FILL_LOWER : FILL_UPPER;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  if (batch > 1) {
    CHECK("copy", copy_input(k, out->data, a));
    void **ptrs = batch_pointers(k, out->data, batch, elt * (size_t)n * (size_t)n);
    CHECK_PTR(ptrs);
    int (*batched)(void *, int, int, void **, int, int *, int) =
        (int (*)(void *, int, int, void **, int, int *, int))solver_sym(
            p, "potrfBatched", "potrfBatched", "");
    CHECK("potrfBatched", batched(handle, uplo, n, ptrs, n, info->data, (int)batch));
    solver_release(handle);
    return finish(k);
  }
  int lwork = 0;
  int (*size)(void *, int, int, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, int *))solver_sym(p, "potrf", "potrf",
                                                                "_bufferSize");
  CHECK("potrf_bufferSize", size(handle, uplo, n, NULL, n, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, int, int, void *, int, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, void *, int, int *))solver_sym(
          p, "potrf", "potrf", "");
  char *o = out->data;
  int *in = info->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("potrf", run(handle, uplo, n, o, n, work, lwork, in));
    o += elt * (size_t)n * (size_t)n;
    in++;
  }
  solver_release(handle);
  return finish(k);
}

static int solver_dtype_code(int dtype) {
  switch (dtype) {
    case DT_F32:
      return 0;
    case DT_F64:
      return 1;
    case DT_C64:
      return 4;
    default:
      return 5;
  }
}

ENTER(h_syevd)
  Buffer *a = ARG(0), *out = RET(0), *w = RET(1), *info = RET(2);
  int64_t batch = batch_of(a, 2);
  int n = (int)a->dims[a->rank - 1];
  int algorithm = attr_u8(f, "algorithm");
  int uplo = attr_flag(f, "lower") ? FILL_LOWER : FILL_UPPER;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  CHECK("copy", copy_input(k, out->data, a));
  if (algorithm == 2 || (algorithm == 0 && n <= 32)) {
    void *params = NULL;
    int (*create)(void **) =
        (int (*)(void **))need(lib_solver, "cusolverDnCreateSyevjInfo");
    int (*destroy)(void *) =
        (int (*)(void *))need(lib_solver, "cusolverDnDestroySyevjInfo");
    CHECK("syevj info", create(&params));
    int lwork = 0;
    if (batch == 1) {
      int (*size)(void *, int, int, int, void *, int, void *, int *, void *) =
          (int (*)(void *, int, int, int, void *, int, void *, int *, void *))
              solver_sym(p, "syevj", "heevj", "_bufferSize");
      CHECK("syevj_bufferSize",
            size(handle, EIG_VECTOR, uplo, n, NULL, n, NULL, &lwork, params));
      void *work = scratch(k, elt * (size_t)lwork, elt);
      CHECK_PTR(work);
      int (*run)(void *, int, int, int, void *, int, void *, void *, int, int *, void *) =
          (int (*)(void *, int, int, int, void *, int, void *, void *, int, int *, void *))
              solver_sym(p, "syevj", "heevj", "");
      CHECK("syevj", run(handle, EIG_VECTOR, uplo, n, out->data, n, w->data, work, lwork,
                         info->data, params));
    } else {
      int (*size)(void *, int, int, int, void *, int, void *, int *, void *, int) =
          (int (*)(void *, int, int, int, void *, int, void *, int *, void *, int))
              solver_sym(p, "syevjBatched", "heevjBatched", "_bufferSize");
      CHECK("syevjBatched_bufferSize", size(handle, EIG_VECTOR, uplo, n, NULL, n, NULL,
                                            &lwork, params, (int)batch));
      void *work = scratch(k, elt * (size_t)lwork, elt);
      CHECK_PTR(work);
      int (*run)(void *, int, int, int, void *, int, void *, void *, int, int *, void *,
                 int) =
          (int (*)(void *, int, int, int, void *, int, void *, void *, int, int *, void *,
                   int))solver_sym(p, "syevjBatched", "heevjBatched", "");
      CHECK("syevjBatched", run(handle, EIG_VECTOR, uplo, n, out->data, n, w->data, work,
                                lwork, info->data, params, (int)batch));
    }
    destroy(params);
  } else {
    int a_type = solver_dtype_code(a->dtype), w_type = solver_dtype_code(w->dtype);
    void *params = NULL;
    int (*create)(void **) =
        (int (*)(void **))need(lib_solver, "cusolverDnCreateParams");
    int (*destroy)(void *) =
        (int (*)(void *))need(lib_solver, "cusolverDnDestroyParams");
    CHECK("params", create(&params));
    int64_t batch_step = 1;
    int64_t matrix_size = (int64_t)n * n * (int64_t)elt;
    if (n > 0) {
      batch_step = (int64_t)INT_MAX / matrix_size;
      if (batch_step < 1) batch_step = 1;
      if (batch_step >= 32 * 1024) batch_step = 32 * 1024;
    }
    size_t on_device = 0, on_host = 0;
    int64_t first = batch < batch_step ? batch : batch_step;
    int (*size)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                size_t *, size_t *, int64_t) =
        (int (*)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                 size_t *, size_t *, int64_t))need(lib_solver,
                                                   "cusolverDnXsyevBatched_bufferSize");
    CHECK("syevBatched_bufferSize",
          size(handle, params, EIG_VECTOR, uplo, n, a_type, NULL, n, w_type, NULL, a_type,
               &on_device, &on_host, first));
    void *device_work = scratch(k, on_device, 64);
    CHECK_PTR(device_work);
    char *host_work = malloc(on_host ? on_host : 1);
    int (*batched)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                   void *, size_t, void *, size_t, int *, int64_t) =
        (int (*)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                 void *, size_t, void *, size_t, int *, int64_t))need(
            lib_solver, "cusolverDnXsyevBatched");
    int (*single)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                  void *, size_t, void *, size_t, int *) =
        (int (*)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int,
                 void *, size_t, void *, size_t, int *))need(lib_solver,
                                                             "cusolverDnXsyevd");
    char *o = out->data, *wd = w->data;
    int *in = info->data;
    size_t out_step = (size_t)n * (size_t)n * elt * (size_t)batch_step;
    size_t w_step = (size_t)n * real_bytes(a->dtype) * (size_t)batch_step;
    for (int64_t i = 0; i < batch; i += batch_step) {
      int64_t count = batch - i < batch_step ? batch - i : batch_step;
      if (batch_step > 1)
        CHECK("syevBatched",
              batched(handle, params, EIG_VECTOR, uplo, n, a_type, o, n, w_type, wd, a_type,
                      device_work, on_device, host_work, on_host, in, count));
      else
        CHECK("syevd", single(handle, params, EIG_VECTOR, uplo, n, a_type, o, n, w_type, wd,
                              a_type, device_work, on_device, host_work, on_host, in));
      o += out_step;
      wd += w_step;
      in += batch_step;
    }
    CHECK("sync", stream_sync(k));
    free(host_work);
    destroy(params);
  }
  solver_release(handle);
  return finish(k);
}

ENTER(h_syrk)
  Buffer *a = ARG(0), *c_in = ARG(1), *alpha = ARG(2), *beta = ARG(3), *c_out = RET(0);
  int64_t batch = batch_of(a, 2);
  int transpose = attr_flag(f, "transpose");
  int rows = (int)a->dims[a->rank - 2], cols = (int)a->dims[a->rank - 1];
  int n = transpose ? cols : rows, kk = transpose ? rows : cols;
  int trans = transpose ? OP_N : OP_T;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  double host_alpha[2], host_beta[2];
  CHECK("copy", device_copy(k, host_alpha, alpha->data, elt, COPY_DEVICE_TO_HOST));
  CHECK("copy", device_copy(k, host_beta, beta->data, elt, COPY_DEVICE_TO_HOST));
  CHECK("sync", stream_sync(k));
  CHECK("copy", copy_input(k, c_out->data, c_in));
  void *handle = blas_handle(k);
  CHECK_PTR(handle);
  int (*run)(void *, int, int, int, int, const void *, const void *, int, const void *,
             void *, int) =
      (int (*)(void *, int, int, int, int, const void *, const void *, int, const void *,
               void *, int))blas_sym(p, "syrk_v2");
  char *ap = a->data, *o = c_out->data;
  int lda = trans == OP_N ? n : kk;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("syrk", run(handle, FILL_UPPER, trans, n, kk, host_alpha, ap, lda, host_beta, o, n));
    ap += elt * (size_t)kk * (size_t)n;
    o += elt * (size_t)n * (size_t)n;
  }
  blas_release(handle);
  CHECK("sync", stream_sync(k));
  return finish(k);
}

ENTER(h_gesvdj)
  Buffer *a = ARG(0), *out = RET(0), *s = RET(1), *u = RET(2), *v = RET(3), *info = RET(4);
  int64_t batch = batch_of(a, 2);
  int m = (int)a->dims[a->rank - 2], n = (int)a->dims[a->rank - 1];
  int full_matrices = attr_flag(f, "full_matrices");
  int job = attr_flag(f, "compute_uv") ? EIG_VECTOR : EIG_NOVECTOR;
  int econ = full_matrices ? 0 : 1;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  void *params = NULL;
  int (*create)(void **) =
      (int (*)(void **))need(lib_solver, "cusolverDnCreateGesvdjInfo");
  int (*destroy)(void *) =
      (int (*)(void *))need(lib_solver, "cusolverDnDestroyGesvdjInfo");
  CHECK("gesvdj info", create(&params));
  CHECK("copy", copy_input(k, out->data, a));
  int lwork = 0;
  if (batch <= 1 || batch > INT_MAX || m > 32 || n > 32 || econ) {
    int (*size)(void *, int, int, int, int, void *, int, void *, void *, int, void *, int,
                int *, void *) =
        (int (*)(void *, int, int, int, int, void *, int, void *, void *, int, void *, int,
                 int *, void *))solver_sym(p, "gesvdj", "gesvdj", "_bufferSize");
    CHECK("gesvdj_bufferSize", size(handle, job, econ, m, n, NULL, m, NULL, NULL, m, NULL, n,
                                    &lwork, params));
    void *work = scratch(k, elt * (size_t)lwork, elt);
    CHECK_PTR(work);
    int mn = m < n ? m : n;
    size_t u_step = elt * (size_t)m * (size_t)(full_matrices ? m : mn);
    size_t v_step = elt * (size_t)n * (size_t)(full_matrices ? n : mn);
    int (*run)(void *, int, int, int, int, void *, int, void *, void *, int, void *, int,
               void *, int, int *, void *) =
        (int (*)(void *, int, int, int, int, void *, int, void *, void *, int, void *, int,
                 void *, int, int *, void *))solver_sym(p, "gesvdj", "gesvdj", "");
    char *o = out->data, *sd = s->data, *ud = u->data, *vd = v->data;
    int *in = info->data;
    for (int64_t i = 0; i < batch; i++) {
      CHECK("gesvdj", run(handle, job, econ, m, n, o, m, sd, ud, m, vd, n, work, lwork, in,
                          params));
      o += elt * (size_t)m * (size_t)n;
      sd += real_bytes(a->dtype) * (size_t)mn;
      ud += u_step;
      vd += v_step;
      in++;
    }
  } else {
    int (*size)(void *, int, int, int, void *, int, void *, void *, int, void *, int, int *,
                void *, int) =
        (int (*)(void *, int, int, int, void *, int, void *, void *, int, void *, int, int *,
                 void *, int))solver_sym(p, "gesvdjBatched", "gesvdjBatched", "_bufferSize");
    CHECK("gesvdjBatched_bufferSize", size(handle, job, m, n, NULL, m, NULL, NULL, m, NULL, n,
                                           &lwork, params, (int)batch));
    void *work = scratch(k, elt * (size_t)lwork, elt);
    CHECK_PTR(work);
    int (*run)(void *, int, int, int, void *, int, void *, void *, int, void *, int, void *,
               int, int *, void *, int) =
        (int (*)(void *, int, int, int, void *, int, void *, void *, int, void *, int, void *,
                 int, int *, void *, int))solver_sym(p, "gesvdjBatched", "gesvdjBatched", "");
    CHECK("gesvdjBatched", run(handle, job, m, n, out->data, m, s->data, u->data, m, v->data, n,
                               work, lwork, info->data, params, (int)batch));
  }
  destroy(params);
  solver_release(handle);
  return finish(k);
}

ENTER(h_sytrd)
  Buffer *a = ARG(0), *out = RET(0), *d = RET(1), *e = RET(2), *tau = RET(3), *info = RET(4);
  int64_t batch = batch_of(a, 2);
  int n = (int)a->dims[a->rank - 1];
  int uplo = attr_flag(f, "lower") ? FILL_LOWER : FILL_UPPER;
  char p = prec_of(a->dtype);
  size_t elt = elt_bytes(a->dtype), real = real_bytes(a->dtype);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  int lwork = 0;
  int (*size)(void *, int, int, void *, int, void *, void *, void *, int *) =
      (int (*)(void *, int, int, void *, int, void *, void *, void *, int *))solver_sym(
          p, "sytrd", "hetrd", "_bufferSize");
  CHECK("sytrd_bufferSize", size(handle, uplo, n, NULL, n, NULL, NULL, NULL, &lwork));
  void *work = scratch(k, elt * (size_t)lwork, elt);
  CHECK_PTR(work);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, int, int, void *, int, void *, void *, void *, void *, int, int *) =
      (int (*)(void *, int, int, void *, int, void *, void *, void *, void *, int, int *))
          solver_sym(p, "sytrd", "hetrd", "");
  char *o = out->data, *dd = d->data, *ed = e->data, *td = tau->data;
  int *in = info->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("sytrd", run(handle, uplo, n, o, n, dd, ed, td, work, lwork, in));
    o += elt * (size_t)n * (size_t)n;
    dd += real * (size_t)n;
    ed += real * (size_t)(n - 1);
    td += elt * (size_t)(n - 1);
    in++;
  }
  solver_release(handle);
  return finish(k);
}

ENTER(h_geev)
  Buffer *a = ARG(0), *out = RET(0), *w = RET(1), *vl = RET(2), *vr = RET(3), *info = RET(4);
  int64_t batch = batch_of(a, 2);
  int64_t n = a->dims[a->rank - 1];
  int left = attr_flag(f, "left"), right = attr_flag(f, "right");
  int jobvl = left ? EIG_VECTOR : EIG_NOVECTOR, jobvr = right ? EIG_VECTOR : EIG_NOVECTOR;
  int a_type = solver_dtype_code(a->dtype), w_type = solver_dtype_code(w->dtype);
  size_t elt = elt_bytes(a->dtype);
  void *handle = solver_handle(k);
  CHECK_PTR(handle);
  void *params = NULL;
  int (*create)(void **) = (int (*)(void **))need(lib_solver, "cusolverDnCreateParams");
  int (*destroy)(void *) = (int (*)(void *))need(lib_solver, "cusolverDnDestroyParams");
  CHECK("params", create(&params));
  size_t on_device = 0, on_host = 0;
  int (*size)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int, void *,
              int64_t, int, void *, int64_t, int, size_t *, size_t *) =
      (int (*)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int, void *,
               int64_t, int, void *, int64_t, int, size_t *, size_t *))need(
          lib_solver, "cusolverDnXgeev_bufferSize");
  CHECK("geev_bufferSize", size(handle, params, jobvl, jobvr, n, a_type, NULL, n, w_type, NULL,
                                a_type, NULL, n, a_type, NULL, n, a_type, &on_device, &on_host));
  void *device_work = scratch(k, on_device, 64);
  CHECK_PTR(device_work);
  char *host_work = malloc(on_host ? on_host : 1);
  CHECK("copy", copy_input(k, out->data, a));
  int (*run)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int, void *,
             int64_t, int, void *, int64_t, int, void *, size_t, void *, size_t, int *) =
      (int (*)(void *, void *, int, int, int64_t, int, void *, int64_t, int, void *, int, void *,
               int64_t, int, void *, int64_t, int, void *, size_t, void *, size_t, int *))need(
          lib_solver, "cusolverDnXgeev");
  size_t out_step = (size_t)n * (size_t)n * elt;
  size_t w_step = (size_t)w->dims[w->rank - 1] * elt_bytes(w->dtype);
  char *o = out->data, *wd = w->data, *vld = vl->data, *vrd = vr->data;
  int *in = info->data;
  for (int64_t i = 0; i < batch; i++) {
    CHECK("geev", run(handle, params, jobvl, jobvr, n, a_type, o, n, w_type, wd, a_type, vld, n,
                      a_type, vrd, n, a_type, device_work, on_device, host_work, on_host, in));
    o += out_step;
    wd += w_step;
    vrd += out_step;
    in++;
  }
  CHECK("sync", stream_sync(k));
  free(host_work);
  destroy(params);
  solver_release(handle);
  return finish(k);
}

ENTER(h_lu_pivots)
  Buffer *pivots = ARG(0), *perm = RET(0);
  int64_t batch = batch_of(pivots, 1);
  int pivot_size = (int)pivots->dims[pivots->rank - 1];
  int perm_size = (int)perm->dims[perm->rank - 1];
  size_t pivot_bytes = sizeof(int32_t) * (size_t)batch * (size_t)pivot_size;
  size_t perm_bytes = sizeof(int32_t) * (size_t)batch * (size_t)perm_size;
  int32_t *hp = malloc(pivot_bytes ? pivot_bytes : 1);
  int32_t *hq = malloc(perm_bytes ? perm_bytes : 1);
  CHECK("copy", device_copy(k, hp, pivots->data, pivot_bytes, COPY_DEVICE_TO_HOST));
  CHECK("sync", stream_sync(k));
  for (int64_t b = 0; b < batch; b++) {
    int32_t *q = hq + b * perm_size;
    const int32_t *pv = hp + b * pivot_size;
    for (int i = 0; i < perm_size; i++) q[i] = i;
    for (int i = 0; i < pivot_size; i++) {
      int32_t j = pv[i], tmp = q[j];
      q[j] = q[i];
      q[i] = tmp;
    }
  }
  CHECK("copy", device_copy(k, perm->data, hq, perm_bytes, COPY_HOST_TO_DEVICE));
  CHECK("sync", stream_sync(k));
  free(hp);
  free(hq);
  return finish(k);
}

ENTER(h_cholesky_update)
  Buffer *mi = ARG(0), *vi = ARG(1), *mo = RET(0), *vo = RET(1);
  int single = mi->dtype == DT_F32;
  int n = (int)mi->dims[mi->rank - 1];
  size_t m_bytes = byte_size(mi), v_bytes = byte_size(vi);
  CHECK("copy", copy_input(k, mo->data, mi));
  CHECK("copy", copy_input(k, vo->data, vi));
  char *hm = malloc(m_bytes), *hv = malloc(v_bytes);
  double *cs = calloc((size_t)n, sizeof(double)), *ss = calloc((size_t)n, sizeof(double));
  CHECK("copy", device_copy(k, hm, mo->data, m_bytes, COPY_DEVICE_TO_HOST));
  CHECK("copy", device_copy(k, hv, vo->data, v_bytes, COPY_DEVICE_TO_HOST));
  CHECK("sync", stream_sync(k));
  int dt = mi->dtype;
  for (int step = 0; step < 2 * n; step++) {
    for (int t = 0; t < n; t++) {
      int i = step - t;
      if (i < t || i >= n) continue;
      if (i == t) {
        double da = load_elt(hm, (int64_t)t * n + t, dt).re;
        double db = load_elt(hv, t, dt).re;
        if (db == 0) {
          cs[t] = 1.0;
          ss[t] = 0.0;
        } else {
          double denom = fmax(fabs(da), fabs(db));
          double aa = round_to(da / denom, single), bb = round_to(db / denom, single);
          double rh = single ? (double)(1.0f / hypotf((float)aa, (float)bb)) : 1.0 / hypot(aa, bb);
          cs[t] = round_to(aa * rh, single);
          ss[t] = -round_to(bb * rh, single);
        }
      }
      double rv = load_elt(hm, (int64_t)t * n + i, dt).re;
      double uv = load_elt(hv, i, dt).re;
      double ri = fused(cs[t], rv, -round_to(ss[t] * uv, single), single);
      double un = fused(ss[t], rv, round_to(cs[t] * uv, single), single);
      cx store_u = {un, 0.0}, store_r = {ri, 0.0};
      store_elt(hv, i, dt, store_u);
      store_elt(hm, (int64_t)t * n + i, dt, store_r);
    }
  }
  CHECK("copy", device_copy(k, mo->data, hm, m_bytes, COPY_HOST_TO_DEVICE));
  CHECK("copy", device_copy(k, vo->data, hv, v_bytes, COPY_HOST_TO_DEVICE));
  CHECK("sync", stream_sync(k));
  free(hm);
  free(hv);
  free(cs);
  free(ss);
  return finish(k);
}

ENTER(h_hybrid_geqp3)
  Buffer *a = ARG(0), *jp = ARG(1), *out = RET(0), *jout = RET(1), *tau = RET(2);
  int64_t batch = batch_of(a, 2);
  int m = (int)a->dims[a->rank - 2], n = (int)a->dims[a->rank - 1];
  int mn = m < n ? m : n;
  size_t elt = elt_bytes(a->dtype);
  size_t a_bytes = byte_size(a), j_bytes = byte_size(jp), t_bytes = byte_size(tau);
  char *ha = malloc(a_bytes ? a_bytes : 1);
  char *hj = malloc(j_bytes ? j_bytes : 1);
  char *ht = malloc(t_bytes ? t_bytes : 1);
  CHECK("sync", stream_sync(k));
  CHECK("copy", device_copy(k, ha, a->data, a_bytes, COPY_DEVICE_TO_HOST));
  CHECK("copy", device_copy(k, hj, jp->data, j_bytes, COPY_DEVICE_TO_HOST));
  CHECK("sync", stream_sync(k));
  char prec = (char)(prec_of(a->dtype) - 'A' + 'a');
  for (int64_t b = 0; b < batch; b++)
    ojax_lapack_call_geqp3(prec, m, n, ha + (size_t)b * elt * (size_t)m * (size_t)n, m,
                           (int32_t *)hj + b * n, ht + (size_t)b * elt * (size_t)mn);
  CHECK("copy", device_copy(k, out->data, ha, a_bytes, COPY_HOST_TO_DEVICE));
  CHECK("copy", device_copy(k, jout->data, hj, j_bytes, COPY_HOST_TO_DEVICE));
  CHECK("copy", device_copy(k, tau->data, ht, t_bytes, COPY_HOST_TO_DEVICE));
  CHECK("sync", stream_sync(k));
  free(ha);
  free(hj);
  free(ht);
  return finish(k);
}

CAMLprim value ojax_gpu_solver_available(value unit) {
  (void)unit;
  return Val_bool(load_libraries());
}

CAMLprim value ojax_gpu_solver_handler(value v_name) {
  CAMLparam1(v_name);
  CAMLlocal1(result);
  const char *name = String_val(v_name);
  static const struct {
    const char *name;
    void *(*fn)(void *);
  } table[] = {{"cusolver_getrf_ffi", h_getrf},
               {"cusolver_geqrf_ffi", h_geqrf},
               {"cusolver_orgqr_ffi", h_orgqr},
               {"cusolver_ormqr_ffi", h_ormqr},
               {"cusolver_potrf_ffi", h_potrf},
               {"cusolver_syevd_ffi", h_syevd},
               {"cusolver_syrk_ffi", h_syrk},
               {"cusolver_gesvdj_ffi", h_gesvdj},
               {"cusolver_sytrd_ffi", h_sytrd},
               {"cusolver_geev_ffi", h_geev},
               {"cu_lu_pivots_to_permutation", h_lu_pivots},
               {"cu_cholesky_update_ffi", h_cholesky_update},
               {"cuhybrid_geqp3", h_hybrid_geqp3}};
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
    if (strcmp(table[i].name, name) == 0) {
      result = caml_copy_nativeint((intnat)table[i].fn);
      CAMLreturn(result);
    }
  caml_failwith("ojax gpu solver: unknown handler");
}
