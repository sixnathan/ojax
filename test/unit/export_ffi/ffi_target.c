#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  const char *ptr;
  size_t len;
} ByteSpan;

typedef struct {
  int dtype;
  void *value;
} Scalar;

typedef struct {
  int dtype;
  size_t size;
  void *data;
} Array;

typedef struct {
  size_t struct_size;
  void *extension_start;
  int dtype;
  void *data;
  int64_t rank;
  int64_t *dims;
} Buffer;

typedef struct {
  size_t struct_size;
  void *extension_start;
  int64_t size;
  int *types;
  void **items;
} Items;

typedef struct {
  size_t struct_size;
  void *extension_start;
  int64_t size;
  int *types;
  ByteSpan **names;
  void **attrs;
} Attrs;

typedef struct {
  size_t struct_size;
  void *extension_start;
  const char *message;
  int errc;
} ErrorCreateArgs;

typedef struct {
  size_t struct_size;
  void *extension_start;
  struct {
    size_t struct_size;
    void *extension_start;
    int major_version;
    int minor_version;
  } api_version;
  const void *internal_api;
  void *(*error_create)(ErrorCreateArgs *);
} Api;

typedef struct {
  size_t struct_size;
  void *extension_start;
  const Api *api;
  void *ctx;
  int stage;
  Items args;
  Items rets;
  Attrs attrs;
  void *future;
} CallFrame;

typedef struct {
  size_t struct_size;
  int type;
  void *next;
} ExtensionBase;

typedef struct {
  size_t struct_size;
  struct {
    size_t struct_size;
    void *extension_start;
    int major_version;
    int minor_version;
  } api_version;
  int traits;
  int64_t state_type_id;
} Metadata;

typedef struct {
  ExtensionBase base;
  Metadata *metadata;
} MetadataExtension;

static int answer_metadata(CallFrame *f) {
  ExtensionBase *e = f->extension_start;
  if (e == NULL || e->type != 1) return 0;
  Metadata *m = ((MetadataExtension *)e)->metadata;
  m->api_version.struct_size = sizeof m->api_version;
  m->api_version.extension_start = NULL;
  m->api_version.major_version = 0;
  m->api_version.minor_version = 2;
  m->traits = 0;
  m->state_type_id = 0;
  return 1;
}

static int64_t elements(const Buffer *b) {
  int64_t n = 1;
  for (int64_t i = 0; i < b->rank; i++) n *= b->dims[i];
  return n;
}

static void *find_attr(CallFrame *f, const char *name) {
  size_t len = strlen(name);
  for (int64_t i = 0; i < f->attrs.size; i++) {
    ByteSpan *n = f->attrs.names[i];
    if (n->len == len && memcmp(n->ptr, name, len) == 0) return f->attrs.attrs[i];
  }
  return NULL;
}

static void *fail(CallFrame *f, const char *message) {
  ErrorCreateArgs a;
  memset(&a, 0, sizeof a);
  a.struct_size = sizeof a;
  a.message = message;
  a.errc = 3;
  return f->api->error_create(&a);
}

void *ojax_scale(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *x = f->args.items[0];
  Buffer *y = f->rets.items[0];
  Scalar *scale = find_attr(f, "scale");
  Scalar *offset = find_attr(f, "offset");
  double s = scale != NULL ? *(double *)scale->value : 1.0;
  int64_t o = offset != NULL ? *(int64_t *)offset->value : 0;
  const float *in = x->data;
  float *out = y->data;
  for (int64_t i = 0; i < elements(x); i++)
    out[i] = (float)(in[i] * s + (double)o);
  return NULL;
}

void *ojax_minmax(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *x = f->args.items[0];
  const float *in = x->data;
  float *range = ((Buffer *)f->rets.items[0])->data;
  int32_t *count = ((Buffer *)f->rets.items[1])->data;
  int64_t n = elements(x);
  range[0] = in[0];
  range[1] = in[0];
  for (int64_t i = 1; i < n; i++) {
    if (in[i] < range[0]) range[0] = in[i];
    if (in[i] > range[1]) range[1] = in[i];
  }
  *count = (int32_t)n;
  return NULL;
}

void *ojax_fail(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  return fail(f, "ojax_fail was asked to fail");
}

static int32_t counter = 0;

void *ojax_bump(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  counter += 1;
  if (f->rets.size > 0) *(int32_t *)((Buffer *)f->rets.items[0])->data = counter;
  return NULL;
}

void *ojax_layout_probe(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *x = f->args.items[0];
  const int32_t *in = x->data;
  int32_t *out = ((Buffer *)f->rets.items[0])->data;
  for (int64_t i = 0; i < elements(x); i++) out[i] = in[i];
  return NULL;
}

void *ojax_fill_raw(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *y = f->rets.items[0];
  int32_t *out = y->data;
  for (int64_t i = 0; i < elements(y); i++) out[i] = (int32_t)i;
  return NULL;
}

void *ojax_alias_check(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *x = f->args.items[0];
  Buffer *y = f->rets.items[0];
  const float *in = x->data;
  float *out = y->data;
  float marker = x->data == y->data ? 100.0f : 0.0f;
  for (int64_t i = 0; i < elements(x); i++) out[i] = in[i] + marker;
  return NULL;
}

void *ojax_attr_probe(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  float *out = ((Buffer *)f->rets.items[0])->data;
  Array *arr = find_attr(f, "arr");
  ByteSpan *name = find_attr(f, "name");
  Scalar *flag = find_attr(f, "flag");
  Attrs *dict = find_attr(f, "d");
  float sum = 0;
  if (arr != NULL)
    for (size_t i = 0; i < arr->size; i++) sum += (float)((int32_t *)arr->data)[i];
  out[0] = sum;
  out[1] = name != NULL ? (float)name->len : -1.0f;
  out[2] = name != NULL && name->len > 0 ? (float)name->ptr[0] : -1.0f;
  out[3] = flag != NULL ? (float)*(uint8_t *)flag->value : -1.0f;
  out[4] = dict != NULL ? (float)dict->size : -1.0f;
  out[5] = (float)f->attrs.size;
  return NULL;
}

int64_t ojax_test_type_id = 0;
void *ojax_test_type_info[3] = {NULL, NULL, NULL};

void *ojax_type_id_read(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  *(int64_t *)((Buffer *)f->rets.items[0])->data = ojax_test_type_id;
  return NULL;
}

void *ojax_add_pair(CallFrame *f) {
  if (answer_metadata(f)) return NULL;
  Buffer *x = f->args.items[0];
  Buffer *y = f->args.items[1];
  const float *a = x->data;
  const float *b = y->data;
  float *out = ((Buffer *)f->rets.items[0])->data;
  int32_t *info = ((Buffer *)f->rets.items[1])->data;
  Buffer *info_buf = f->rets.items[1];
  int64_t ny = elements(y);
  for (int64_t i = 0; i < elements(x); i++) out[i] = a[i] + b[i % ny];
  for (int64_t i = 0; i < elements(info_buf); i += 3) {
    info[i] = (int32_t)x->rank;
    info[i + 1] = (int32_t)y->rank;
    info[i + 2] = y->rank > 0 ? (int32_t)y->dims[0] : -1;
  }
  return NULL;
}

int64_t ojax_test_other_type_id = 5000;
