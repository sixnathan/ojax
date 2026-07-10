import os
import re
import subprocess
import sys
import tempfile

HEADER = "reference/pjrt/pjrt_c_api.h"
OUT = "src/pjrt/pjrt_abi_assert.h"
GUARD = "OJAX_PJRT_ABI_ASSERT_H_"

VERSION_FIELDS = ["struct_size", "extension_start", "major_version", "minor_version"]
BUFFER_TYPES = ["PRED", "S8", "S16", "S32", "S64", "U8", "U16", "U32", "U64",
                "F16", "F32", "F64", "BF16", "C64", "C128"]


def api_fields(header_text):
    body = header_text[header_text.index("typedef struct PJRT_Api {"):
                       header_text.index("} PJRT_Api;")]
    return re.findall(r"_PJRT_API_STRUCT_FIELD\((\w+)\)", body)


def probe(root, fields):
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "pjrt_c_api.h"',
             "int main(void){"]
    lines.append('printf("PJRT_API_MAJOR %d\\n", (int)PJRT_API_MAJOR);')
    lines.append('printf("PJRT_API_MINOR %d\\n", (int)PJRT_API_MINOR);')
    lines.append('printf("sizeof_PJRT_Api %zu\\n", sizeof(PJRT_Api));')
    lines.append('printf("sizeof_PJRT_Api_Version %zu\\n", sizeof(PJRT_Api_Version));')
    for f in ["struct_size", "extension_start", "pjrt_api_version"]:
        lines.append('printf("api.%s %%zu\\n", offsetof(PJRT_Api, %s));' % (f, f))
    for f in VERSION_FIELDS:
        lines.append('printf("ver.%s %%zu\\n", offsetof(PJRT_Api_Version, %s));' % (f, f))
    for f in fields:
        lines.append('printf("fn.%s %%zu\\n", offsetof(PJRT_Api, %s));' % (f, f))
    for b in BUFFER_TYPES:
        lines.append('printf("bt.%s %%d\\n", (int)PJRT_Buffer_Type_%s);' % (b, b))
    lines.append("return 0;}")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.c")
        exe = os.path.join(d, "probe")
        with open(src, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        subprocess.run(["cc", "-I", os.path.join(root, "reference/pjrt"), src, "-o", exe],
                       check=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        key, val = line.split()
        values[key] = int(val)
    return values


def emit(fields, v):
    out = []
    out.append("#ifndef " + GUARD)
    out.append("#define " + GUARD)
    out.append("#include <stddef.h>")
    out.append('#include "pjrt_c_api.h"')
    out.append('_Static_assert(PJRT_API_MAJOR == %d, "PJRT_API_MAJOR drift");' % v["PJRT_API_MAJOR"])
    out.append('_Static_assert(PJRT_API_MINOR == %d, "PJRT_API_MINOR drift");' % v["PJRT_API_MINOR"])
    out.append('_Static_assert(sizeof(PJRT_Api) == %d, "PJRT_Api size drift");' % v["sizeof_PJRT_Api"])
    out.append('_Static_assert(sizeof(PJRT_Api_Version) == %d, "PJRT_Api_Version size drift");' % v["sizeof_PJRT_Api_Version"])
    for f in ["struct_size", "extension_start", "pjrt_api_version"]:
        out.append('_Static_assert(offsetof(PJRT_Api, %s) == %d, "PJRT_Api.%s offset drift");'
                   % (f, v["api." + f], f))
    for f in VERSION_FIELDS:
        out.append('_Static_assert(offsetof(PJRT_Api_Version, %s) == %d, "PJRT_Api_Version.%s offset drift");'
                   % (f, v["ver." + f], f))
    for f in fields:
        out.append('_Static_assert(offsetof(PJRT_Api, %s) == %d, "PJRT_Api.%s offset drift");'
                   % (f, v["fn." + f], f))
    for b in BUFFER_TYPES:
        out.append('_Static_assert(PJRT_Buffer_Type_%s == %d, "PJRT_Buffer_Type_%s drift");'
                   % (b, v["bt." + b], b))
    out.append("#endif")
    return "\n".join(out) + "\n"


def main():
    root = os.getcwd()
    header_text = open(os.path.join(root, HEADER)).read()
    fields = api_fields(header_text)
    v = probe(root, fields)
    with open(os.path.join(root, OUT), "w") as fh:
        fh.write(emit(fields, v))
    sys.stderr.write("wrote %s (%d fn-ptr asserts)\n" % (OUT, len(fields)))


if __name__ == "__main__":
    main()
