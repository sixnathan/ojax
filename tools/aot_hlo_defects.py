import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "golden", "aot_hlo_defects.tsv")
FRAMES = re.compile(r"\n\nFileNames\n.*?\n\nStackFrames\n(?:\d+ \{[^\n]*\}\n)*\n", re.S)


def strip_frames(text):
    text = FRAMES.sub("\n", text)
    text = re.sub(r", metadata=\{stack_frame_id=\d+\}", "", text)
    return re.sub(r" stack_frame_id=\d+", "", text)


def python_files(text):
    block = FRAMES.search(text)
    names = re.findall(r'^\d+ "([^"]*)"$', block.group(0), re.M) if block else []
    return sorted({os.path.basename(n) for n in names if n.endswith(".py")})


def main():
    goldens = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "goldens")
    lines = ["suite\trow\tkind\tproof\n"]
    for tag in ("x64_off", "x64_on"):
        with open(os.path.join(goldens, "aot_hlo", tag + ".json"), encoding="utf-8") as fh:
            raw = {(r["suite"], r["case_id"]): r for r in json.load(fh)["rows"]}
        with open(os.path.join(goldens, "aot_hlo_noframe", tag + ".json"), encoding="utf-8") as fh:
            bare = {(r["suite"], r["case_id"]): r for r in json.load(fh)["rows"]}
        for key in sorted(raw):
            row = raw[key]
            if "hlo" not in row or "FileNames" not in row["hlo"]:
                continue
            other = bare.get(key, {}).get("hlo")
            if other is None or strip_frames(row["hlo"]) != other or "FileNames" in other:
                sys.exit(f"aot_hlo_defects: {tag} {key} is not its frame-free text plus Python frames")
            files = ", ".join(python_files(row["hlo"]))
            lines.append(
                f"aot_hlo:{tag}\t{key[0]}/{key[1]}\tc\tJAX's compiled text carries the FileNames, FunctionNames, "
                f"FileLocations and StackFrames tables and stack_frame_id fields, the Python frames of the traced "
                f"callable ({files}); with jax_traceback_in_locations_limit 0 and a fresh function "
                f"(goldens/aot_hlo_noframe, the scored aot_hlo_noframe row) JAX prints exactly this text with only "
                f"those parts removed, checked by tools/aot_hlo_defects.py\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    sys.stdout.write(f"aot_hlo_defects: {len(lines) - 1} rows\n")


if __name__ == "__main__":
    main()
