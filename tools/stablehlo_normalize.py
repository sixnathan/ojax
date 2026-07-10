import re
import sys

_LOC = re.compile(r"\s*loc\([^)]*\)")
_MODULE = re.compile(r"module @\S+( attributes \{[^}]*\})? \{")
_RESULT_INFO = re.compile(r' \{jax\.result_info = "[^"]*"\}')
_SSA = re.compile(r"%[A-Za-z0-9_]+")


def normalize(text):
    text = _LOC.sub("", text)
    text = _MODULE.sub("module {", text, count=1)
    text = _RESULT_INFO.sub("", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    text = "\n".join(lines)
    mapping = {}
    order = []

    def repl(m):
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = len(order)
            order.append(tok)
        return "%" + str(mapping[tok])

    text = _SSA.sub(repl, text)
    return text + "\n"


def main():
    sys.stdout.write(normalize(sys.stdin.read()))


if __name__ == "__main__":
    main()
