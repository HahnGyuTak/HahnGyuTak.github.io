# _scripts/apply_replace_inplace.py
# usage:
#   python _scripts/apply_replace_inplace.py /path/to/file.md /path/to/replace_map.yaml

import sys, re
from pathlib import Path

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr); raise

# ---- protect code/math segments so they don't get replaced ----
FENCE_RE        = re.compile(r"```[\s\S]*?```", re.DOTALL)     # fenced code blocks
INLINE_CODE_RE  = re.compile(r"`[^`]*`")                       # `inline code`
BLOCK_MATH_RE   = re.compile(r"\$\$[\s\S]*?\$\$", re.DOTALL)   # $$ ... $$
INLINE_MATH_RE  = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)  # $ ... $

def _mask_segments(text: str):
    masks = {}
    idx = 0
    def put(m):
        nonlocal idx
        key = f"__MASK_{idx}__"
        masks[key] = m.group(0)
        idx += 1
        return key

    # 순서 중요: 긴 블록부터
    text = FENCE_RE.sub(put, text)
    text = BLOCK_MATH_RE.sub(put, text)
    text = INLINE_CODE_RE.sub(put, text)
    text = INLINE_MATH_RE.sub(put, text)
    return text, masks

def _unmask(text: str, masks: dict):
    for k, v in masks.items():
        text = text.replace(k, v)
    return text

# ---- load mapping rules ----
def load_rules(yaml_path: Path):
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("YAML은 리스트 형태여야 합니다. 각 항목은 {pattern, replace[, case_sensitive, word_boundary]} 키를 가집니다.")
    rules = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "pattern" not in item or "replace" not in item:
            raise ValueError(f"{i}번째 항목이 잘못되었습니다: {item}")
        pattern = item["pattern"]
        repl = item["replace"]
        flags = re.UNICODE
        if not item.get("case_sensitive", True):
            flags |= re.IGNORECASE
        pat = pattern
        if item.get("word_boundary", False):
            pat = rf"\b(?:{pattern})\b"
        try:
            rx = re.compile(pat, flags)
        except re.error as e:
            raise ValueError(f"정규식 컴파일 오류 (pattern: {pat}): {e}")
        rules.append((rx, repl))
    return rules

def apply_rules_inplace(md_path: Path, map_path: Path):
    text = md_path.read_text(encoding="utf-8")
    rules = load_rules(map_path)

    masked, masks = _mask_segments(text)
    s = masked
    for rx, repl in rules:
        s = rx.sub(repl, s)
    s = _unmask(s, masks)

    md_path.write_text(s, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("usage: python _scripts/replace.py <markdown.md> <replace_map.yaml>", file=sys.stderr)
        sys.exit(2)

    md_path = Path(sys.argv[1])
    map_path = Path('_posts/replace_map.yaml')

    if not md_path.is_file():
        print(f"Markdown 파일을 찾을 수 없습니다: {md_path}", file=sys.stderr)
        sys.exit(1)
    if not map_path.is_file():
        print(f"매핑 YAML 파일을 찾을 수 없습니다: {map_path}", file=sys.stderr)
        sys.exit(1)

    apply_rules_inplace(md_path, map_path)
    print(f"[updated] {md_path}")

if __name__ == "__main__":
    main()