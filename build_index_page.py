"""Generate the index page index.html from index.yaml (data embedded, page kept in sync with the yaml).

  python build_index_page.py
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
data = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
tpl = (ROOT / "index_page_template.html").read_text(encoding="utf-8")
out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
(ROOT / "index.html").write_text(out, encoding="utf-8")
print("wrote", ROOT / "index.html", len(out), "bytes")
