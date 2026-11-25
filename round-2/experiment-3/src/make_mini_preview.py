#!/usr/bin/env python3
"""Write full / mini / preview variants of results/method_out.json.

aii-json's format script only accepts top-level arrays and refuses paths outside its own
project tree on macOS, and method_out.json is an object. So the same three variants are
produced here with the same rules:
  full    - identical copy of the original
  mini    - the first 3 examples of datasets[0].examples; metadata kept whole
  preview - mini with every string truncated to 200 characters (recursively)
"""

import json
from pathlib import Path

WS = Path(__file__).resolve().parent
SRC = WS / "results" / "method_out.json"
MAX_STRING = 200
MARK = "...[truncated]"


def truncate(v):
    if isinstance(v, list):
        return [truncate(x) for x in v]
    if isinstance(v, dict):
        return {k: truncate(x) for k, x in v.items()}
    if isinstance(v, str) and len(v) > MAX_STRING:
        return v[:MAX_STRING] + MARK
    return v


def main() -> None:
    data = json.loads(SRC.read_text())
    full = json.dumps(data, indent=2)
    mini_obj = {
        "metadata": data["metadata"],
        "datasets": [
            {"dataset": ds["dataset"], "examples": ds["examples"][:3]}
            for ds in data["datasets"]
        ],
    }
    preview_obj = truncate(mini_obj)
    (WS / "full_method_out.json").write_text(full)
    (WS / "mini_method_out.json").write_text(json.dumps(mini_obj, indent=2))
    (WS / "preview_method_out.json").write_text(json.dumps(preview_obj, indent=2))
    print("wrote full_method_out.json, mini_method_out.json, preview_method_out.json")


if __name__ == "__main__":
    main()
