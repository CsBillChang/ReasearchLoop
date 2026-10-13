# 参考配置导入工具：读取本地 DPR 来源连接配置；不把配置密钥作为用户结果输出。
"""Copy only DPR's read-only source config, never execute its scripts or import .env."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import yaml
from research_loop.providers.dpr_sources import resolve_backends


def import_config(source: Path, target: Path) -> dict:
    raw = source.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("DPR config must be a mapping")
    allowed = {"enabled", "url", "anon_key", "schema", "bm25_rpc", "use_bm25_rpc"}
    backends = resolve_backends(config)
    for source_key, backend in backends.items():
        key = backend.get("anon_key") or ""
        if not key:
            continue
        if not key.startswith("sb_publishable_"):
            try:
                payload = key.split(".")[1]
                role = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("role")
            except (ValueError, IndexError):
                role = None
            if role != "anon":
                raise ValueError(f"{source_key}: only an anonymous/public read key may be imported")
    output = {"source_backends": {s: {k: v for k, v in b.items() if k in allowed} for s, b in backends.items()},
              "upstream_config_sha256": hashlib.sha256(raw).hexdigest(), "upstream_config": str(source.resolve())}
    target.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output": str(target), "source_count": len(backends),
            "upstream_config_sha256": output["upstream_config_sha256"], "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/config/dpr-sources.json")
    args = parser.parse_args()
    print(json.dumps(import_config(args.source, args.output), ensure_ascii=False, indent=2))
