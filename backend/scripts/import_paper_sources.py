"""导入本地只读论文源配置；不执行来源脚本，不导入 .env，不输出凭据。"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import yaml
from research_loop.providers.paper_sources import resolve_backends


def import_config(source: Path, target: Path) -> dict:
    """复制合法的只读来源配置，凭据仅写入本机受限文件。

    Args:
        source: 已有 YAML 配置文件，不执行其中引用的程序。
        target: 尚不存在的本机 JSON 目标文件。

    Returns:
        目标路径、来源数量和输入 hash，不包含凭据值。

    Raises:
        ValueError: 配置格式或读取凭据角色不符合要求。
        FileExistsError: 目标已存在，拒绝覆盖。
    """
    raw = source.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("paper source config must be a mapping")
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
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/config/paper-sources.json")
    args = parser.parse_args()
    print(json.dumps(import_config(args.source, args.output), ensure_ascii=False, indent=2))
