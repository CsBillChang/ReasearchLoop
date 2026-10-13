"""Qdrant原文索引适配：显式集合身份、确定性point ID和实际读回，不调用LLM。"""
from __future__ import annotations

import math
import re
import uuid

import httpx


def point_id(index_id: str, chunk_id: str) -> str:
    """同一版本/片段生成同一UUID，重复upsert不会新增点。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "researchloop:" + index_id + ":" + chunk_id))


class VectorIndex:
    """原文集合读写与精确范围查询；权限和融合由业务服务负责。"""

    def __init__(self, url: str, namespace: str, client: httpx.Client | None = None):
        """接收本机服务和显式命名空间，不探测、不删除既有集合。

        Args:
            url: Qdrant REST地址。
            namespace: 当前项目或隔离验证的集合前缀。
            client: 测试可注入HTTP客户端；调用者拥有其生命周期。
        """
        if not re.fullmatch(r"researchloop_[a-z0-9_]{1,64}", namespace):
            raise ValueError("INDEX_NAMESPACE_INVALID")
        self.namespace = namespace
        self.client = client or httpx.Client(base_url=url.rstrip("/"), timeout=30)
        self.owns_client = client is None

    def close(self) -> None:
        """释放自建客户端，不关闭调用方传入的客户端。"""
        if self.owns_client:
            self.client.close()

    def collection(self, model_hash: str) -> str:
        """模型完整hash参与集合身份，不把不同编码空间写进同一集合。"""
        return self.namespace + "_" + model_hash

    def _request(self, method: str, path: str, **kwargs):
        """将HTTP错误原样交给构建层记录；不在传输层自动重试写入。"""
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise ValueError("QDRANT_RESULT_NOT_OK")
        return payload["result"]

    def ensure_collection(self, name: str, dimension: int) -> None:
        """只创建缺失集合；既有维度/距离不兼容时停止，绝不覆盖集合。"""
        response = self.client.get("/collections/" + name)
        if response.status_code == 404:
            response = self.client.put("/collections/" + name, json={
                "vectors": {"size": dimension, "distance": "Cosine"}})
            # 不同索引共享模型集合，并发创建时再次读回权威配置。
            if response.status_code not in (200, 409):
                response.raise_for_status()
        else:
            response.raise_for_status()
        result = self._request("GET", "/collections/" + name)
        vectors = result["config"]["params"]["vectors"]
        if vectors.get("size") != dimension or vectors.get("distance") != "Cosine":
            raise ValueError("QDRANT_COLLECTION_CONFIG_CONFLICT")

    def upsert(self, collection: str, points: list[dict]) -> None:
        """等待该批落盘；业务ready仍须后续逐点和总数读回。"""
        self._request("PUT", f"/collections/{collection}/points", params={"wait": "true"}, json={"points": points})

    def retrieve(self, collection: str, ids: list[str]) -> dict[str, dict]:
        """按确切point ID读回payload与向量，不跨版本模糊搜索。"""
        result = self._request("POST", f"/collections/{collection}/points",
            json={"ids": ids, "with_payload": True, "with_vector": True})
        return {str(point["id"]): point for point in result}

    def count(self, collection: str, index_id: str) -> int:
        """精确统计当前index_id，不把同集合其他文档算作本次产物。"""
        result = self._request("POST", f"/collections/{collection}/points/count", json={"exact": True,
            "filter": {"must": [{"key": "index_id", "match": {"value": index_id}}]}})
        return result["count"]

    def query(self, collection: str, vector: list[float], ids: list[str], limit: int) -> list[dict]:
        """查询明确授权的point集合；空范围绝不退化成全库查询。

        Args:
            collection: 固定编码模型的集合。
            vector: 同模型归一化query向量。
            ids: 经过原文及任务权限校验的确切point ID。
            limit: 当前范围的候选上限。
        """
        if not ids or not valid_vector(vector, len(vector)) or not 1 <= limit <= 64:
            raise ValueError("VECTOR_QUERY_INVALID")
        result = self._request("POST", f"/collections/{collection}/points/query", json={
            "query": vector, "filter": {"must": [{"has_id": ids}]}, "limit": limit,
            "params": {"exact": True}, "with_payload": True, "with_vector": False})
        return result["points"]


def valid_vector(vector: object, dimension: int) -> bool:
    """拒绝错误维度、非有限值和非单位向量，避免坏编码被当成成功收据。"""
    return (isinstance(vector, list) and len(vector) == dimension
        and all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
        and abs(sum(x*x for x in vector) - 1) < .002)


def matches(actual: dict | None, expected: dict) -> bool:
    """核验身份与数值；允许Qdrant余弦归一化产生的小幅浮点差异。"""
    if not actual or actual.get("payload") != expected["payload"]:
        return False
    vector = actual.get("vector")
    return valid_vector(vector, len(expected["vector"])) and all(
        abs(x-y) < 1e-5 for x, y in zip(vector, expected["vector"]))
