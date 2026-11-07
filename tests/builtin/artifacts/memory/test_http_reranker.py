# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior tests for the HTTP scoring reranker of Memory candidates."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _rerank(handler, candidates: tuple[MemoryHit, ...], limit: int, batch_size: int = 2):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=batch_size)
            return await reranker.rerank("query", candidates, limit)

    return asyncio.run(scenario())


def test_batches_keep_the_global_rank_and_select_over_the_complete_pool() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        scores = {1: 0.2, 2: 0.9, 3: 0.9}
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]},
        )

    decision = _rerank(handler, tuple(_hit(rank) for rank in (1, 2, 3)), 2)

    assert decision.selected_ranks == (2, 3)
    assert decision.usage.requests == 2
    assert len(seen) == 2
    assert all(1 <= len(body["candidates"]) <= 2 for body in seen)
    assert sorted(item["rank"] for body in seen for item in body["candidates"]) == [1, 2, 3]
    assert all(item["text"] == _hit(item["rank"]).text for body in seen for item in body["candidates"])


def test_out_of_order_scores_select_the_same_ranks() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 2, "score": 0.1}, {"rank": 1, "score": 0.8}]})

    assert _rerank(handler, (_hit(1), _hit(2)), 1).selected_ranks == (1,)


@pytest.mark.parametrize(
    "payload",
    [
        {"scores": [], "extra": True},
        {"scores": [{"rank": 2, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]},
        {"scores": [{"rank": 1, "score": float("nan")}]},
        {"scores": [{"rank": 1, "score": float("inf")}]},
        {"scores": [{"rank": 1, "score": 0.5, "extra": True}]},
    ],
)
def test_an_invalid_response_is_rejected(payload) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    with pytest.raises(ValueError):
        _rerank(handler, (_hit(1),), 1)


def test_invalid_bounds_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}]})

    with pytest.raises(ValueError):
        _rerank(handler, (), 1)
    with pytest.raises(ValueError):
        _rerank(handler, (_hit(1),), 2)
    with pytest.raises(ValueError):
        HTTPMemoryReranker("https://reranker.invalid", client=object(), batch_size=0)


def test_a_transport_error_keeps_its_httpx_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(httpx.ConnectError):
        _rerank(handler, (_hit(1),), 1)
