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

"""Rerank Memory candidates with an external HTTP scoring service.

The service scores the candidates of one query.  A request carries the
original rank of every candidate of its batch, so the scores of the batches
can be mapped back to the complete candidate pool and the final selection is
made once over that pool.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
_HTTP_RERANK_SCORES_KEY = "scores"


class HTTPMemoryReranker:
    """Select the final Memory hits with the scores of an HTTP service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("the rerank endpoint must be a non-empty string")  # noqa: TRY003
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("the rerank batch size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the original ranks of the best candidates of the whole pool."""

        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for batch in self._batches(candidates):
            requests += 1
            scores.update(await self._score_batch(query, batch))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _score in ordered[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
        )

    def _batches(self, candidates: tuple[MemoryHit, ...]) -> tuple[tuple[tuple[int, MemoryHit], ...], ...]:
        """Split the pool into batches that keep the original rank."""

        entries = tuple(enumerate(candidates, start=1))
        return tuple(entries[index : index + self._batch_size] for index in range(0, len(entries), self._batch_size))

    async def _score_batch(
        self,
        query: str,
        batch: tuple[tuple[int, MemoryHit], ...],
    ) -> dict[int, float]:
        """Ask the service for the scores of one batch and validate them."""

        payload = {
            "query": query,
            "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
        }
        response = await self._client.post(self._endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or set(body) != {_HTTP_RERANK_SCORES_KEY}:
            raise ValueError("the rerank response must contain exactly the scores")  # noqa: TRY003
        records = body[_HTTP_RERANK_SCORES_KEY]
        if not isinstance(records, list):
            raise ValueError("the rerank scores must be a list")  # noqa: TRY003, TRY004

        expected = {rank for rank, _hit in batch}
        scores: dict[int, float] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"rank", "score"}:
                raise ValueError("a rerank score must contain exactly the rank and the score")  # noqa: TRY003
            rank = record["rank"]
            if type(rank) is not int or rank not in expected:
                raise ValueError("a rerank score refers to a rank outside of its batch")  # noqa: TRY003
            if rank in scores:
                raise ValueError("a rerank batch reported the same rank twice")  # noqa: TRY003
            scores[rank] = _finite_score(record["score"])
        if set(scores) != expected:
            raise ValueError("a rerank batch must score every rank of its batch once")  # noqa: TRY003
        return scores


def _finite_score(value: Any) -> float:
    """Accept a finite number only, a NaN or an infinity is not a score."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a rerank score must be a number")  # noqa: TRY003, TRY004
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("a rerank score must be a finite number")  # noqa: TRY003
    return score


__all__ = ["HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
