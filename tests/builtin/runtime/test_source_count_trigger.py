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

"""Behavior tests for the Source count that admits a busy Scope."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, RuntimeConfig, open_builtin_contexts
from powercontext.builtin.runtime.processing_discovery import SourceCountPendingProvider
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentCapture
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME

_THRESHOLDS = ("memory_source_count_threshold", "topic_memory_source_count_threshold")


@pytest.mark.parametrize("field", _THRESHOLDS)
def test_the_count_condition_is_disabled_by_default(field: str) -> None:
    assert field in RuntimeConfig.model_fields
    assert getattr(RuntimeConfig(), field) is None


@pytest.mark.parametrize("field", _THRESHOLDS)
@pytest.mark.parametrize("value", [1, 3, 100])
def test_a_positive_integer_count_is_accepted(field: str, value: int) -> None:
    assert getattr(RuntimeConfig(**{field: value}), field) == value


@pytest.mark.parametrize("field", _THRESHOLDS)
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "3"])
def test_anything_but_a_positive_integer_count_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(**{field: value})


def test_the_pending_count_follows_the_journal_behind_the_processor_cursor(tmp_path):
    async def scenario():
        database = tmp_path / "count.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_contexts(config) as contexts:
            scope = await contexts.scopes.create(
                ScopeDraft(title="count", summary="pending count", idempotency_key="pending-count")
            )
            scoped = await contexts.get(scope.scope_id)
            for index in range(3):
                await scoped.sources.capture(ContentCapture(source_id=f"row-{index}", content=f"row {index}"))

            provider = SourceCountPendingProvider(contexts.database, SOURCE_WINDOW_TRIGGER_NAME, "memory")
            async with contexts.database.transaction() as connection:
                _, reached = await provider.over_threshold(connection, 3, limit=10)
            assert reached == frozenset({scope.scope_id})

            async with contexts.database.transaction() as connection:
                _, reached = await provider.over_threshold(connection, 4, limit=10)
            assert reached == frozenset()

    asyncio.run(scenario())


def test_a_repeated_capture_adds_no_pending_entry(tmp_path):
    async def scenario():
        database = tmp_path / "repeat.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_contexts(config) as contexts:
            scope = await contexts.scopes.create(
                ScopeDraft(title="repeat", summary="repeated capture", idempotency_key="repeated-capture")
            )
            scoped = await contexts.get(scope.scope_id)
            await scoped.sources.capture(ContentCapture(source_id="row-0", content="row 0"))
            await scoped.sources.capture(ContentCapture(source_id="row-0", content="row 0"))

            provider = SourceCountPendingProvider(contexts.database, SOURCE_WINDOW_TRIGGER_NAME, "memory")
            async with contexts.database.transaction() as connection:
                _, reached = await provider.over_threshold(connection, 2, limit=10)
            assert reached == frozenset()

    asyncio.run(scenario())
