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

"""Behavior tests for the row level capture of an offline CSV document."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import SourceDefinitionRegistry, TextEvidence
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

_DOCUMENT = b'id, name ,note\n001,Airport,first\n002,"Airport, East",say "hi"\n'


def _parse(content: bytes = _DOCUMENT, document_id: str = "airports", primary_key: str = "id"):
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


def test_rows_keep_the_original_column_order_and_field_values() -> None:
    rows = _parse()

    assert rows == (
        CSVRowInput(
            document_id="airports",
            primary_key="id",
            primary_key_value="001",
            columns=("id", " name ", "note"),
            values=("001", "Airport", "first"),
        ),
        CSVRowInput(
            document_id="airports",
            primary_key="id",
            primary_key_value="002",
            columns=("id", " name ", "note"),
            values=("002", "Airport, East", 'say "hi"'),
        ),
    )


def test_a_repeated_identity_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n1,duplicate\n")


def test_a_blank_primary_key_value_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n,empty\n")


def test_a_row_that_does_not_match_the_header_width_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n1\n")


def test_a_blank_or_repeated_header_name_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,id\n1,2\n")
    with pytest.raises(ValueError):
        _parse(b" \t,name\n1,one\n")


def test_a_missing_primary_key_column_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n", primary_key="missing")


def test_a_malformed_document_is_a_rejected_document() -> None:
    with pytest.raises(ValueError):
        _parse(b'id,name\n1,"unclosed\n')
    with pytest.raises(ValueError):
        _parse(b"id,name\n\xff,invalid\n")


def test_an_empty_document_and_a_header_only_document_differ() -> None:
    with pytest.raises(ValueError):
        _parse(b"")
    assert _parse(b"id,name\n\n") == ()


def test_the_row_identity_is_delimiter_safe_and_bounded() -> None:
    assert csv_row_source_id("doc:one", "key/one") == "csv:doc%3Aone:key%2Fone"
    assert csv_row_source_id("doc:one", "key:one") != csv_row_source_id("doc:two", "key:one")
    assert csv_row_source_id("doc:one", "key:one") != csv_row_source_id("doc:one", "key/one")

    with pytest.raises(ValueError):
        csv_row_source_id(" doc ", "key")
    with pytest.raises(ValueError):
        csv_row_source_id("doc", "x" * 260)
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id=" doc ", primary_key="id", content=b"id,name\n1,one\n"))


def test_the_projection_exposes_the_complete_row_without_changing_its_values() -> None:
    row = _parse()[1]
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(row))

    assert isinstance(source, CSVRowSource)
    assert source.source_id == source.name == "csv:airports:002"
    assert source.materialization.value == "captured"
    assert asyncio.run(registry.read(source)) == row

    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "csv-row"
    assert evidence.source_id == "csv:airports:002"
    assert evidence.content == json.dumps(
        {"id": "002", " name ": "Airport, East", "note": 'say "hi"'},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert evidence.metadata == {
        "document_id": "airports",
        "primary_key": "id",
        "primary_key_value": "002",
        "columns": ["id", " name ", "note"],
        "row": {"id": "002", " name ": "Airport, East", "note": 'say "hi"'},
    }
