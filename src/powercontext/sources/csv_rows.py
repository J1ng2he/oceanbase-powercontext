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

"""Row level capture of an offline CSV document.

One document is captured row by row: every row becomes an independent
Source that keeps the original column order and the original field values,
so a retrieval citation can point back to the complete original record.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_ID_PREFIX = "csv"
CSV_ROW_SOURCE_ID_SEPARATOR = ":"


class CSVDocumentInput(BaseModel):
    """One offline CSV document addressed by its primary key."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One captured row with its column order and field values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row that can be used as Artifact evidence."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the stable identity of one row.

    Both components are percent encoded, so the identity of two different
    rows can never collide on the separator.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("the document id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("the document id must not contain leading or trailing whitespace")  # noqa: TRY003
    identity = (
        f"{CSV_ROW_SOURCE_ID_PREFIX}{CSV_ROW_SOURCE_ID_SEPARATOR}"
        f"{quote(document_id, safe='')}{CSV_ROW_SOURCE_ID_SEPARATOR}"
        f"{quote(primary_key_value, safe='')}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"the source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into its rows.

    The document is validated as a whole: a malformed file, a malformed
    header or a malformed row fails with a ``ValueError`` and never returns
    a partial result.
    """

    document_id = value.document_id
    primary_key = value.primary_key
    if not isinstance(primary_key, str) or not primary_key.strip():
        raise ValueError("the primary key must be a non-empty string")  # noqa: TRY003
    csv_row_source_id(document_id, "probe")

    text = _decode(value.content)
    records = _read_records(text)
    if not records:
        raise ValueError("the CSV document has no header")  # noqa: TRY003

    columns = _header(records[0])
    if primary_key not in columns:
        raise ValueError(f"the primary key {primary_key!r} is not a column of the document")  # noqa: TRY003
    key_index = columns.index(primary_key)

    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not record:
            # an empty physical record carries no field
            continue
        if len(record) != len(columns):
            raise ValueError("a row does not have the width of the header")  # noqa: TRY003
        key_value = record[key_index]
        if not key_value.strip():
            raise ValueError("the primary key value of a row must not be blank")  # noqa: TRY003
        if key_value in seen:
            raise ValueError(f"the primary key value {key_value!r} is not unique")  # noqa: TRY003
        seen.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _decode(content: bytes) -> str:
    """Decode the document as strict UTF-8 and accept a leading BOM."""

    if not isinstance(content, bytes):
        raise ValueError("the content of a CSV document must be bytes")  # noqa: TRY003, TRY004
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("the CSV document is not valid UTF-8") from error  # noqa: TRY003
    if text.startswith("\ufeff"):
        text = text[1:]
    return text


def _read_records(text: str) -> list[list[str]]:
    """Read the records of the document with the semantics of ``csv.reader``."""

    try:
        return list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"the CSV document is malformed: {error}") from error  # noqa: TRY003


def _header(record: list[str]) -> tuple[str, ...]:
    """Validate the header of the document."""

    if not record:
        raise ValueError("the CSV document has no header")  # noqa: TRY003
    columns = tuple(record)
    for column in columns:
        if not column.strip():
            raise ValueError("a column name of the header must not be blank")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("the column names of the header must be unique")  # noqa: TRY003
    return columns


class CSVRowAdapter:
    """Resolve and read the row level Source of a CSV document."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one captured row without coupling consumers to ``CSVRowSource``."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowAdapter",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
