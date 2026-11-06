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

"""Interoperable Context Pack and Context Pack Delta wire formats.

A Context Pack carries one complete prepared context as an ordered tuple of
byte chunks. A Context Pack Delta carries only the suffix a receiver still
needs after it kept a prefix of the previous Pack. Both formats are pure data:
decoding and rebuilding never touch the sending process, so a receiver can
reproduce the prepared body from serialized bytes alone.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512
_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = len(_DIGEST_PREFIX) + 64
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """A Context Pack protocol or chunking violation."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context payload with its digest and exact wire size."""

    items: tuple[bytes, ...]
    digest: str

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build one complete Pack from an ordered tuple of byte chunks."""

        normalized = _normalize_items(items)
        return cls(items=normalized, digest=_digest(normalized))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split one text body into deterministic, UTF-8 safe chunks.

        Every chunk holds whole characters and never exceeds ``chunk_bytes``.
        A character that cannot fit the limit has no legal boundary, so the
        body is rejected instead of being split into invalid UTF-8.
        """

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        chunks: list[bytes] = []
        current = bytearray()
        for character in content:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError("a character does not fit the chunk limit")
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current = bytearray()
            current += encoded
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        """Serialize one complete Pack, including its exact byte count."""

        encoded = _encode_items(self.items)
        count = 0
        for _ in range(8):
            payload = {
                "schema": CONTEXT_PACK_SCHEMA,
                "items": encoded,
                "digest": self.digest,
                "byte_count": count,
            }
            data = _canonical(payload)
            if len(data) == count:
                return data
            count = len(data)
        raise ContextPackError("byte_count did not converge")

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode one complete Pack and verify its digest and byte count."""

        payload = _decode_object(data)
        if set(payload) != _PACK_FIELDS:
            raise ContextPackError("a Pack carries exactly schema, items, digest, and byte_count")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported Pack schema")
        items = _decode_items(payload["items"])
        digest = _require_digest(payload["digest"], field="digest")
        if digest != _digest(items):
            raise ContextPackError("Pack digest does not match its items")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ContextPackError("byte_count must be a non-negative integer")
        canonical = _canonical(
            {
                "schema": CONTEXT_PACK_SCHEMA,
                "items": _encode_items(items),
                "digest": digest,
                "byte_count": byte_count,
            }
        )
        if byte_count != len(canonical):
            raise ContextPackError("Pack byte count does not match its wire size")
        return cls(items=items, digest=digest)

    @property
    def byte_count(self) -> int:
        """Return the exact serialized size of this Pack."""

        return len(self.to_bytes())


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One suffix a receiver applies to a retained Pack prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Select the shared chunk prefix and carry the remaining target suffix."""

        limit = min(len(base.items), len(target.items))
        prefix = 0
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    def to_bytes(self) -> bytes:
        """Serialize one Delta; its byte count is derived, never stored."""

        payload = {
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        }
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode one Delta and reject every field outside the protocol."""

        payload = _decode_object(data)
        if set(payload) != _DELTA_FIELDS:
            raise ContextPackError("a Delta carries exactly schema, digests, prefix_items, and appended")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported Delta schema")
        base_digest = _require_digest(payload["base_digest"], field="base_digest")
        target_digest = _require_digest(payload["target_digest"], field="target_digest")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=_decode_items(payload["appended"]),
        )

    @property
    def byte_count(self) -> int:
        """Return the exact serialized size of this Delta."""

        return len(self.to_bytes())


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target Pack from a retained base Pack and one Delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("a Delta applies to a Context Pack")
    if delta.base_digest != base.digest:
        raise ContextPackError("Delta base digest does not match the base Pack")
    prefix_items = delta.prefix_items
    if type(prefix_items) is not int or prefix_items < 0 or prefix_items > len(base.items):
        raise ContextPackError("prefix_items is outside the base Pack")
    target = ContextPack.build(base.items[:prefix_items] + tuple(delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("Delta target digest does not match the rebuilt Pack")
    return target


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(items)}
    return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _normalize_items(items: Any) -> tuple[bytes, ...]:
    if isinstance(items, (str, bytes, bytearray)):
        raise ContextPackError("items must be a sequence of byte chunks")
    try:
        candidates = tuple(items)
    except TypeError as error:
        raise ContextPackError("items must be a sequence of byte chunks") from error
    normalized: list[bytes] = []
    for item in candidates:
        if isinstance(item, bytes):
            normalized.append(item)
        elif isinstance(item, (bytearray, memoryview)):
            normalized.append(bytes(item))
        else:
            raise ContextPackError("items must contain byte chunks")
    return tuple(normalized)


def _decode_items(value: Any) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("items must be a list of Base64 strings")
    decoded: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError("items must be a list of Base64 strings")
        try:
            item = base64.b64decode(entry, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("items must contain valid Base64") from error
        if base64.b64encode(item).decode("ascii") != entry:
            raise ContextPackError("items must contain canonical Base64")
        decoded.append(item)
    return tuple(decoded)


def _require_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH or not value.startswith(_DIGEST_PREFIX):
        raise ContextPackError(f"{field} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in value[len(_DIGEST_PREFIX) :]):
        raise ContextPackError(f"{field} must be a sha256 digest")
    return value


def _decode_object(data: Any) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContextPackError("a wire payload must be bytes")
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("a wire payload must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("a wire payload must be a JSON object")
    return payload


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
