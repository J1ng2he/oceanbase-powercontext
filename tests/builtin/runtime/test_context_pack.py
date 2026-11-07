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

"""Behavior tests for the Context Pack and Context Pack Delta wire formats."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


def test_the_wire_payload_matches_an_independent_digest() -> None:
    items = (b"alpha\n", "海洋".encode())
    unsigned = {
        "schema": "powercontext.context-pack.v1",
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    pack = ContextPack.build(items)

    payload = json.loads(pack.to_bytes())
    assert payload == {
        **unsigned,
        "digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "byte_count": len(pack.to_bytes()),
    }
    assert pack.to_bytes() == json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert pack.byte_count == len(pack.to_bytes())
    assert ContextPack.from_bytes(pack.to_bytes()) == pack


def test_content_is_chunked_into_valid_utf8_and_rebuilt_unchanged() -> None:
    pack = ContextPack.from_content("alpha 海洋 beta", chunk_bytes=7)

    assert b"".join(pack.items).decode("utf-8") == "alpha 海洋 beta"
    assert all(len(item) <= 7 for item in pack.items)
    assert all(item.decode("utf-8") for item in pack.items)
    with pytest.raises(ContextPackError):
        ContextPack.from_content("海", chunk_bytes=2)


def test_a_delta_rebuilds_a_target_that_changes_at_any_position() -> None:
    base = ContextPack.build((b"a", b"b", b"old"))
    for target_items in ((b"a", b"b", b"old", b"tail"), (b"new", b"b", b"old"), (b"a", b"new", b"old"), ()):
        target = ContextPack.build(target_items)
        delta = ContextPackDelta.between(base, target)
        rebuilt = apply_context_pack_delta(
            ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
        )
        assert rebuilt.to_bytes() == target.to_bytes()
        assert delta.byte_count == len(delta.to_bytes())


def test_a_delta_carries_only_the_documented_fields() -> None:
    base = ContextPack.build((b"alpha",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
    payload = json.loads(delta.to_bytes())

    assert set(payload) == {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
    assert payload["schema"] == "powercontext.context-pack-delta.v1"
    assert payload["base_digest"] == base.digest
    assert "byte_count" not in payload


@pytest.mark.parametrize("field", ["schema", "items", "digest", "byte_count"])
def test_an_incomplete_pack_is_rejected(field: str) -> None:
    pack = ContextPack.build((b"alpha",))
    payload = json.loads(pack.to_bytes())
    incomplete = {key: value for key, value in payload.items() if key != field}
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(incomplete).encode())


@pytest.mark.parametrize("prefix", [True, 1.5, -1])
def test_a_delta_rejects_a_prefix_that_is_not_a_non_negative_integer(prefix: object) -> None:
    base = ContextPack.build((b"alpha",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
    payload = json.loads(delta.to_bytes())
    payload["prefix_items"] = prefix
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def test_a_tampered_pack_or_delta_is_rejected() -> None:
    pack = ContextPack.build((b"alpha",))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(pack.to_bytes().replace(pack.digest.encode(), b"sha256:" + b"0" * 64))

    empty = ContextPack.build(())
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(empty, ContextPackDelta(empty.digest, "sha256:" + "0" * 64, 1, ()))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(empty, ContextPackDelta(empty.digest, "sha256:" + "0" * 64, 0, (b"x",)))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(pack, ContextPackDelta("sha256:" + "0" * 64, pack.digest, 1, ()))
