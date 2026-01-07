import pytest

from verifiers.utils.rlm_data_serialization_utils import (
    build_default_data_serializers,
    prepare_context_data,
)


def test_prepare_text_context_data():
    serializers = build_default_data_serializers()
    prepared = prepare_context_data("hello", None, serializers, max_payload_bytes=1024)

    spec = prepared.context_dict["input_data_spec"]
    assert spec is not None
    assert spec["dtype"] == "text"
    assert spec["payload_path"] is not None
    assert prepared.payload_bytes == b"hello"

    metadata = prepared.context_dict["input_data_metadata"]
    assert "str" in metadata["type"]
    assert metadata["size"] == 5


def test_prepare_json_context_data():
    serializers = build_default_data_serializers()
    prepared = prepare_context_data({"a": 1}, None, serializers, max_payload_bytes=1024)

    spec = prepared.context_dict["input_data_spec"]
    assert spec is not None
    assert spec["dtype"] == "json"
    assert spec["payload_path"] is not None

    metadata = prepared.context_dict["input_data_metadata"]
    assert metadata["dtype"] == "json"


def test_prepare_context_data_requires_supported_dtype():
    serializers = build_default_data_serializers()
    with pytest.raises(ValueError):
        prepare_context_data({"a": 1}, "unknown", serializers, max_payload_bytes=1024)


def test_prepare_context_data_rejects_unknown_type():
    serializers = build_default_data_serializers()
    with pytest.raises(ValueError):
        prepare_context_data((1, 2), None, serializers, max_payload_bytes=1024)
