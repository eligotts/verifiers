import pytest

from verifiers.utils.rlm_data_serialization_utils import (
    DataSerializer,
    SerializedData,
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
    assert "hash" not in metadata


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
    with pytest.raises(ValueError, match="Unsupported dtype.*dict"):
        prepare_context_data({"a": 1}, "unknown", serializers, max_payload_bytes=1024)


def test_prepare_context_data_rejects_unknown_type():
    serializers = build_default_data_serializers()
    with pytest.raises(ValueError, match="Unsupported data type.*tuple"):
        prepare_context_data((1, 2), None, serializers, max_payload_bytes=1024)


def test_prepare_file_payload_with_deserializer():
    serializer = DataSerializer(
        dtype="file",
        serialize=lambda data: SerializedData(
            dtype="file",
            inline_data=None,
            file_bytes=b"payload",
            file_name="payload.bin",
            metadata={"type": "file"},
            deserializer_code="def decode(payload, spec):\n    return payload\n",
            deserializer_function="decode",
        ),
    )
    prepared = prepare_context_data(
        object(), "file", [serializer], max_payload_bytes=1024
    )

    spec = prepared.context_dict["input_data_spec"]
    assert spec["payload_path"] is not None
    assert spec["deserializer_code"] is not None
    assert spec["deserializer_function"] == "decode"


def test_inline_payload_rejected():
    serializer = DataSerializer(
        dtype="inline",
        serialize=lambda data: SerializedData(
            dtype="inline",
            inline_data={"value": "nope"},
            file_bytes=None,
            file_name=None,
            metadata={"type": "inline"},
        ),
    )
    with pytest.raises(ValueError, match="Inline payloads are not supported"):
        prepare_context_data(object(), "inline", [serializer], max_payload_bytes=1024)


def test_prepare_context_data_requires_deserializer_for_custom_dtype():
    serializer = DataSerializer(
        dtype="binary",
        serialize=lambda data: SerializedData(
            dtype="binary",
            inline_data=None,
            file_bytes=b"payload",
            file_name="payload.bin",
            metadata={"type": "binary"},
        ),
    )
    with pytest.raises(ValueError, match="requires a deserializer"):
        prepare_context_data(object(), "binary", [serializer], max_payload_bytes=1024)


def test_payload_size_enforced():
    serializers = build_default_data_serializers()
    with pytest.raises(ValueError, match="Payload exceeds sandbox storage limit"):
        prepare_context_data("hello", None, serializers, max_payload_bytes=1)


def test_prepare_context_data_ambiguous_match_requires_dtype():
    serializer_a = DataSerializer(
        dtype="a",
        serialize=lambda data: SerializedData(
            dtype="a",
            inline_data={"value": "a"},
            file_bytes=None,
            file_name=None,
            metadata={"type": "a"},
        ),
        can_handle=lambda data: True,
    )
    serializer_b = DataSerializer(
        dtype="b",
        serialize=lambda data: SerializedData(
            dtype="b",
            inline_data={"value": "b"},
            file_bytes=None,
            file_name=None,
            metadata={"type": "b"},
        ),
        can_handle=lambda data: True,
    )
    with pytest.raises(ValueError, match="Ambiguous data type"):
        prepare_context_data(object(), None, [serializer_a, serializer_b], None)
