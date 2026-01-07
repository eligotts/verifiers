from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SerializedData:
    dtype: str
    inline_data: Any | None
    file_bytes: bytes | None
    file_name: str | None
    metadata: dict[str, Any]
    format: str | None = None
    encoding: str | None = None
    deserializer_code: str | None = None
    deserializer_function: str | None = None


@dataclass(frozen=True)
class DataSerializer:
    dtype: str
    serialize: Callable[[Any], SerializedData]
    can_handle: Callable[[Any], bool] | None = None
    deserializer_code: str | None = None
    deserializer_function: str | None = None


@dataclass(frozen=True)
class PreparedContextData:
    context_dict: dict[str, Any]
    payload_bytes: bytes | None
    payload_path: str | None
    payload_name: str | None


def build_default_data_serializers() -> list[DataSerializer]:
    return [
        DataSerializer(
            dtype="text",
            serialize=serialize_text_data,
            can_handle=lambda value: isinstance(value, str),
        ),
        DataSerializer(
            dtype="json",
            serialize=serialize_json_data,
            can_handle=lambda value: isinstance(value, (dict, list)),
        ),
    ]


def resolve_data_serializer(
    data: Any, dtype: str | None, serializers: list[DataSerializer]
) -> DataSerializer:
    if dtype:
        for serializer in serializers:
            if serializer.dtype == dtype:
                return serializer
        supported = ", ".join(sorted({s.dtype for s in serializers}))
        raise ValueError(
            f"Unsupported dtype '{dtype}' for data type {type(data)}. "
            f"Supported dtypes: {supported}. Provide a custom serializer or use a supported dtype."
        )

    matches: list[DataSerializer] = []
    for serializer in serializers:
        if serializer.can_handle and serializer.can_handle(data):
            matches.append(serializer)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matched = ", ".join(sorted({s.dtype for s in matches}))
        raise ValueError(
            f"Ambiguous data type {type(data)} matched multiple serializers: {matched}. "
            "Specify dtype or provide a custom serializer."
        )

    raise ValueError(
        f"Unsupported data type {type(data)}. Specify dtype or provide a custom serializer."
    )


def prepare_context_data(
    data: Any,
    dtype: str | None,
    serializers: list[DataSerializer],
    max_payload_bytes: int | None,
) -> PreparedContextData:
    if data is None:
        metadata = build_base_metadata(data)
        context_dict = {
            "input_data_spec": None,
            "input_data_metadata": metadata,
        }
        return PreparedContextData(context_dict, None, None, None)

    serializer = resolve_data_serializer(data, dtype, serializers)
    serialized = serializer.serialize(data)

    if serialized.inline_data is not None:
        raise ValueError(
            "Inline payloads are not supported. Provide file bytes via the serializer."
        )
    if serialized.file_bytes is None:
        raise ValueError("Serialized data must include file bytes.")

    deserializer_code = serialized.deserializer_code or serializer.deserializer_code
    deserializer_function = (
        serialized.deserializer_function or serializer.deserializer_function
    )

    payload_bytes = None
    payload_path = None
    payload_name = None
    payload_size = None
    payload_hash = None

    payload_bytes = serialized.file_bytes
    payload_name = validate_file_name(
        serialized.file_name or default_payload_file_name(serialized)
    )
    payload_path = f"/tmp/{payload_name}"
    payload_size = len(payload_bytes)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    ensure_payload_size(payload_size, max_payload_bytes)

    metadata = build_metadata(
        data,
        serialized,
        payload_path=payload_path,
        payload_size=payload_size,
        payload_hash=payload_hash,
    )

    spec = {
        "dtype": serialized.dtype,
        "format": serialized.format,
        "payload_path": payload_path,
        "payload_encoding": serialized.encoding,
        "deserializer_code": deserializer_code,
        "deserializer_function": deserializer_function,
        "metadata": metadata,
    }

    context_dict = {
        "input_data_spec": spec,
        "input_data_metadata": metadata,
    }

    return PreparedContextData(context_dict, payload_bytes, payload_path, payload_name)


def serialize_text_data(data: Any) -> SerializedData:
    if not isinstance(data, str):
        raise ValueError("Text serializer expects a string input.")

    metadata = build_base_metadata(data)
    payload_bytes = data.encode("utf-8")
    return SerializedData(
        dtype="text",
        inline_data=None,
        file_bytes=payload_bytes,
        file_name="rlm_input_data.txt",
        metadata=metadata,
        format="text",
        encoding="utf-8",
    )


def serialize_json_data(data: Any) -> SerializedData:
    validate_json_value(data)
    metadata = build_base_metadata(data)
    payload_text = json.dumps(data, ensure_ascii=True)
    payload_bytes = payload_text.encode("utf-8")
    return SerializedData(
        dtype="json",
        inline_data=None,
        file_bytes=payload_bytes,
        file_name="rlm_input_data.json",
        metadata=metadata,
        format="json",
        encoding="utf-8",
    )


def validate_json_value(
    value: Any, path: str = "root", seen: set[int] | None = None
) -> None:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            raise ValueError(f"Cycle detected in JSON data at {path}.")
        seen.add(obj_id)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"JSON object keys must be strings. Invalid key at {path}."
                )
            validate_json_value(item, f"{path}.{key}", seen)
        seen.remove(obj_id)
        return

    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in seen:
            raise ValueError(f"Cycle detected in JSON data at {path}.")
        seen.add(obj_id)
        for index, item in enumerate(value):
            validate_json_value(item, f"{path}[{index}]", seen)
        seen.remove(obj_id)
        return

    raise ValueError(
        f"Unsupported type for JSON serialization at {path}: {type(value)}"
    )


def build_base_metadata(data: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": str(type(data))}
    if data is None:
        metadata["size"] = 0
        return metadata
    size_value = get_length_if_available(data)
    if size_value is not None:
        metadata["size"] = size_value
    return metadata


def build_metadata(
    data: Any,
    serialized: SerializedData,
    payload_path: str | None,
    payload_size: int | None,
    payload_hash: str | None,
) -> dict[str, Any]:
    metadata = dict(serialized.metadata)
    metadata.setdefault("type", str(type(data)))

    size_value = get_length_if_available(data)
    if size_value is not None and "size" not in metadata:
        metadata["size"] = size_value

    metadata["dtype"] = serialized.dtype
    if serialized.format:
        metadata["format"] = serialized.format

    if payload_path:
        metadata["path"] = payload_path
    if payload_size is not None:
        metadata["file_size"] = payload_size
    if payload_hash:
        metadata["hash"] = payload_hash

    return normalize_metadata(metadata)


def get_length_if_available(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (int, str)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def default_payload_file_name(serialized: SerializedData) -> str:
    if serialized.format == "json":
        ext = "json"
    elif serialized.format == "text" or serialized.dtype == "text":
        ext = "txt"
    else:
        ext = "bin"
    return f"rlm_input_data.{ext}"


def validate_file_name(file_name: str) -> str:
    if "/" in file_name or "\\" in file_name:
        raise ValueError("File name must not include path separators.")
    return file_name


def ensure_payload_size(payload_size: int, max_payload_bytes: int | None) -> None:
    if max_payload_bytes is None:
        return
    if payload_size > max_payload_bytes:
        raise ValueError(
            "Payload exceeds sandbox storage limit: "
            f"{payload_size} bytes > {max_payload_bytes} bytes."
        )
