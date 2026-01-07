import io
import json
from typing import Literal

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State
from verifiers.utils.rlm_data_serialization_utils import DataSerializer, SerializedData

ContextDType = Literal["text", "json", "tuple", "polars"]

_TUPLE_DESERIALIZER = """
import json

def deserialize_tuple(payload, spec):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    return tuple(data)
"""

_POLARS_DESERIALIZER = """
import io
import polars as pl

def deserialize_polars(payload, spec):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(payload, bytes):
        buffer = io.BytesIO(payload)
        return pl.read_parquet(buffer)
    return payload
"""


def build_tuple_serializer() -> DataSerializer:
    def serialize_tuple(data):
        if isinstance(data, list):
            data = tuple(data)
        if not isinstance(data, tuple):
            raise ValueError("Tuple serializer expects a tuple or list input.")
        payload = json.dumps(list(data)).encode("utf-8")
        return SerializedData(
            dtype="tuple",
            inline_data=None,
            file_bytes=payload,
            file_name="tuple.json",
            metadata={},
            format="json",
            encoding="utf-8",
            deserializer_code=_TUPLE_DESERIALIZER,
            deserializer_function="deserialize_tuple",
        )

    return DataSerializer(
        dtype="tuple",
        serialize=serialize_tuple,
        can_handle=lambda value: isinstance(value, (tuple, list)),
    )


def build_polars_serializer() -> DataSerializer:
    import polars as pl

    def serialize_polars(data):
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Polars serializer expects a polars.DataFrame input.")
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        return SerializedData(
            dtype="polars",
            inline_data=None,
            file_bytes=buffer.getvalue(),
            file_name="dataframe.parquet",
            metadata={},
            format="parquet",
            deserializer_code=_POLARS_DESERIALIZER,
            deserializer_function="deserialize_polars",
        )

    return DataSerializer(
        dtype="polars",
        serialize=serialize_polars,
        can_handle=lambda value: isinstance(value, pl.DataFrame),
    )


def build_dataframe(num_rows: int = 200):
    import polars as pl

    return pl.DataFrame(
        {
            "a": list(range(num_rows)),
            "b": [i * 2 for i in range(num_rows)],
        }
    )


class ArbitraryDataRLMEnv(RLMEnv):
    def __init__(self, context_data, **kwargs):
        super().__init__(**kwargs)
        self._context_data = context_data

    async def setup_state(self, state: State, **kwargs) -> State:
        info = state.get("info") or {}
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info[self.context_key] = self._context_data
        state["info"] = info
        return await super().setup_state(state, **kwargs)


def _build_dataset(question: str, answer: str) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": question,
                "answer": answer,
                "info": {},
            }
        ]
    )


def load_environment(context_dtype: ContextDType = "text", **kwargs) -> vf.Environment:
    serializers: list[DataSerializer] = []

    if context_dtype == "text":
        context_data = "Numbers: 1, 2, 3, 4"
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = "10"
        pip_install_packages = ""
    elif context_dtype == "json":
        context_data = {"values": [1, 2, 3, 4]}
        question = "Sum extra_data['values'] and reply with the integer."
        answer = "10"
        pip_install_packages = ""
    elif context_dtype == "tuple":
        context_data = (1, 2, 3, 4)
        serializers.append(build_tuple_serializer())
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = "10"
        pip_install_packages = ""
    elif context_dtype == "polars":
        dataframe = build_dataframe()
        context_data = dataframe
        serializers.append(build_polars_serializer())
        total = int((dataframe["a"] + dataframe["b"]).sum())
        question = "Compute the sum of columns a and b in extra_data and reply with the integer."
        answer = str(total)
        pip_install_packages = "polars>=0.20.0"
    else:
        raise ValueError(f"Unsupported context_dtype: {context_dtype}")

    dataset = _build_dataset(question, answer)

    async def exact_answer(state, answer, **_kwargs) -> float:
        final_answer = (state.get("final_answer") or "").strip()
        return 1.0 if final_answer == answer else 0.0

    rubric = vf.Rubric(funcs=[exact_answer])

    env = ArbitraryDataRLMEnv(
        dataset=dataset,
        rubric=rubric,
        context_dtype=context_dtype,
        data_serializers=serializers,
        context_data=context_data,
        pip_install_packages=pip_install_packages,
        **kwargs,
    )

    return env
