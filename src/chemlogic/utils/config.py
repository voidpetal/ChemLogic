"""Pydantic config models for Pipeline construction and training."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

_VALID_TASKS = {"regression", "classification", "multi_class", "multi_regression"}

# Fields that belong in the checkpoint architecture dict.
_ARCH_FIELDS = {
    "dataset_name",
    "model_name",
    "param_size",
    "layers",
    "max_depth",
    "max_subgraph_depth",
    "max_cycle_size",
    "subgraphs",
    "chem_rules",
    "architecture",
    "funnel",
    "task",
    "num_outputs",
}


class PipelineConfig(BaseModel):
    """Validated architecture parameters — serialises directly to checkpoint."""

    model_config = {"arbitrary_types_allowed": True}

    dataset_name: str
    model_name: str
    param_size: Annotated[int, Field(ge=1)]
    layers: Annotated[int, Field(ge=1)]
    max_depth: Annotated[int, Field(ge=1)] = 1
    max_subgraph_depth: Annotated[int, Field(ge=0)] = 5
    max_cycle_size: Annotated[int, Field(ge=0)] = 10
    subgraphs: Any = None
    chem_rules: Any = None
    architecture: str = "BARE"
    funnel: bool = False
    task: str = "classification"
    num_outputs: Annotated[int, Field(ge=1)] = 1

    @field_validator("task")
    @classmethod
    def task_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got {v!r}")
        return v

    def to_architecture_dict(self) -> dict:
        d = self.model_dump(include=_ARCH_FIELDS)
        # tuples aren't JSON-serialisable — normalise to list
        for key in ("subgraphs", "chem_rules"):
            if isinstance(d[key], tuple):
                d[key] = list(d[key])
        return d


class TrainConfig(BaseModel):
    lr: Annotated[float, Field(gt=0.0)] = 0.001
    epochs: Annotated[int, Field(ge=1)] = 100
    split_ratio: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.75
    batches: Annotated[int, Field(ge=1)] = 1
    early_stopping_threshold: Annotated[float, Field(ge=0.0)] = 0.001
    early_stopping_rounds: Annotated[int, Field(ge=1)] = 10
    checkpoint_every: Annotated[int, Field(ge=1)] | None = None
