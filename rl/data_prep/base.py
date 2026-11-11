"""Interface for dataset-specific conversion to verl Parquet records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, TypedDict

from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

# Pydantic models for data preparation configuration:
class OutputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid") # 注意：model_config 不是字段，而是 Pydantic 模型自身的配置
    train: Path
    val: Path


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data_preparation: str
    outputs: OutputsConfig
    system_prompt: str
    user_prompt: str


# Data preparation processor interface and record partitioning
class PreparedSplits(TypedDict):
    """Training and validation records partitioned by the dataset processor."""

    train: Iterable[dict[str, Any]]
    val: Iterable[dict[str, Any]]


class DataPrepProcessor(ABC):
    """Read and validate samples and construct complete records for verl.

    Subclasses partition the data into train and val. The shared prepare entry
    point owns Parquet writing.
    Configuration paths are relative to the working directory.
    """

    config_model: ClassVar[type[DataConfig]] = DataConfig

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt

    @abstractmethod
    def process(self) -> PreparedSplits:
        """Return train and val records containing prompt, images and rewards.

        Subclasses read and validate the source dataset, partition its samples
        and prevent object leakage between train and val. This method is called
        once for the complete dataset.
        """
