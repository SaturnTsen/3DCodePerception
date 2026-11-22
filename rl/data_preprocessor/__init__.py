"""Shared data preparation configuration models and processor interface."""

from rl.data_preprocessor.base import DataConfig, DataPrepProcessor, OutputsConfig
from rl.data_preprocessor.threed_code_data_factory import ThreeDCodeDataFactoryProcessor

PROCESSORS: dict[str, type[DataPrepProcessor]] = {
    '3DCodeDataFactoryPreprocessor': ThreeDCodeDataFactoryProcessor,
}
