"""Run a configured dataset processor and write its splits as Parquet for verl."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rl.data_prep import PROCESSORS, DataConfig

def parse_data_config(path: Path) -> DataConfig:
    """Read YAML and validate it against the Pydantic configuration model."""
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    common = DataConfig.model_validate(raw, extra='ignore')
    if common.data_preparation not in PROCESSORS:
        raise ValueError(f'Unknown data preparation processor {common.data_preparation!r}')
    return PROCESSORS[common.data_preparation].config_model.model_validate(raw)


def prepare(config: DataConfig) -> None:
    """Write the training and validation records partitioned by the processor.

    Processors receive the complete configuration. Each split must produce at
    least one record. Paths are relative to the working directory.
    """
    if config.data_preparation not in PROCESSORS:
        raise ValueError(f'Unknown data preparation processor {config.data_preparation!r}')
    processor_type = PROCESSORS[config.data_preparation]
    destinations: dict[str, Path] = {}
    for split, output in (('train', config.outputs.train), ('val', config.outputs.val)):
        destination = output.resolve()
        if destination in destinations.values():
            raise ValueError(f'Output path for {split!r} is also used by another split')
        destinations[split] = destination

    processor = processor_type(config)
    splits = processor.process()
    for split, records in (('train', splits['train']), ('val', splits['val'])):
        destination = destinations[split]
        rows = list(records)
        if not rows:
            raise ValueError(f'{split} has no samples')
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(parse_data_config(args.config))


if __name__ == '__main__':
    main()
