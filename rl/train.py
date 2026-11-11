"""Validate task inputs, then delegate training unchanged to verl."""
from pathlib import Path
import sys

import yaml


def main():
    config_path = Path(sys.argv[1]).resolve()
    config = yaml.safe_load(config_path.read_text())
    for split in ('train_files', 'val_files'):
        paths = config['data'][split]
        if not paths:
            raise ValueError(f'data.{split} is empty; prepare and configure the dataset first')
        for path in ([paths] if isinstance(paths, str) else paths):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    from verl.trainer.main_ppo import main as verl_main
    sys.argv = [sys.argv[0], '--config-path', str(config_path.parent), '--config-name', config_path.stem]
    verl_main()


if __name__ == '__main__':
    main()
