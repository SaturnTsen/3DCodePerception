"""Load the experiment with Hydra and launch verl's training workflow."""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@hydra.main(config_path='../configs', config_name='grpo', version_base=None)
def main(config: DictConfig) -> None:
    """Train with verl using configs/grpo.yaml.

    Run: pixi run train-rl
    Override YAML values: pixi run train-rl trainer.total_epochs=2
    """
    for split in ('train_files', 'val_files'):
        paths = config.data[split]
        if not paths:
            raise ValueError(f'data.{split} is empty; prepare and configure the dataset first')
        for path in ([paths] if isinstance(paths, str) else paths):
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    if not config.trainer.use_v1:
        raise ValueError('This training entry requires trainer.use_v1=true')
    
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(
        config,
        task_runner_class=TaskRunnerV1
    )


if __name__ == '__main__':
    main()
