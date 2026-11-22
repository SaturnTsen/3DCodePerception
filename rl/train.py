"""Load the experiment with Hydra and launch verl's training workflow."""
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def tee_output(log_path: Path) -> None:
    """Copy all stdout/stderr of this process, including Ray worker output forwarded to the driver, into log_path."""
    read_fd, write_fd = os.pipe()
    subprocess.Popen(['tee', '-i', str(log_path)], stdin=read_fd)
    os.close(read_fd)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(write_fd, sys.stdout.fileno())
    os.dup2(write_fd, sys.stderr.fileno())
    os.close(write_fd)


@hydra.main(config_path='../configs', config_name='train-3d_code_data_factory-grpo', version_base=None)
def main(config: DictConfig) -> None:
    """Train with verl using configs/train-3d_code_data_factory-grpo.yaml.

    Run: pixi run train-rl
    Override YAML values: pixi run train-rl trainer.total_epochs=2
    """
    tee_output(Path(HydraConfig.get().runtime.output_dir) / 'train.log')

    for split in ('train_files', 'val_files'):
        paths = config.data[split]
        if not paths:
            raise ValueError(f'data.{split} is empty; prepare and configure the dataset first')
        for path in ([paths] if isinstance(paths, str) else paths):
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    if not config.trainer.use_v1:
        raise ValueError('This training entry requires trainer.use_v1=true')

    # The reward service keeps the files of every request of this run under run_id.
    config.reward.custom_reward_function.reward_kwargs.run_id = (
        f'{config.trainer.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}')

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
