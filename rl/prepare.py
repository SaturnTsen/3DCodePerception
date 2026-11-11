"""Convert explicitly split JSONL manifests to verl's native multimodal Parquet."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml
from rl.geometry import sample_surface


def prepare(config):
    seen = set()
    for split in ('train', 'val'):
        source = config['manifests'][split]
        if source is None:
            raise ValueError(f'manifests.{split} must be filled before preparing data')
        source = Path(source).resolve()
        rows, objects = [], set()
        for index, line in enumerate(source.read_text().splitlines()):
            item = json.loads(line)
            objects.add(item['object_id'])
            images = [(source.parent / name).resolve() for name in item['images']]
            if not images:
                raise ValueError('VLM samples require images')
            for image in images:
                with Image.open(image) as opened:
                    opened.verify()
            target = (source.parent / item['target']).resolve()
            with np.load(target, allow_pickle=False) as mesh:
                sample_surface(mesh['vertices'], mesh['faces'], 1, 0)
            prompt = '\n'.join(['<image>'] * len(images)) + '\n' + item['instruction']
            rows.append(dict(data_source='3d_mesh', prompt=[dict(role='system', content=config['system_prompt']), dict(role='user', content=prompt)], images=[dict(bytes=p.read_bytes(), path=None) for p in images], reward_model=dict(style='rule', ground_truth=str(target)), extra_info=dict(index=index, object_id=item['object_id'], split=split)))
        if not rows or seen & objects:
            raise ValueError('empty split or object leakage between train and validation')
        seen.update(objects)
        destination = Path(config['outputs'][split])
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(destination, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()))
