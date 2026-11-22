"""verl dataset that writes the response token limit into the prompts.

The prompts written by rl/prepare.py contain RESPONSE_LIMIT. Each sample replaces it
with data.max_response_length when its messages are built, so the limit is set only
in the training YAML and the prepared data does not depend on it.

Selected in the training YAML through data.custom_cls.
"""
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

RESPONSE_LIMIT = '{max_response_length}'


class ResponseLimitDataset(RLHFDataset):
    def _build_messages(self, example: dict[str, Any], key: str) -> list[dict[str, Any]]:
        messages = [dict(message) for message in example[key]]
        if not any(RESPONSE_LIMIT in message['content'] for message in messages):
            raise ValueError(f'the prompt has no {RESPONSE_LIMIT} placeholder; prepare the data again')
        limit = str(self.config.max_response_length)
        example[key] = [
            message | {'content': message['content'].replace(RESPONSE_LIMIT, limit)} for message in messages]
        return super()._build_messages(example, key)
