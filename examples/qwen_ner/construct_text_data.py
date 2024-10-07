import json
import argparse

import pandas as pd
from datasets import load_dataset

from qwen2ner.default import DEFAULT_SYSTEM_PROMPT

TEMPLATE = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n'

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    ds = load_dataset(path='json', data_files={args.split: f'dataset/{args.split}.json'}, split=args.split)

    results = {'text': []}

    for example in ds:
        full_text = example['full_text']
        tokens = example['tokens']
        labels = example['labels']

        outputs = "\n".join([json.dumps({'token': token, 'label': label}, ensure_ascii=False) for token, label in zip(tokens, labels) if label != 'O'])
        results['text'].append(TEMPLATE.format(DEFAULT_SYSTEM_PROMPT, full_text, outputs))

    df = pd.DataFrame(results)
    df.to_csv(f'dataset/{args.split}.csv', index=False)