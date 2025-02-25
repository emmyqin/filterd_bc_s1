import os
import torch
from dataclasses import dataclass, field
from datasets import load_from_disk, concatenate_datasets, Dataset

import transformers



@dataclass
class ProcessingConfig:
    source_datasets_dir: str = field(default="s1k_with_ref_collator_test")
    save_path: str = field(default="s1k_with_ref_collator_test.hf")
    

if __name__ == "__main__":

    parser = transformers.HfArgumentParser((ProcessingConfig,))
    config = parser.parse_args_into_dataclasses()[0]

    print(f"Will save to directory: {config.save_path}")
    
    paths = os.listdir(config.source_datasets_dir)
    print(f"Found {len(paths)} datasets in {config.source_datasets_dir}")

    datasets = []
    for path in paths:
        datasets.append(load_from_disk(os.path.join(config.source_datasets_dir, path)))

    res_dataset = concatenate_datasets(datasets)
    example = res_dataset[0]
    print(f"Example keys: {example.keys()}")
    # shapes = {k: v.shape for k, v in example.items()}
    # print(f"Example shapes: {shapes}")
    res_dataset.save_to_disk(config.save_path)

