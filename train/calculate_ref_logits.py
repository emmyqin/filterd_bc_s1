import os
import torch
from dataclasses import dataclass, field
from tqdm import tqdm
from torch import nn
from datasets import load_dataset, Dataset
import trl
from trl.trainer.utils import disable_dropout_in_model

import transformers



@dataclass
class ProcessingConfig:
    source_dataset: str = field(default="simplescaling/s1K-1.1_tokenized")
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    target_path: str = field(default="s1k_with_ref_collator_logits")
    num_splits: int = 8
    split: int = 0

def process_example(model, tokenizer, collator, example):
    torch.cuda.empty_cache()
    curr_prompt = example['text']
    inputs = tokenizer(curr_prompt, return_tensors="pt")
    # print(f"**********INPUTS IS OF SHAPE**********{inputs.input_ids.shape}")
    # for k, v in inputs.items():
    #     print(f"{k}.....{v.shape}")
    inputs = {k: v[0, :] for k, v in inputs.items()}
    inputs = collator([inputs,])
    input_ids, mask = inputs.input_ids.to('cuda'), inputs.attention_mask.to('cuda')
    # print(f"Input ids{input_ids.shape}.......mask {mask.shape}")
    outputs = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=False)
    # print(f".....{input_ids.shape}.....")
    labels = input_ids[..., 1:]
    labels = torch.clamp(labels, min=0)  
    ref_log_probs = nn.functional.log_softmax(outputs['logits'][..., :-1, :], dim=-1)
    if labels.dim() == ref_log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)
    example['ref_log_probs'] = ref_log_probs.gather(dim=-1, index=labels)[0, :, 0].to('cpu')
    example['ref_logits'] = outputs['logits'].to('cpu')
 
    # print(f"example .... {example['ref_log_probs'].shape}")

    torch.cuda.empty_cache()
    return example
    

if __name__ == "__main__":

    parser = transformers.HfArgumentParser((ProcessingConfig,))
    config = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset(config.source_dataset)['train']
    
    res_dataset = []
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name).requires_grad_(False)
    model = model.to('cuda')

    disable_dropout_in_model(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    tokenizer.pad_token = "<|fim_pad|>"
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    # dataset = collator(dataset)
    save_path = f"{config.target_path}/split_{config.split}.hf"
    print(f"Processing dataset split {config.split}/{config.num_splits}")
    print(f"Will save to directory: {save_path}")
    print(f"Available devices: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        raise ValueError("Error: you have multiple GPUs but this script only uses one restrict CUDA_VISIBLE_DEVICES")
  
    for idx in tqdm(range(config.split, len(dataset), config.num_splits), f"Processing dataset split {config.split}"):
        example = dataset[idx]
        result = process_example(model, tokenizer, collator, example)
        if result is not None:
            res_dataset.append(result)
    new_dataset = Dataset.from_list(res_dataset)
    if not os.path.exists(config.target_path):
        os.mkdir(config.target_path)
    new_dataset.save_to_disk(save_path)
