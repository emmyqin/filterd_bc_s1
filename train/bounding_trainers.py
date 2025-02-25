import trl
import gc
from copy import deepcopy
import torch
import torch.amp as amp
from torch import nn
from trl.models import PreTrainedModelWrapper
from trl.trainer.utils import disable_dropout_in_model
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available, is_torch_xpu_available
import deepspeed

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=True, num_items_in_batch=None, ref_log_probs=None):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        log_probs = - nn.functional.log_softmax(logits, dim=-1)
        
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        if ref_log_probs is not None:
            with torch.no_grad():      
                # ref_nll_loss = ref_log_probs.gather(dim=-1, index=labels).detach()
                # del ref_log_probs
                iw = torch.exp(- ref_log_probs - nll_loss.detach())
                iw.masked_fill_(padding_mask, 1.0)
                # iw = torch.clamp(iw, 1. - 0.8, 1 + 0.8)
                print(f"********THE IW IS*********{iw[:, -30:]}")
                print(f"********THE IW IS*********{torch.sum(iw < 0.9)}")
                print(f"********THE IW IS*********{(iw.mean(), iw.max(), iw.min())}")
                # print(f"What are model outputs {model_output['mean_iw']}")
        else:
            iw = None

        if iw is not None:
            nll_loss *= 1.

        nll_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        if num_items_in_batch:
            nll_loss = nll_loss.sum() / num_items_in_batch
        else:
            nll_loss = nll_loss.mean()
        return nll_loss
    

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    print(f"-----------logits are of shape {logits.shape}---------")
    print(f"-----------labels are of shape {labels.shape}---------")
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    print(f"-----------AFTER PADDINGIN: labels are of shape {labels.shape}---------")
    shift_labels = labels[..., 1:].contiguous()
    print(f"-----------SHIFTED: labels are of shape {shift_labels.shape}---------")

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    print(f"-----------LOGITS ARE FLATTENED: labels are of shape {logits.shape}---------")
    shift_labels = shift_labels.view(-1)
    print(f"-----------SHIFTED LABELS ARE FLATTENED: labels are of shape {shift_labels.shape}---------")
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    print(f"--------------THE LOSS IS {loss} ------------")
    return loss




class BoundingTrainer(trl.SFTTrainer):

    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ref_model = self._wrap_model(ref_model, training=True)
        self.ref_model = ref_model

        if self.ref_model is not None:
            disable_dropout_in_model(self.ref_model)
            
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self._peft_has_been_casted_to_bf16 = False
    
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    
    def training_step(self, model, inputs):
        loss_step = super().training_step(model, inputs)
        torch.cuda.empty_cache()
        gc.collect()
        # get_accelerator().empty_cache()
        return loss_step
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        disable_dropout_in_model(model)
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # print(f"************inputs keys are*********{inputs.keys()}")
        # print(f"*********DIFF INPUT IDS*******{torch.abs(inputs['input_ids'] - inputs['inputs_ref']).max()}")
        # print(f"*********DIFF ATTN MASK*******{torch.abs(inputs['attention_mask'] - inputs['attn_mask']).max()}")
        # torch.clamp(labels, min=0) 
        # print(f"*********DIFF LABELS*******{inputs['labels'][:, -30:]}....{inputs['input_ids'][:, -30:]}")
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            raise ValueError("Got labels in input, expecting autoregressive training!")
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            if num_items_in_batch:
                print(f"THE INPUTS ARE {inputs.keys()}")
                if self.ref_model is None:
                    if not "ref_log_probs" in inputs:
                        raise ValueError("We need ref_log_probs in inputs if no ref_model is provided!")
                    ref_log_probs = inputs["ref_log_probs"]
                    # print("**************SHAPE OF THE REF LOG PROBS************")
                else:
                    # TODO: Currently this if else loop is a hacky way to define train or eval losses.
                    # print("WE ARE HERE, MAYBE IN TRAIN?")
                    # device_type = "xpu" if is_torch_xpu_available() else "cuda"
                    # compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    with torch.no_grad():
                        # ref_inputs = {k: v.to(self.ref_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        ref_outputs = self.ref_model(**inputs)
                        ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[0]
                        del ref_outputs
                        ref_logits = ref_logits.to(model.device)
                        ref_logits = ref_logits.detach()
                        #ref_log_probs = - nn.functional.log_softmax(ref_logits, dim=-1).detach()
                        ref_logits -= ref_logits.max(dim=-1, keepdim=True).values
                        torch.exp(ref_logits, out=ref_logits)
                        ref_logits /= ref_logits.sum(dim=-1, keepdim=True)
                        torch.log(ref_logits, out=ref_logits)
                        targets = inputs['labels'][..., 1:]
                        ref_log_probs = log_probs.gather(dim=-1, index=targets)
                        del ref_logits

                loss = LabelSmoother(epsilon=0.0)(
                    outputs, inputs['labels'], shift_labels=True, 
                    num_items_in_batch=num_items_in_batch, ref_log_probs=ref_log_probs)
                
            else:
                # print("WE ARE HERE, MAYBE IN EVAL?")
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    
        # return loss
