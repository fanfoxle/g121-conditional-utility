"""Frozen Qwen/LoRA/optimizer/determinism construction shared by P3 and formal training."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_frozen_model(
    model_path: Path, init_seed: int, *, initial_adapter_path: Path | None = None
):
    """Load the permanent BF16 base and exact FP32 LoRA closure."""

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.to("cuda:0")
    configure_determinism(init_seed)
    if initial_adapter_path is None:
        model = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=list(TARGET_MODULES),
                task_type="CAUSAL_LM",
                init_lora_weights=True,
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            base,
            str(initial_adapter_path),
            is_trainable=True,
            local_files_only=True,
        )
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(parameter.dtype != torch.float32 for _, parameter in trainable):
        raise RuntimeError("PREC1 FP32 adapter closure failed")
    b_parameters = [parameter for name, parameter in trainable if "lora_B" in name]
    if not b_parameters or any(
        torch.count_nonzero(parameter).item() for parameter in b_parameters
    ):
        raise RuntimeError("LI1 zero-B initialization failed")
    return model, tokenizer, trainable


def build_frozen_optimizer(trainable, peak_lr: float):
    if peak_lr not in (2e-4, 1e-4, 5e-5):
        raise ValueError("peak learning rate is outside the frozen recipe ladder")
    return torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
