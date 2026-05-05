import torch
from modelscope import AutoTokenizer,AutoModelForCausalLM
from trl import SFTConfig,SFTTrainer
from transformers import AutoConfig

from utils import get_gsm8k_dataset

from swanlab.integration.huggingface import SwanLabCallback


def train(args):
    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        log_on_each_node=False,
        report_to="swanlab"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map=None,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    ).to("cuda")

    swanlab_callback = SwanLabCallback(
    project="Qwen3_0.5b_r1",
    experiment_name="Qwen3_0.5b_r1",
    description="使用通义千问Qwen3_0.5b_r1模型在gsm8k数据集上微调，实现思维链。",
    config={
        "model": args.model_name_or_path,
        "model_dir": args.cache_dir,
        "dataset": "gsm8k",},)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=get_gsm8k_dataset(sft=True, cache_dir=args.cache_dir,
                                        first_half=args.split_half=="first_half",
                                        second_half=args.split_half=="second_half"),
    )
    trainer.train()
