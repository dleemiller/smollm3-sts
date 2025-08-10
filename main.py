#!/usr/bin/env python
"""
GRPO fine-tuning for similarity rating with SmolLM3-3B.

• Fine-tune model to rate similarity between sentence pairs on 0-10 scale
• Two reward functions: format compliance + similarity accuracy  
• Dataset: Uniformly sampled from wiki-sim and STSB datasets
• Uses Liger + fast rollouts for efficiency
"""

import argparse, random, re
from pathlib import Path
from contextlib import contextmanager

import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Import our custom modules
from rewards import format_compliance_reward_fn, similarity_accuracy_reward_fn
from dataset import create_uniform_sampled_dataset

# ---------------------------- CLI ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # Reasoning mode probability
    ap.add_argument("--think-prob", type=float, default=0.5, help="Probability to enable <think> reasoning per sample [0..1].")
    # Dataset size
    ap.add_argument("--n-wiki-sim", type=int, default=2750, help="Number of wiki-sim samples (250 per bin * 11 bins)")
    ap.add_argument("--n-stsb", type=int, default=2750, help="Number of STSB samples (250 per bin * 11 bins)")
    # Throughput / capacity
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--generation-batch-size", type=int, default=0, help="If 0, auto-snaps to a valid multiple.")
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--max-steps", type=int, default=5000)
    # Generation behavior
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-completion-length", type=int, default=2048)
    # Optimizations
    ap.add_argument("--use-liger-loss", action="store_true", default=False)
    ap.add_argument("--use-liger-kernel", action="store_true", default=False)
    ap.add_argument("--use-flash-attn", action="store_true", default=False)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ---------------------------- Constants ----------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
OUTPUT_DIR = "./smollm3-grpo-sts"
LOG_DIR = f"{OUTPUT_DIR}/logs"

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------- Utils ----------------------------
@contextmanager
def rollout_cache(model):
    """Enable KV cache and temporarily disable grad-ckpt ONLY for generation."""
    was_ckpt = getattr(model, "is_gradient_checkpointing", False)
    try:
        if was_ckpt:
            model.gradient_checkpointing_disable()
        old = getattr(model.config, "use_cache", False)
        model.config.use_cache = True
        torch.set_grad_enabled(False)
        yield
    finally:
        torch.set_grad_enabled(True)
        model.config.use_cache = False
        if was_ckpt:
            model.gradient_checkpointing_enable()

# ---------------------------- Reward Function Wrappers ----------------------------
def format_reward_fn(prompts, completions, bins, **kwargs):
    """Wrapper for format compliance reward function"""
    return format_compliance_reward_fn(prompts, completions, bins, **kwargs)

def accuracy_reward_fn(prompts, completions, bins, **kwargs):
    """Wrapper for similarity accuracy reward function"""
    return similarity_accuracy_reward_fn(prompts, completions, bins, **kwargs)

# ---------------------------- Main ----------------------------
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ----- Create dataset -----
    print("Creating uniformly sampled similarity dataset...")
    df = create_uniform_sampled_dataset(
        n_wiki_sim=args.n_wiki_sim,
        n_stsb=args.n_stsb,
        n_bins=11,
        random_state=args.seed
    )
    
    if df.empty:
        raise ValueError("Failed to create dataset")
    
    print(f"Created dataset with {len(df)} examples")

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ----- Model (optional Liger kernels / Flash-Attn2) -----
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = None
    if args.use_liger_kernel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model = AutoLigerKernelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
            print("✅ Liger kernels: enabled (AutoLigerKernelForCausalLM).")
        except Exception as e:
            print(f"⚠️  Liger kernels not applied ({e}). Falling back to HF model.")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    # ----- Monkey-patch model.generate to use cache during rollout -----
    _orig_generate = model.generate
    def _cached_generate(*a, **k):
        with rollout_cache(model):
            k.setdefault("use_cache", True)
            return _orig_generate(*a, **k)
    model.generate = _cached_generate

    # ----- LoRA -----
    #lora_cfg = LoraConfig(
    #    r=args.lora_r, 
    #    lora_alpha=args.lora_alpha, 
    #    lora_dropout=0.05, 
    #    bias="none",
    #    task_type="CAUSAL_LM", 
    #    target_modules="all-linear",
    #)

    # ----- Dataset formatting -----
    SIMILARITY_INSTRUCTION = """Rate the semantic similarity (STS) between the reference and candidate text on a scale from 0 to 10, where:
- SCORE: 0  # unrelated
- SCORE: 5  # somewhat related
- SCORE: 10  # identical or nearly identical

Respond with your reasoning (if using <think> tags) followed by your final rating in this exact format:
SCORE: X

Where X is an integer from 0 to 10.
Use no extra whitespace or text."""

    def format_example(example):
        enable_think = rng.random() < args.think_prob
        
        user_content = f"""{SIMILARITY_INSTRUCTION}

# REFERENCE
{example['sentence1']}

# CANDIDATE
{example['sentence2']}"""
        
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=enable_think
        )
        
        return {
            "prompt": prompt,
            "bins": example["bin"],
            "scores": example["score"],
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"]
        }

    # Convert pandas DataFrame to HuggingFace Dataset and format
    dataset = Dataset.from_pandas(df)
    train_ds = dataset.map(format_example, remove_columns=dataset.column_names)
    train_ds = train_ds.shuffle(seed=args.seed)

    print(f"Formatted dataset with {len(train_ds)} examples")
    print("Sample prompt:")
    print(train_ds[0]["prompt"])
    print(f"Target bin: {train_ds[0]['bins']}")

    # ----- Batch math -----
    base = args.per_device_train_batch_size * args.gradient_accumulation_steps
    gen_bs = args.generation_batch_size or max(args.num_generations, (base // args.num_generations) * args.num_generations)
    if gen_bs % args.num_generations != 0:
        gen_bs = (gen_bs // args.num_generations) * args.num_generations or args.num_generations
    print(f"[dbg] base={base} num_generations={args.num_generations} -> generation_batch_size={gen_bs}")

    # ----- GRPO config -----
    grpo_cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        report_to=["tensorboard"],
        seed=args.seed,

        # Loss / trust-region
        beta=0.0,
        epsilon=0.2,
        loss_type="dr_grpo",
        scale_rewards=False,
        mask_truncated_completions=True,
        use_liger_loss=args.use_liger_loss,

        # Online generation
        num_generations=args.num_generations,
        generation_batch_size=gen_bs,
        max_completion_length=args.max_completion_length,
        generation_kwargs=dict(
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            cache_implementation="static",
            max_new_tokens=args.max_completion_length,
            use_cache=True,
        ),

        # Efficiency
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        learning_rate=args.learning_rate,
        use_vllm=False,

        # Horizon
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=300,
        save_total_limit=3,
    )

    # ----- Trainer -----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_fn, accuracy_reward_fn],  # Both format compliance and accuracy
        train_dataset=train_ds,
        args=grpo_cfg,
        processing_class=tokenizer,
        #peft_config=lora_cfg,
    )

    # ----- Train → merge LoRA → save -----
    print("Starting training...")
    trainer.train()
    
    print("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    save_path = Path(OUTPUT_DIR) / "merged"
    merged.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Done. Merged model saved to: {save_path}")
    print(f"🔭 TensorBoard: tensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    main()
