import torch
from datasets import load_dataset
from dataclasses import dataclass
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import bitsandbytes as bnb
import argparse

from trl import SFTTrainer

tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: str
    dataset_name: str
    val_dataset_name: str
    dataset_text_field: str
    log_with: str
    learning_rate: float
    batch_size: int
    seq_length: int
    gradient_accumulation_steps: int
    load_in_8bit: bool
    load_in_4bit: bool
    use_peft: bool
    trust_remote_code: bool
    output_dir: str
    peft_lora_r: int
    peft_lora_alpha: int
    lora_dropout: float
    logging_steps: int
    use_auth_token: bool
    num_train_epochs: int
    max_steps: int


def find_all_linear_names(script_args, model):
    cls = bnb.nn.Linear4bit if script_args.load_in_4bit else (bnb.nn.Linear8bitLt if script_args.load_in_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main(args):
    # Step 1: Load the model
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=(True if args.load_in_4bit else False),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=args.use_auth_token,
    )

    modules = find_all_linear_names(args, model)

    # Step 2: Load the training and validation dataset
    train_dataset = load_dataset("json", data_files=str(args.dataset_name), split="train")
    val_dataset = load_dataset("json", data_files=str(args.val_dataset_name), split="train")

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Directory to save the output files, including trained models, logs, etc.
        per_device_train_batch_size=args.batch_size,  # Batch size per GPU during training.
        per_device_eval_batch_size=args.eval_batch_size,  # Batch size per GPU during evaluation (validation/testing).
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of gradient accumulation steps to perform.
        learning_rate=args.learning_rate,  # Learning rate for the optimizer.
        logging_steps=args.logging_steps,  # Log training metrics every specified number of steps.
        num_train_epochs=args.num_train_epochs,  # Total number of training epochs.
        max_steps=args.max_steps,  # Total number of training steps to run.
        evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch during training.
        load_best_model_at_end=True,  # Load the model with the best metric (specified by `metric_for_best_model`) at the end of training.
        save_total_limit=1,  # Number of checkpoints to keep. Set to 1 to save only the best model.
        metric_for_best_model="loss",  # Metric used to determine the best model. In this case, it's based on the loss value.
        save_strategy="epoch",  # Save model checkpoints at the end of each epoch during training.
    )

    # Step 4: Define the LoraConfig
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=args.seq_length,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field=args.dataset_text_field,
        peft_config=peft_config,
    )

    print("----- DEVICE MAPS -----")
    print(model.hf_device_map)
    print("----- DEVICE MAPS -----")

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--val_dataset_name", default=None, type=str)
    parser.add_argument("--dataset_text_field", default=None, type=str)
    parser.add_argument("--log_with", default=None, type=str)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--seq_length", default=200, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--load_in_8bit", default=False, action="store_true")
    parser.add_argument("--load_in_4bit", default=True, action="store_true")
    parser.add_argument("--use_peft", default=True, action="store_true")
    parser.add_argument("--trust_remote_code", default=True, action="store_true")
    parser.add_argument("--output_dir", default="./lamma-7b-biowic", type=str)
    parser.add_argument("--peft_lora_r", default=4, type=int)
    parser.add_argument("--peft_lora_alpha", default=4, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--logging_steps", default=1, type=int)
    parser.add_argument("--use_auth_token", default='None', action="store_true")
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)

    args = parser.parse_args()

    main(args)
