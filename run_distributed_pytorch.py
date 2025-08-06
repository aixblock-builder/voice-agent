import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (  # BitsAndBytesConfig,AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

from logging_class import start_queue, stop_log, write_log

# from diffusers import DiffusionPipeline


def run_inference(args):
    if args.predict_args:
        with open(args.predict_args, "r") as f:
            predict_args = json.load(f)
    else:
        predict_args = {}
    if "prompt" in predict_args:
        prompt = int(predict_args["prompt"])

    if "text" in predict_args:
        text = int(predict_args["text"])
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["RANK"] = args.rank
    os.environ["WORLD_SIZE"] = args.world_size
    dist.init_process_group(
        args.backend,
        init_method="tcp://{master_addr}:{master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    quantized_model_dir = args.hf_model_id

    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        quantized_model_dir, trust_remote_code=True
    )
    # load_model(model, os.path.join("", f"model{args.rank}-mp{args.world_size}.safetensors"))
    # Prepare input tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Distributed run
    outputs = model(inputs)
    print(outputs)


def run_predict(args):
    run_inference(args)
    mp.spawn(
        run_inference,
        args=(args),
        nprocs=args.world_size * torch.cuda.device_count(),
        join=True,
    )


class TrainOnStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        # Log training loss at step 0
        logs = logs or {}
        self.log(logs)

    def log(self, logs):
        print(f"Logging at start: {logs}")


def is_valid_type(value, expected_type):
    from typing import Union, get_args, get_origin

    # Nếu không có type hint (Empty), chấp nhận giá trị
    if expected_type is inspect._empty:
        return True
    # Nếu type hint là generic (Union, Optional, List, etc.)
    origin = get_origin(expected_type)
    if origin is Union:  # Xử lý Union hoặc Optional
        return any(is_valid_type(value, arg) for arg in get_args(expected_type))
    if origin is list:  # Xử lý List
        return isinstance(value, list) and all(
            is_valid_type(v, get_args(expected_type)[0]) for v in value
        )
    if origin is dict:  # Xử lý Dict
        key_type, value_type = get_args(expected_type)
        return (
            isinstance(value, dict)
            and all(is_valid_type(k, key_type) for k in value.keys())
            and all(is_valid_type(v, value_type) for v in value.values())
        )
    # Kiểm tra kiểu cơ bản (int, float, str, etc.)
    return isinstance(value, expected_type)


def parse_args():
    # Tạo parser cho dòng lệnh
    parser = argparse.ArgumentParser(description="AIxBlock")
    # infrastructure args
    parser.add_argument("--world_size", type=int, default=1, help="world_size")
    parser.add_argument(
        "--master_addr", type=str, default="0.0.0.0", help="master_addr"
    )
    parser.add_argument("--master_port", type=str, default="23456", help="master_port")
    parser.add_argument("--rank", type=str, default=0, help="rank")
    parser.add_argument(
        "--backend", type=str, default="nccl", help="JSON string for backend"
    )
    # model args
    parser.add_argument("--hf_model_id", type=str, default=None, help="hf_model_id")
    parser.add_argument("--dataset_local", type=str, default=None, help="dataset id")
    parser.add_argument(
        "--training_args_json",
        type=str,
        default=None,
        help="JSON string for training arguments",
    )
    parser.add_argument(
        "--predict_args", type=str, default="predict_args.json", help="predict_args"
    )
    parser.add_argument("--hf_token", type=str, default=None, help="hf_token")
    parser.add_argument("--push_to_hub", type=str, default=None, help="push_to_hub")
    parser.add_argument(
        "--push_to_hub_token", type=str, default=None, help="push_to_hub_token"
    )
    # custom args
    parser.add_argument(
        "--action", type=str, default=None, help="/action, command field"
    )
    parser.add_argument("--channel_log", type=str, default=None, help="channel_log")

    # Phân tích các tham số dòng lệnh
    args = parser.parse_args()

    if args.channel_log:
        log_queue, logging_thread = start_queue(args.channel_log)
        write_log(log_queue)

    return args


def run_train(args):
    dataset_local = args.dataset_local
    is_use_local = False
    num_train_epochs = 1
    per_train_dataset = 0.8
    per_test_dataset = 0.2
    output_dir = "./data/checkpoint"

    push_to_hub = True if args.push_to_hub and args.push_to_hub == "True" else False
    hf_model_id = args.hf_model_id if args.hf_model_id else "llama4"
    push_to_hub_token = (
        args.push_to_hub_token
        if args.push_to_hub_token
        else "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"
    )

    training_args_dict = {}
    # Nếu có file JSON, đọc và phân tích nó
    print(args.training_args_json)
    if args.training_args_json:
        with open(args.training_args_json, "r") as f:
            training_args_dict = json.load(f)

    wandb.login("allow", "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")

    # Hugging Face model id
    model_id = "facebook/opt-125m"  # default
    if "model_id" in training_args_dict and training_args_dict["model_id"] != "None":
        model_id = training_args_dict["model_id"]
        # "facebook/opt-125m" # or  `appvoid/llama-3-1b` tiiuae/falcon-7b` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`

    dataset_id = "Sujithanumala/Llama_3.2_1B_IT_dataset"
    if "dataset_id" in training_args_dict:
        dataset_id = training_args_dict[
            "dataset_id"
        ]  # "Sujithanumala/Llama_3.2_1B_IT_dataset"
    else:
        dataset_id = "Sujithanumala/Llama_3.2_1B_IT_dataset"

        if dataset_local and dataset_local != "None":
            dataset_id = dataset_local
            is_use_local = True

    if "num_train_epochs" in training_args_dict:
        num_train_epochs = int(training_args_dict["num_train_epochs"])

    if "per_train_dataset" in training_args_dict:
        per_train_dataset = int(training_args_dict["per_train_dataset"])

    if "per_test_dataset" in training_args_dict:
        per_test_dataset = int(training_args_dict["per_test_dataset"])

    if not is_use_local:
        dataset = load_dataset(dataset_id, split="train")
        # eval_dataset = train_dataset
        train_test_split = dataset.train_test_split(
            test_size=per_test_dataset, seed=42
        )  # 20% cho eval
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:
        # Load dataset từ thư mục local
        dataset = load_dataset(
            "json",
            data_files={
                "train": f"{dataset_id}/train_data.json",
                "test": f"{dataset_id}/test_data.json",
            },
        )

        # Truy cập từng tập dữ liệu
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )

    tokenized_datasets = train_dataset.map(
        preprocess_function,
        batched=True,
        # remove_columns=train_dataset['train'].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)

    print("Data is formatted and ready!")
    print("===========")

    training_args = TrainingArguments(
        output_dir=output_dir,  # directory to save and repository id
        # logging_dir= '/app/data/logs',
        learning_rate=2e-4,
        log_level="debug",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        # report_to="tensorboard",
        # report_to="wandb",
        use_cpu=False,
        bf16=False,
        fp16=False,
        push_to_hub=push_to_hub,
        push_to_hub_model_id=hf_model_id if push_to_hub else None,
        push_to_hub_token=push_to_hub_token if push_to_hub else None,
    )
    print(training_args)

    trainer = SFTTrainer(
        dataset_text_field="text",
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # eval_dataset=eval_tokenized_datasets,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        # compute_metrics=compute_metrics,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
            "skip_prepare_dataset": True,  # skip the dataset preparation
        },
        callbacks=[TrainOnStartCallback()],
    )
    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()


def main(args):
    if args.action == "predict":
        run_predict(args)
    elif args.action == "train":
        run_train(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
