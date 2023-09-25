import os
from dotenv import load_dotenv
from huggingface_hub import login, HfApi 
from llama_recipes.finetuning import main as finetuning

# def main():
#     load_dotenv()
#     login(token=os.environ["HUGGINGFACE_TOKEN"])
    
#     kwargs = {
#         "model_name": "meta-llama/Llama-2-7b-hf",
#         "use_peft": True,
#         "peft_method": "lora",
#         "quantization": True,
#         "batch_size_training": 2,
#         "dataset": "custom_dataset",
#         "custom_dataset.file": "./custom_dataset.py",
#         "output_dir": "./output_dir ",
#     }
    
#     finetuning(**kwargs)

#     api = HfApi() 

#     api.upload_folder( 
#         folder_path='./output_dir/', 
#         repo_id=os.environ["HUGGINGFACE_REPO"], 
#         repo_type='model', 
#     )

# if __name__ == "__main__":
#     main()
    
    
# -*- coding: utf-8 -*-
"""Neurips_efficiency_LLM_challenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tm0lUxZ9ilJJL8-tFw2eF7PrAMH5hjGy

# Fine-tuning✨✨ LLaMA using PEFT, QLora and Hugging Face utilities

In this project we'll use 4-bit quantification to fine tune LLama-2 in purpose to produce QA engine
"""

# @title Install dependancies

# @title Import dependancies
from datasets import load_dataset
from random import randrange

import torch as th
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from torch.utils.data import DataLoader, Dataset
import argparse,time

def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="GAIR/lima")
    parser.add_argument("--split", type=str, default="train[:10%]")
    parser.add_argument("--hf_rep", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fine-tuned-model-name", type=str, required=True)
    parser.add_argument('--bf16', action='store_true')
    args= parser.parse_args()
    
    # @title GLobal parameters setting

    """
        HuggingFace configuration
    """
    # import model you want to train from hugging face
    model_id = args.model_name

    # dataset name
    dataset_name = args.dataset

    # dataset_split
    dataset_split = args.split
    # fine tune model name
    new_model = args.fine_tuned_model_name
    # HuggingFace repository
    hf_model_rep = args.hf_rep
    # Load the entire model on the GPU 0
    # device_map = th.device('cuda' if th.cuda.is_available() else 'cpu')
    device_map = {'':0}

    """
    bitsandBytes parameters
    """
    # activation 4-bit precision base model loaded
    use_4bits = True

    # Compute dtype for 4-bit base models
    bnb_4bits_compute_dtype = "float16"
    # quantisation type
    bnb_4bits_quan_type = "nf4" #  we can use nf4 of fp4
    # activation nested quantization for 4-bits base model (double quantization)
    use_double_quant_nested = False

    """
    QloRa parameters
    """
    # LoRa attention dimension
    lora_r = 64
    # alpha parameter for lora scaling
    lora_alpha = 16
    # dropout probality for lora layer
    lora_dropout = 0.1

    """
    TrainingArgument parameters
    """
    # Output directory where the model predictions and checkpoints will be stored
    ouput_dir = new_model
    # number_of_training epochs
    N_EPOCHS = 1
    # Enable fp16/bf16 training
    fp16 = False
    bf16 = True
    # Batch size per GPU for training
    per_device_train_batch_size = 1
    # Number of update steps to accumulate the gradients
    gradient_accumulation_steps = 1
    # Enable gradient checkpointing
    gradient_checkpointing = True
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3
    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4 #1e-5
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001
    # Optimizer to use
    optim = "paged_adamw_32bit"
    # Learning rate schedule
    lr_scheduler_type = "cosine"
    # Number of training steps
    max_steps = -1
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03
    # Group sequences into batches with same length
    group_by_length = False
    # Save checkpoint every X updates steps
    save_steps = 0
    # Log every X updates steps
    logging_steps = 25
    # Disable tqdm
    disable_tqdm= True

    """
    SFTTrainer parameters
    """
    # Maximum sequence length to use
    max_seq_length = 2048
    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = True #False

    device_map

    # @title Login on hugging face
    from huggingface_hub import login,notebook_login
    from dotenv import load_dotenv
    # notebook_login()

    # @title Load environments variables
    # from dotenv import load_dotenv
    import os

    # # Load the enviroment variables
    load_dotenv()
    # Login to the Hugging Face Hub
    login(token=os.getenv("HF_HUB_TOKEN"))

    # @title load dataset with instructions
    train_data = load_dataset(dataset_name, split=dataset_split)
    test_data = load_dataset(dataset_name, split="test")
    
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("Train data size", train_data_size, "\n Test data size: ", test_data_size)
    
    
    # displays an example
    # dataset[randrange(dataset_size)]

    for i in train_data[randrange(train_data_size)]['conversations']:
        print(i,"\n")
        print("size: ", len(train_data[randrange(train_data_size)]['conversations']))

    # Check the dataset structure
    train_data, test_data

    """# fine-tune a Llama 2 model using trl and the SFTTrainer

    to fine tuning our model, we need to convert our structured example of tasks by instructions. we define a formating function that take as inputs a sample and return string with our function format
    """

    # @title fine-tune a Llama 2 model using trl and the SFTTrainer
    def format_instruction(sample):
        question = sample['conversations'][0]
        response = sample['conversations'][1]
        string = f"""
            ### User:
            {question}

            ### Assistant
            {response}

            ### Source
            {sample['source']}

        """
        return string

    # Show a formatted instruction
    print(format_instruction(train_data[randrange(len(train_data))]))

    # @title Using QLoRA technique to reduce memory footprint during the fine-tuning

    # get the type
    compute_dtype = getattr(th, bnb_4bits_compute_dtype)
    print(compute_dtype)

    # BitAndBytesConfg int-4 configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bits,
        bnb_4bit_use_double_quant=use_double_quant_nested,
        bnb_4bit_quant_type= bnb_4bits_quan_type,
        bnb_4bits_compute_dtype=compute_dtype
    )
    # bnb_config.bnb_4bit_use_double_quant

    # @title Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map=device_map)
    model.config.pretraining_tp = 1

    # @title Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.name_or_path

    # @title Lora config based on Qlora paper
    """
    The SFTTrainer supports a native integration with peft, which makes it
    super easy to efficiently instruction tune LLMs.
    We only need to create our LoRAConfig and provide it to the trainer.
    """
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_config

    """Before we can start our training, we need to define the hypersparemeters In a TrainingArgument object we want to use"""

    # @title Define parameters in TrainingArguments
    args = TrainingArguments(
        output_dir=ouput_dir,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        # bf16=bf16,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        # disable_tqdm=disable_tqdm,
        report_to="tensorboard",
        seed=42

    )

    # @title Create a Trainer
    """
    We now have every building block we need to create our SFTTrainer to start then training our model.
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        formatting_func=format_instruction,
        args=args
    )

    # @title Start Training
    """
    Start training our model by calling the train() method on our Trainer instance.
    """
    start_time = time.time()
    print("Start Training", start_time)
    trainer.train()
    print(f"Total training time {(time.time() - start_time) / 60:.2f} min")
    
    # save metrics
    trainer.save_metrics()

    # save_model in local
    trainer.save_model()

    """# Merge the model and adpater and save it

    if running in a T4 instance we have to clean the memory
    """

    # @title empty VRAM
    import gc
    del model
    del trainer
    gc.collect()

    th.cuda.empty_cache()

    gc.collect()

    # @title Reload the trained and saved model and merge it then we can save the whole model
    new_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=th.float16,
        device_map=device_map
    )

    # @title Merge LoRa and Base Model
    merged_model = new_model.merge_and_unload()

    # save the merge model
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    tokenizer.save_pretrained("merged_model")

    # @title Push Merged Model to the Hub
    merged_model.push_to_hub(hf_model_rep)
    tokenizer.push_to_hub(hf_model_rep)

    # Test the Merged Model
    # sample = test_data[randrange(len(test_data))]
    # prompt = f"""### System
    # ### Users:
    # {sample['conversations'][0]}

    # ### Assistant:

    # """
    # input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=1000,
    #                                 do_sample=True, top_p=0.9, temperature=0.5)
    # generated_instruction = tokenizer.batch_decode(outputs.detach().cpu().numpy(),
    #                                             skip_special_tokens=True)[0][len(prompt):]
    # solution = sample['conversations'][1]
    # print("Prompt: \n", prompt, "\n")
    # print("Generated Instruction: \n", generated_instruction, "\n")
    # print("Ground Thruth: \n", solution)

    # @title Execute a new inference

    # @title End Fine-tuning

if __name__ == "__main__":
    
   main1()