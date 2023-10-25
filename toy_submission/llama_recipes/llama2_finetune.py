import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer
import torch as th

import bitsandbytes as bnb
from dataset.custom_dataset import TsotsaDataset
from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )

def loginHub():
    # @title Login on hugging face
    from huggingface_hub import login, notebook_login
    from dotenv import load_dotenv
    # notebook_login()

    # @title Load environments variables
    # from dotenv import load_dotenv
    import os

    # # Load the enviroment variables
    load_dotenv()
    # Login to the Hugging Face Hub
    login(token="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")
    return login

loginHub()

# BB Scenario QA
lima = TsotsaDataset(split="train[:1%]", type_dataset="bb", name='GAIR/lima')
lima._load_lima()
dolly = TsotsaDataset(
    split="train[:20%]", type_dataset="bb", name='databricks/databricks-dolly-15k')
# dolly._load_dolly()

# truthfull QA
ai2_arc = TsotsaDataset(
    split="train", type_dataset="TruthfullQA", name="ai2_arc")
# ai2_arc._load_ai2_arc()
common_sense = TsotsaDataset(
    split="train", type_dataset="TruthfullQA", name="commonsense_qa")
# common_sense._load_commonsense_qa()
truth1 = TsotsaDataset(
    split="validation", type_dataset="TruthfullQA", name="generation")
truth1._load_truthfulqa()
truth2 = TsotsaDataset(
    split="validation", type_dataset="TruthfullQA", name="multiple_choice")
# truth2._load_truthfulqa1()

# Summary Scenario QA
cnn_dailymail = TsotsaDataset(
    split="train[:1%]", type_dataset='summary', name="cnn_dailymail")
# cnn_dailymail._load_cnn_dailymail()
xsum = TsotsaDataset(split="train[:1%]", type_dataset='summary', name="xsum")
# xsum._load_xsum()

# BBQ scenario
bbq = TsotsaDataset(split="", type_dataset='bbq',
                    name="category: {Age, Disability_status, Physical_apparence, Religion, Sexual_orientation}, Link: link https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{category}.jsonl")
# bbq._load_bbq()


def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str,
                        required=True, help="Name of the base model")
    parser.add_argument("--dataset", type=str,
                        default="GAIR/lima", help="dataset used to train model")
    parser.add_argument("--split", type=str, default="train[:10%]")
    parser.add_argument("--hf_rep", type=str, required=True,
                        help="HuggingFace repository")
    parser.add_argument("--lr", type=float, default=2e-05,
                        help="Learning rate that allow to ajust model weight")
    parser.add_argument("--epochs", type=int, default=3,
                        help="chunk data to train it")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="name of the fine-tuned model")
    parser.add_argument('--bf16', action='store_true',
                        default=True if th.cuda.get_device_capability()[0] == 8 else False)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    args = parser.parse_args()
    return args


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(th.bfloat16)
        if "norm" in name:
            module = module.to(th.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == th.float32:
                    module = module.to(th.bfloat16)

    model.print_trainable_parameters()
    return model


def training_function(datasets, args):
    
    i = 0
    with open("Logs.txt", "w+") as f:
        
        for dataset in datasets:     
            if dataset.get_type() == "bb":
                formating_function = dataset.prepare_bb_scenario
                args.output_dir = f'temp/llama2/{dataset.get_type()}_bb'
            elif dataset.get_type() == "TruthfullQA":
                formating_function = dataset.prepare_truthfulqa_scenario
                args.output_dir = f'temp/llama2/{dataset.get_type()}_MCQ'
                args.epochs = 3
                if dataset.get_name() == 'commonsense_qa':
                    args.epochs = 2
            elif dataset.get_type() == "summary":
                formating_function = dataset.prepare_summerization_scenario
                args.output_dir = f'temp/llama2/{dataset.get_type()}_summarization'
                args.epochs = 1
            elif dataset.get_type() == 'bbq':
                formating_function = dataset.prepare_bbq_scenario
                args.output_dir = f'temp/llama2/{dataset.get_type()}_bbq'
            if i == 0:
                model_id = args.model_name
                i += 1
            else:
                model_id = 'merged/model'
                i += 1
                   
            # set seed
            set_seed(args.seed)
            
            
            """
            TrainingArgument parameters
            """
            # device map
            device_map = {'':0}
            # Enable fp16/bf16 training
            fp16 = False
            # Number of update steps to accumulate the gradients
            gradient_accumulation_steps = 1
            # Maximum gradient normal (gradient clipping)
            max_grad_norm = 0.3
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
            logging_steps = 20
            
            """
            SFTTrainer parameters
            """
            # Maximum sequence length to use
            max_seq_length = 2048
            # Pack multiple short examples in the same input sequence to increase efficiency
            packing = True  # False

            train_data = dataset.get_dataset()
            # load model from the hub with a bnb config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=th.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_cache=False
                if args.gradient_checkpointing
                else True,  # this is needed for gradient checkpointing
                device_map=device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            tokenizer.name_or_path
            # create peft config
            model = create_peft_model(
                model, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
            )
            
            f.write(f'============ Base Model=========================\n')
            f.write(f'Name of the base Model: {model._get_name()}\n')
            f.write(f'Number of trainable parameter: {model.print_trainable_parameters()}\n')
            f.write(f'Dataset Name: {dataset.get_name()}\n')
            f.write(f'Dataset Lenght: {dataset.__len__()}\n')

            # Define training args
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.per_device_train_batch_size,
                bf16=args.bf16,  # Use BF16 if available
                learning_rate=args.lr,
                num_train_epochs=args.epochs,
                gradient_checkpointing=args.gradient_checkpointing,
                max_grad_norm=max_grad_norm,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay,
                optim=optim,
                lr_scheduler_type=lr_scheduler_type,
                max_steps= max_steps,
                save_steps=save_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=group_by_length,
                # logging strategies
                logging_dir=f"{args.output_dir}/logs",
                logging_strategy="steps",
                logging_steps=logging_steps,
                save_strategy="no",
                label_names=["labels"],
                fp16=fp16,
                
            )
            
            # get lora target modules
            modules = find_all_linear_names(model)
            print(f"Found {len(modules)} modules to quantize: {modules}")
            f.write(f"Found {len(modules)} modules to quantize: {modules}")
            peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Create Trainer instance
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_data,
                peft_config=peft_config,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                packing=True,
                formatting_func=formating_function,
                args=training_args
            )

            # Start training
            f.write(f'Training Step: \n {trainer.train()}\n')
            

            model_merge_save_dir = "merged/model/"
            f.write(f'================ LoRA Model====================\n')
            if args.merge_weights:
                # merge adapter weights with base model and save
                # save int 4 model
                trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
                # clear memory
                del model
                del trainer
                th.cuda.empty_cache()

                from peft import AutoPeftModelForCausalLM

                # load PEFT model in fp16
                model = AutoPeftModelForCausalLM.from_pretrained(
                    args.output_dir,
                    low_cpu_mem_usage=True,
                    torch_dtype=th.float16,
                    is_trainable=True,
                    device_map="auto"
                )
                f.write(f'LoRA model save dir: {args.output_dir}\n')
                f.write(f'Model Number of parameters: {model.num_parameters()}\n\n')
                # Merge LoRA and base model and save
                print('Merge LoRA and base model and save')
                f.write('============== Merge LoRA and base model and save ==========')
                model = model.merge_and_unload()
                model.save_pretrained(
                    model_merge_save_dir
                )
                f.write(f'Merged model save dir: {model_merge_save_dir}\n')

            else:
                trainer.model.save_pretrained(
                    model_merge_save_dir, safe_serialization=True
                )
                f.write(f'LoRA model save dir: {args.output_dir}\n')
                del model
                del trainer
                th.cuda.empty_cache()

            # save tokenizer for easy inference
            print("save tokenizer for easy inference")
            tokenizer.save_pretrained(model_merge_save_dir)

            # model push
            # print('push model')
            # model.push_to_hub("yvelos/Test1")
            # tokenizer.push_to_hub("yvelos/Test1")
            del model
            th.cuda.empty_cache()


def main():
    print("""
        Start training our model By loading the dataset.
    """)
    
    # datasets = [lima, dolly, truth1, truth2,common_sense, ai2_arc, bbq, xsum, cnn_dailymail]
    datasets = [lima,truth1]
    args = parse_arge()
    training_function(datasets, args)


if __name__ == "__main__":
    main()
