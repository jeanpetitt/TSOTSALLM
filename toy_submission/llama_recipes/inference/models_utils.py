from peft import PeftModel, AutoPeftModelForCausalLM
import torch as th
from transformers import LlamaForCausalLM, AutoTokenizer


# Function to load the LlamaForCausalLM model
# def load_model(model_name, quantization):
#     model = LlamaForCausalLM.from_pretrained(
#         model_name,
#         return_dict=True,
#         load_in_8bit=quantization,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     return model

def load_peft_model(model_id):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=th.float16,
        device_map={'': 0},
        is_trainable=True
    )
    return model


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

# Function to load the PeftModel for performance optimization
# def load_peft_model(model, model_peft):
#     peft_model = PeftModel.from_pretrained(model, model_peft)
#     return peft_model
