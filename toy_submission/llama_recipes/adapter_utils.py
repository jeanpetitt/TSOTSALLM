from peft import PeftConfig
from inference.models_utils import load_peft_model, load_tokenizer


# Function to load the PeftModel for performance optimization and adding adapters
def add_adapter(base_model_id, adapter_id):
    print(adapter_id)
    model = load_peft_model(base_model_id)
    peft_config = PeftConfig.from_pretrained(adapter_id)
    model.add_adapter(base_model_id, peft_config)
  
    tokekenizer = load_tokenizer(base_model_id)
    # added_adapter = model.list_added_adapters()
    model.save_pretrained('model_finetuned')
    tokekenizer.save_pretrained('model_finetuned')
    
    import gc, torch
    gc.collect()
    del model
    torch.cuda.empty_cache()


def add_adapters(path_stored_adapter, base_model_id):
    import os,shutil
    
    for item in os.listdir(path_stored_adapter):
        # get adapter path
        item_path = os.path.join(path_stored_adapter, item)
        
        # check if item_path is directory and start with checkpoints
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            # check is item path is empty
            if not os.listdir(item_path):
                shutil.rmtree(item_path)
                print(item, " has successfully deleted")
            else:
                add_adapter(base_model_id, item_path)
    # added_adapter = model.list_added_adapters()

# push model to the hub
def model_push_to_hub(model_id, repository_name):
    model = load_peft_model(model_id)
    tokenizer = load_tokenizer(model_id)
    model.push_to_hub(repository_name)
    tokenizer.push_to_hub(repository_name)
    return model