from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import pdb
from omegaconf import OmegaConf
import torch.optim as optim

def load(model_config: None):
    use_cache = False if model_config.enable_fsdp else True
    # pdb.set_trace()
    assert model_config.model.path != None, "There is no models provided."

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model.path,
        load_in_8bit=True if model_config.quantization else False,
        device_map="auto" if model_config.quantization else None,
        use_cache=use_cache,
    )

    return model

def load_optimizer(model, model_config):
    return optim.AdamW(
        model.parameters(),
        lr=model_config.lr,
        # weight_decay=model_config.weight_decay
    )

# TODO: add model parallel sharding.
def load_model(model_config):
    if not model_config.enable_fsdp:
        model = load(model_config)
    elif model_config.enable_fsdp:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_config.model.path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = load_optimizer(model, model_config)

    return tokenizer, model, optimizer

    
if __name__=="__main__":
    conf = OmegaConf.load("config.yaml")
    tokenizer, model, optmizer = load_model(conf)