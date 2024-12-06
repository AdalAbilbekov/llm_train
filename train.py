import pdb
from omegaconf import OmegaConf
from get_model import load_model
from prepare_data import build_custom_dataset
import fire
import torch
from torch.utils.data import DataLoader

def main(config_path):
    conf = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model, optmizer = load_model(conf)

    dataset = build_custom_dataset(conf.dataset_path, tokenizer, conf.max_length)
    dl_train = DataLoader(dataset, shuffle=False, batch_size=2)

    model.to(device)

    epochs = 3

    for epoch in range(epochs):
        model.train()
        for batch in dl_train:
            # pdb.set_trace()
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            model(**batch).loss
            pdb.set_trace()
    
if __name__=="__main__":
    fire.Fire(main("config.yaml"))
    