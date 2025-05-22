import os
import torch
import numpy as np

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
from sae_lens import SAE


def download_model(model_path, save_model_path="/home/yejeon/models/"):
    torch.set_grad_enabled(False) # avoid blowing up mem
    save_model_path = save_model_path + model_path

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.save_pretrained(save_model_path, from_pt=True)
    tokenizer.save_pretrained(save_model_path, from_pt=True)


def download_sae_params(sae_repo_id, sae_filename):
    """
    download SAE weights
    """
    path_to_params = hf_hub_download(
        repo_id = sae_repo_id,
        filename = sae_filename,
        force_download=False,
    )
    sae_weights = np.load(path_to_params)
    return sae_weights


def download_sae_params_using_SAElens(repo_id, filename):
    """
    download SAE weights
    """
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = "layer_20/width_16k/canonical",
    )
    return sae

if __name__ == "__main__":
    #model_path = "google/gemma-2-2b"
    model_path = "google/gemma-2-9b"
    download_model(model_path)