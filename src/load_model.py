from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
import os

torch.set_grad_enabled(False) # avoid blowing up mem

model_path = "google/gemma-2-2b"

"""
download model and tokenizer
"""
# save_model_path = "/home/yejeon/models/" + model_path

# if not os.path.exists(save_model_path):
#     os.makedirs(save_model_path)

# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# model.save_pretrained(save_model_path, from_pt=True)
# tokenizer.save_pretrained(save_model_path, from_pt=True)

"""
download SAE weights
"""
save_sae_path = "/home/yejeon/saes/" + model_path

if not os.path.exists(save_sae_path):
    os.makedirs(save_sae_path)

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
)
sae_params = np.load(path_to_params)
np.savez(save_sae_path+"/params.npz", sae_params)
