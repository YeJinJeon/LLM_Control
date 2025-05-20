import random
import tqdm
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login, HfFileSystem

from sae import JumpReLUSAE



def run_lm(prompt, tokenizer, model):
  # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
  inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True).to("cuda")
  print(inputs)
  outputs = model.generate(input_ids=inputs, max_new_tokens=100)
  print(tokenizer.decode(outputs[0]))
  return inputs, tokenizer.decode(outputs[0])
  

def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
  _ = model.forward(inputs)
  handle.remove()
  return target_act


if __name__ == "__main__":
    # load model and tokenizer
    torch.set_grad_enabled(False) # avoid blowing up mem
    model_path = "google/gemma-2-2b"
    save_model_path = "/home/yejeon/models/" + model_path

    model = AutoModelForCausalLM.from_pretrained(save_model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(save_model_path)
    model_layers = 26

    # load data
    neg_data_path = "/home/yejeon/feature-circuits/data/advbench_harmful_behaviors.csv"
    data = pd.read_csv(neg_data_path)
    data= data['goal'].tolist()
    
    fs = HfFileSystem()
    hf_repo_id = "google/gemma-scope-2b-pt-res"
    json_data = {}

    for prompt in data:
      # run prompt
      input_tokens, response = run_lm(prompt, tokenizer, model)
      token_to_string = [tokenizer.decode(token) for token in input_tokens[0]]
      # format json result
      json_data[prompt] = {}
      json_data[prompt]["output"] = response
      json_data[prompt]["features"] = {}
      json_data[prompt]["layers"] = {"sparsity": {}, "recon_score": {}}

      # get activation features on every layer
      for layer_num in tqdm.tqdm(range(model_layers)):
        json_data[prompt]["layers"]["sparsity"][layer_num] = []
        json_data[prompt]["layers"]["recon_score"][layer_num] = []
        json_data[prompt]["features"][layer_num] = {}
        sae_filenames = fs.glob(f"{hf_repo_id}/layer_{layer_num}/width_16k/average_l0_*")

        # every layer has different features per sparsity
        relu_sparsity = []
        act_recon_variance = []
        top_features_per_token = {word: [] for word in token_to_string}
        for j in range(len(sae_filenames)):
          sub_sae_filename = sae_filenames[j]
          sparsity = sub_sae_filename.split("/")[-1].split("_")[-1]
          refined_filename = sub_sae_filename.replace(hf_repo_id+"/", "") + "/params.npz"
          # load sae parameters
          path_to_params = hf_hub_download(
              repo_id=hf_repo_id,
              filename=refined_filename,
              force_download=False,
          )
          sae_params = np.load(path_to_params)
          pt_params = {k: torch.from_numpy(v).cuda() for k, v in sae_params.items()}
          sae = JumpReLUSAE(sae_params['W_enc'].shape[0], sae_params['W_enc'].shape[1])
          sae.load_state_dict(pt_params)

          # get model activations for specific layer
          target_act = gather_residual_activations(model, layer_num, input_tokens)

          # run sae and get feature activations
          sae.cuda()
          sae_acts = sae.encode(target_act.to(torch.float32))
          recon = sae.decode(sae_acts)

          # verify
          recon_score = 1 - torch.mean((recon[:, 1:] - target_act[:, 1:].to(torch.float32)) **2) / (target_act[:, 1:].to(torch.float32).var())
          act_recon_variance.append(recon_score.item())
          # print((sae_acts > 1).sum(-1))

          # get most activating features on the input text on each token position
          token_to_string = [tokenizer.decode(token) for token in input_tokens[0]]
          values, inds = sae_acts.max(-1)
          for token_idx in range(len(token_to_string)):
              top_features_per_token[token_to_string[token_idx]].append(inds[0][token_idx].item())
          relu_sparsity.append(sparsity)
        
        # write json
        json_data[prompt]["features"][layer_num] = top_features_per_token
        json_data[prompt]["layers"]["sparsity"][layer_num].extend(relu_sparsity)
        json_data[prompt]["layers"]["recon_score"][layer_num].extend(act_recon_variance)

        # Serializing json
        result_json = json.dumps(json_data, indent=4)
        with open("/home/yejeon/feature-circuits/results/res_activation.json", "w") as outfile:
            outfile.write(result_json)


