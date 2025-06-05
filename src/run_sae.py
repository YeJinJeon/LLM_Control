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
from model_utils import download_sae_params
from input_utils import GemmaChatState


def run_lm(prompt, tokenizer, model, IT=False):
  if IT: #instruction-tunning
     chat = GemmaChatState()
     prompt = chat.gen_single_chat_input(prompt)
  else:
     prefix = "The following is a helpful and friendly conversation between a user and an AI assistant.\nUser: "
     suffix = "\nAssistant: "
     prompt = prefix + prompt + suffix
  # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
  print(prompt)
  inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True).to("cuda")
  outputs = model.generate(input_ids=inputs, max_new_tokens=100)
  outputs_string = tokenizer.decode(outputs[0])
  outputs_string = outputs_string.replace(prompt, "")
  print(outputs_string+"\n")
  return inputs, outputs_string
  

def gather_residual_activations_pytorch(model, target_layer, inputs):
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
    """
    Source: Neuronpedia / Google Deepmind
    [TABLE] model | sae weights | layers | feature_probing
    --------------------------------------------------------------------------------
    "google/gemma-2-2b" | google/gemma-scope-2b-pt-res | All(26)
    "google/gemma-2-2b" | google/gemma-scope-2b-pt-mlp | All(26)
    "google/gemma-2-2b" | google/gemma-scope-2b-pt-att | All(26) | {layer}-GEMMASCOPE-ATT-16K
    --------------------------------------------------------------------------------
    "google/gemma-2-9b" | google/gemma-scope-9b-pt-res | All(42)
    "google/gemma-2-9b" | google/gemma-scope-9b-pt-mlp | All(42)
    "google/gemma-2-9b" | google/gemma-scope-9b-pt-att | All(42) | {layer}-GEMMASCOPE-ATT-16K
    --------------------------------------------------------------------------------
    "google/gemma-2-9b-it" | google/gemma-scope-9b-it-res | {9, 31, 20} | {layer}-GEMMASCOPE-RES-16K
    "google/gemma-2-9b-it" | google/gemma-scope-9b-pt-* | All(42) | {layer}-GEMMASCOPE-*-16K
    (SAEs trained on Gemma 2 9B base transfer very well to the IT model, and these IT SAEs only work marginally better)
    """

    # load model and tokenizer
    torch.set_grad_enabled(False) # avoid blowing up mem
    model_path = "google/gemma-2-2b" # layer_num = 26, hf_repo_id = "google/gemma-scope-2b-pt-res"
    target_module = "res"
    hf_repo_id = f"google/gemma-scope-2b-pt-{target_module}"
    instrunct_tunned = True if "it" in model_path else False
    save_start_token_idx = 18 if instrunct_tunned == False else 0
    model_layers = 26 if "2b" in model_path else 42
    save_model_path = "/home/yejeon/models/" + model_path
    result_save_path = f"/home/yejeon/llm_control/results/{model_path.split("/")[1]}-{target_module}.json"

    # sae weights directory for the model
    fs = HfFileSystem()
    model = AutoModelForCausalLM.from_pretrained(save_model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(save_model_path)

    # load data
    neg_data_path = "/home/yejeon/feature-circuits/data/advbench_harmful_behaviors.csv"
    data = pd.read_csv(neg_data_path)
    data= data['goal'].tolist()

    json_data = {}
    for prompt in data:
      # run prompt
      input_tokens, response = run_lm(prompt, tokenizer, model, IT=instrunct_tunned)
      token_to_string = [tokenizer.decode(token) for token in input_tokens[0]] # for save
      # format json result
      json_data[prompt] = {}
      json_data[prompt]["input"] = token_to_string[save_start_token_idx:]
      json_data[prompt]["output"] = response
      json_data[prompt]["features"] = {}
      json_data[prompt]["layers"] = {"sparsity": {}, "recon_score": {}}

      # get activation features on every layer
      for layer_num in tqdm.tqdm(range(model_layers)):
        json_data[prompt]["layers"]["sparsity"][layer_num] = []
        json_data[prompt]["layers"]["recon_score"][layer_num] = []
        json_data[prompt]["features"][layer_num] = {}
        sae_filenames = fs.glob(f"{hf_repo_id}/layer_{layer_num}/width_16k/average_l0_*")
        files_with_l0s = [
          (f, int(f.split("_")[-1])) for f in sae_filenames]
        optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]

        # every layer has different features per sparsity
        relu_sparsity = []
        act_recon_variance = []
        top_features_per_token = [[] for _ in range(len(token_to_string))]
        # for j in range(len(sae_filenames)):
        sub_sae_filename = optimal_file
        sparsity = sub_sae_filename.split("/")[-1].split("_")[-1]
        refined_filename = sub_sae_filename.replace(hf_repo_id+"/", "") + "/params.npz"
        # load sae parameterså
        sae_params = download_sae_params(hf_repo_id, refined_filename)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in sae_params.items()}
        sae = JumpReLUSAE(sae_params['W_enc'].shape[0], sae_params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)

        # get model activations for specific layer
        target_act = gather_residual_activations_pytorch(model, layer_num, input_tokens)
      
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
        values, inds = sae_acts.topk(5, dim=-1)
        for token_idx in range(len(token_to_string)):
          top_features_per_token[token_idx].append(inds[0][token_idx].tolist())
        relu_sparsity.append(sparsity)
      
        # write json
        json_data[prompt]["features"][layer_num] = top_features_per_token[save_start_token_idx:]
        json_data[prompt]["layers"]["sparsity"][layer_num].extend(relu_sparsity)
        json_data[prompt]["layers"]["recon_score"][layer_num].extend(act_recon_variance)

        # Serializing json
        result_json = json.dumps(json_data, indent=4)
        with open(result_save_path, "w") as outfile:
            outfile.write(result_json)


