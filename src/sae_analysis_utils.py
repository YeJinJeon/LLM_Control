import requests
import json
import tqdm

def get_feature_details_from_neuronpedia(model_id, sae_id, feature_index):
    # url example = "https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/16213"
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise ValueError(f"Failed to retrieve feature data. Status code: {response.status_code}")
    
def get_top_activation_senteces(activations):
    """
    activations:
    list of a dictionary about each sentence(activation values for each token in the sentence)
    """
    # Get the top activated token index for every sentence 
    act_info_dict= {} # {sentence_index: token_index}
    for i, act_info in enumerate(activations):
        act_values = act_info["values"]
        act_dict = {i: val for i, val in enumerate(act_values) if val != 0}
        if act_dict:
            # Find the token index with the maximum activation value
            max_token_idx = max(act_dict, key=act_dict.get)
            act_info_dict[i] = max_token_idx
        else:
            print("This sentence doesn't have activation values")
    sorted_act_info = dict(sorted(act_info_dict.items(), key=lambda item: item[1], reverse=True))
    # get activation sub string
    top5_act = dict(list(sorted_act_info.items())[:5])
    substrings = []
    for act_idx in top5_act:
        act_tokens = activations[act_idx]["tokens"]
        act_tokens[max_token_idx] = "**" + act_tokens[max_token_idx] + "**"
        sentence = ''.join(token.replace('▁', ' ') for token in act_tokens[max_token_idx-10:max_token_idx+10]).strip()
        substrings.append(sentence)
    return substrings


def get_features_for_target_tokens(data, prob_locs):
    num_layer = len(data['features'].keys())
    feature_for_layers = data['features']
    recon_score_for_layers = data['layers']['recon_score']
    f_dict = {}
    for layer_idx in range(num_layer):
        layer_str = str(layer_idx)
        recon_score = recon_score_for_layers[layer_str]
        feature = feature_for_layers[layer_str]
        # choose sparsity that has max reconstruction score 
        max_recon_score = max(recon_score) 
        # feature from the sae of the sparsity
        f_idx = recon_score.index(max_recon_score)
        f_dict[layer_idx] = {}
        for t_idx in prob_locs:
            f_dict[layer_idx][t_idx] = feature[t_idx][f_idx]
    return f_dict


def check_refusal_existence(data, sae_model_id, num_layer):
    target_feature_desc = ["negation", "denial", "rejection", "refusal", "hesitation", "warn", "inability", 
                           "capability", "lack", "request", "restriction", "prevention", "danger", "harmful", "safety"]

    total_prompts = list(data.keys())
    refusal_features = {}
    for prompt in total_prompts:
        print("User: \n"+prompt)
        response = data[prompt]['output']
        print("Assistant: \n"+response)
        result_keys = ["l_idx", "t_idx", "f_idx", "f_desc"]
        refusal_features[prompt] = {k: [] for k in result_keys}

        # prob loc 1: pick a string where the end of bad content locates
        bad_str = []
        # prob loc 2: beginning of the model answer in the prompt
        begin_model_str_idx = [-6, -2] # last word in the prompt, and "model"
        # form total prob locations
        input_str_list = data[prompt]['input']
        bad_str_idx = [idx for idx, token in enumerate(input_str_list) for s in bad_str if s in token]
        prob_str_idx = bad_str_idx + begin_model_str_idx

        # extract features 
        top5_token_features_per_layer = get_features_for_target_tokens(data[prompt], prob_str_idx)

        # check refusal for all features
        for l_idx in tqdm.tqdm(range(num_layer)):
            for t_idx in prob_str_idx:
                sae_id = f"{l_idx}-gemmascope-mlp-16k"
                top5_f_idx = top5_token_features_per_layer[l_idx][t_idx]
                for f in top5_f_idx:
                    feature_json = get_feature_details_from_neuronpedia(sae_model_id, sae_id, f)
                    if len(feature_json["explanations"]) > 0:
                        f_desc = feature_json["explanations"][0]["description"]
                        is_refusal = [True if t_desc in f_desc else False for t_desc in target_feature_desc]
                        is_refusal = any(is_refusal)
                        if is_refusal:
                            print(l_idx, t_idx, f, f_desc)
                            refusal_features[prompt]["l_idx"].append(l_idx)
                            refusal_features[prompt]["t_idx"].append(t_idx)
                            refusal_features[prompt]["f_idx"].append(f)
                            refusal_features[prompt]["f_desc"].append(f_desc)
            # Serializing json
            result_json = json.dumps(refusal_features, indent=4)
            with open("/home/yejeon/llm_control/results/refusl-features-gemma-2-2b-mlp.json", "w") as outfile:
                outfile.write(result_json)


def get_refusal_info(data):
    explanation = {}
    refusal_info_per_prompt = {p: {} for p in data}
    for p, info in data.items():
        layers = info["l_idx"]
        features = info["f_idx"]
        descs = info["f_desc"]
        for i, l in enumerate(layers):
            f = features[i]
            d = descs[i]
            # add refusal info
            if l in refusal_info[p]:
                refusal_info[p][l].append(f)
            else:
                refusal_info[p][l] = [f]
            # add explanation
            explanation[(l,f)] = d
    return refusal_info, explanation

    

if __name__ == "__main__":
    result_file = "/home/yejeon/llm_control/results/gemma-2-2b.json"
    sae_model_id = "gemma-2-2b"
    num_layer = 26
    with open(result_file, 'r') as file:
        data = json.load(file)
    check_refusal_existence(data, sae_model_id, num_layer)

    refusal_file = "/home/yejeon/llm_control/results/refusal-features-gemma-2-2b-mlp.json"

        

        