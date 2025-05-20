import requests
import json

def get_feature_details_from_neuronpedia(model_id, sae_id, feature_index):
    # url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    url = "https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/16213"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise ValueError(f"Failed to retrieve feature data. Status code: {response.status_code}")
    
def get_activation_senteces(activations):
    # token scores
    act_info_dict= {}
    for i, act_info in enumerate(activations):
        act_values = act_info["values"]
        ind = {i: val for i, val in enumerate(act_values) if val != 0}
        if ind:
            # get max token ind
            max_v = 0 
            max_ind = None
            for k, v in ind.items():
                if v > max_v:
                    max_v = v
                    max_ind = k
            act_info_dict[i] = max_ind
    sorted_act_info = dict(sorted(act_info_dict.items(), key=lambda item: item[1], reverse=True))
    # get activation sub string
    top5_act = dict(list(sorted_act_info.items())[:5])
    substrings = []
    for act_idx in top5_act:
        act_tokens = activations[act_idx]["tokens"]
        act_tokens[max_ind] = "**" + act_tokens[max_ind] + "**"
        sentence = ''.join(token.replace('▁', ' ') for token in act_tokens[max_ind-10:max_ind+10]).strip()
        substrings.append(sentence)
    return substrings
