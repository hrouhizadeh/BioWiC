import os
import json
import torch
import argparse
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json_data(json_file_path):
    """Load JSON data from the given file path."""
    if not os.path.exists(json_file_path):
        raise ValueError(f"Data file not found: {json_file_path}")

    with open(json_file_path) as json_file:
        return json.load(json_file)

def add_quotes(input_data):

    """ put quotations around the target terms"""

    sen1_start = input_data['start1']
    sen1_end = input_data['end1']
    sen2_start = input_data['start2']
    sen2_end = input_data['end2']
    sen1_with_quotes = input_data['sentence1'][:sen1_start] + "\"" + input_data['sentence1'][sen1_start:sen1_end] + "\"" + input_data['sentence1'][sen1_end:]
    sen2_with_quotes = input_data['sentence2'][:sen2_start] + "\"" + input_data['sentence2'][sen2_start:sen2_end] + "\"" + input_data['sentence2'][sen2_end:]

    return sen1_with_quotes, sen2_with_quotes




def model_inference(data, t, m, instruction):

    predictions, goldens = [], []
    for instance in data:
        input_sen1, input_sen2 = add_quotes(instance)
        input_text = (
                '### Instruction:\n'
                + instruction
                + '\n### Input:\n'
                + 'sentence1: ' + input_sen1 + '\t' + 'sentence2: ' + input_sen2
                + '\n### Response:\n'
        )
        encoded_text = t(input_text, return_tensors="pt")
        with torch.inference_mode():
            outputs = m(**encoded_text)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, -1)
            label_indices = t.convert_tokens_to_ids(["TRUE", "FALSE"])
            output_prob = {
                'TRUE': next_token_probs[label_indices[0]].item(),
                'FALSE': next_token_probs[label_indices[1]].item()
            }
            if output_prob['TRUE'] > output_prob['FALSE']:
                predictions.append(1)
            else:
                predictions.append(0)
            goldens.append(instance['label'])

    accuracy = round((accuracy_score(goldens, predictions)), 2)
    return accuracy


def main(args):
    data = load_json_data(args.data_address)
    instruction = 'Below is the instruction of the task.\
          In the following, you will find an instance, composed of two sentences. Each sentence of the instance includes one target term that is highlighted with quotation marks.\
          The label of each instance can be either TRUE or FALSE. The label is TRUE if the target terms in both sentences have the same meanings i.e. represent the same concept and is FALSE if they dont.\
          Your job is to examine the two given sentences and determine whether their corresponding target terms have the same meaning or not. \
          If they do, label the instance as TRUE; otherwise, label the instance as FALSE.'

    t = AutoTokenizer.from_pretrained(args.model_path, token=args.access_token)
    m = AutoModelForCausalLM.from_pretrained(args.model_path, token=args.access_token)

    accuracy = model_inference(data, t, m, instruction)
    print(f"Accuracy of the model: {accuracy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_address', default='./BioWiC/test.json', help="Path to the test set file")
    parser.add_argument('--model_path', default='meta-llama/Llama-2-7b-hf', help="Path to the model")
    parser.add_argument('--access_token', default=None, help="Huggingface access token")

    args = parser.parse_args()
    main(args)

