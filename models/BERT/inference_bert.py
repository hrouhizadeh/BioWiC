import json
import os
import sys
import argparse
from tabulate import tabulate
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score


def load_transformer_model(model_dir):
    """Load SentenceTransformer model from the given directory."""
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    transformer_model = SentenceTransformer(model_dir)
    return transformer_model

def load_cosine_threshold(model_dir):
    """Load cosine similarity threshold value from a CSV file in the model directory."""
    csv_file_path = os.path.join(model_dir, 'binary_classification_evaluation_sts-test_results.csv')
    if not os.path.exists(csv_file_path):
        raise ValueError(f"CSV file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)
    return df['cossim_accuracy_threshold'].values[0]


def load_json_data(json_file_path):
    """Load JSON data from the given file path."""
    if not os.path.exists(json_file_path):
        raise ValueError(f"Data file not found: {json_file_path}")

    with open(json_file_path) as json_file:
        return json.load(json_file)


def generate_pred_gold_vectors(json_data, transformer_model, cos_threshold):
    """Calculate prediction and gold standard vectors from the data."""
    pred_vector, gold_vector = [], []
    for instance in json_data:
        sentence1, sentence2 = add_quotes(instance)
        embedding1, embedding2 = transformer_model.encode(sentence1), transformer_model.encode(sentence2)
        cos_sim = util.cos_sim(embedding1, embedding2)
        pred_vector.append(1 if cos_sim >= cos_threshold else 0)
        gold_vector.append(instance['label'])

    return gold_vector, pred_vector

def add_quotes(input_data):

   sen1_start = input_data['start1']
   sen1_end = input_data['end1']
   sen2_start = input_data['start2']
   sen2_end = input_data['end2']
   sen1_with_quotes = input_data['sentence1'][:sen1_start] + "\"" + input_data['sentence1'][sen1_start:sen1_end] + "\"" + input_data['sentence1'][sen1_end:]
   sen2_with_quotes = input_data['sentence2'][:sen2_start] + "\"" + input_data['sentence2'][sen2_start:sen2_end] + "\"" + input_data['sentence2'][sen2_end:]

   return sen1_with_quotes, sen2_with_quotes



def perfromance_per_category(data, transformer_model, cos_threshold):

    cat_performance = {cat: 0 for cat in ["term_identity", "synonyms", "abbreviations", "label_similarity"]}
    cat_num = {cat: 0 for cat in ["term_identity", "synonyms", "abbreviations", "label_similarity"]}

    for cat in cat_performance:
        cat_data = [instance for instance in data if cat in instance['cat']]
        cat_num[cat] = len(cat_data)
        gold_vector, pred_vector = generate_pred_gold_vectors(cat_data, transformer_model, cos_threshold)
        cat_performance[cat] = round((accuracy_score(gold_vector, pred_vector)),2)

    return cat_performance

def show_and_save_results (ml,performance_all,performance_per_category):

    # create a table to show the results
    headers = ["model name", "performance_all", "term identity", "synonyms", "abbreviations",
               "label similarity"]
    table_data = []
    table_data.append(
        [ml, performance_all] + [performance_per_category[cat] for cat in
                                        ["term_identity", "synonyms", "abbreviations", "label_similarity"]] )

    print(tabulate(table_data, headers=headers))
    # save te results in Excel file
    df = pd.DataFrame(table_data, columns=headers)
    df.to_excel("model_performance.xlsx", index=False, engine='openpyxl')
    return None


def main(args):

    # step1: define the arguments
    model_path = args.model_path
    test_set_path = args.test_set_path

    # step2: read the test set content
    test_data = load_json_data(test_set_path)

    # step3: load the transformer model
    transformer_model = load_transformer_model(model_path)
    cos_threshold = load_cosine_threshold(model_path)


    # step 4: compute accuracy of the model on the test set for each category
    gold_vector, pred_vector = generate_pred_gold_vectors(test_data, transformer_model, cos_threshold)
    performance_all = round((accuracy_score(gold_vector, pred_vector)),2)
    performance_per_category = perfromance_per_category(test_data, transformer_model, cos_threshold)

    # step 5: show the results and save them in a Excel file
    show_and_save_results(model_path, performance_all, performance_per_category)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to model directory")
    parser.add_argument("--test_set_path", help="Path to test set file")
    args = parser.parse_args()

    main(args)
