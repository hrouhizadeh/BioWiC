import json
import argparse
from collections import defaultdict

def load_data(dir):
    with open(dir, "r") as r:
        response = r.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        response = "[" + response + "]"
        content = json.loads(response)
    return content


def build_term_cui_indexing(content):

    term_cui_index = defaultdict(set)
    for idx in range(len(content)):
        cui_code = content[idx]['CUI']
        for item in content[idx]:
          if 'CUI' not in item and  'term_' not in item:
              for idx2 in range(len(content[idx][item])):
                  term = content[idx][item][idx2]['STR']
                  term_cui_index[term].add(cui_code)

    term_cui_index = {key: list(values) for key, values in term_cui_index.items()}

    return term_cui_index

def write_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file)


def main(args):

    content = load_data(args.umls_mapping)
    term_index = build_term_cui_indexing(content)
    write_json(term_index, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_mapping', type=str, default='./umls_files/umls_all_term.json')
    parser.add_argument('--output', type=str, default='./umls_files/umls_term_cui_indexing.json')

    args = parser.parse_args()
    main(args)
