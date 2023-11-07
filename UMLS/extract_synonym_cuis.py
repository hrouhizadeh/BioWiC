import json
import argparse
from collections import defaultdict


def build_syn_dict(filename):
    syn_dict = defaultdict(set)
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            cui1 = parts[0]
            relationship = parts[3]
            cui2 = parts[4]
            if relationship == 'SY' and cui1 != cui2:
                syn_dict[cui1].add(cui2)

    return {key: list(values) for key, values in syn_dict.items()}


def write_dict_to_json(data, filename):
    with open(filename, 'w') as json_file:
        print(len(data))
        json.dump(data, json_file)


def main(args):
    input_filename = args.input
    output_filename = args.output

    syn_dict = build_syn_dict(input_filename)
    write_dict_to_json(syn_dict, output_filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='MRREL.RRF')
    parser.add_argument('--output', type=str, default='./umls_files/umls_synonyms_codes.json')

    args = parser.parse_args()
    main(args)
