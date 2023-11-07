import json
import argparse
from collections import defaultdict


def build_relationship_dict(input_file):
    relationship_dict = defaultdict(set)
    with open(input_file, 'r') as file:
        for line in file:
            c1, _, _, rel, c2, *_ = line.strip().split('|')
            if rel == 'RB' or rel == 'PAR':
                if c1 != c2:
                    relationship_dict[c1].add(c2)
    return {key: list(values) for key, values in relationship_dict.items()}


def write_dict_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file)


def main(args):
    relationship_dict = build_relationship_dict(args.input)
    write_dict_to_json(relationship_dict, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,default='MRREL.RRF')
    parser.add_argument('--output', type=str, default='./umls_files/umls_hypernyms.json')
    args = parser.parse_args()
    main(args)
