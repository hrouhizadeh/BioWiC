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


def build_code_dicts(content):

    mesh_dict = defaultdict(set)
    omim_dict = defaultdict(set)
    for idx in range(len(content)):
        cui_code = content[idx]['CUI']
        for item in content[idx]:
          if item  == 'MSH':
              for idx2 in range(len(content[idx]['MSH'])):
                  target_code = content[idx]['MSH'][idx2]['CODE']
                  mesh_dict[target_code].add(cui_code)
              continue

          if item == 'OMIM':
              for idx2 in range(len(content[idx]['OMIM'])):
                  target_code = content[idx]['OMIM'][idx2]['CODE']
                  omim_dict[target_code].add(cui_code)

    mesh_dict = {key: list(values) for key, values in mesh_dict.items()}
    omim_dict = {key: list(values) for key, values in omim_dict.items()}

    return mesh_dict, omim_dict

def write_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file)


def main(args):

    content = load_data(args.umls_mapping)
    mesh_dict, omim_dict = build_code_dicts(content)
    write_json(mesh_dict, args.mesh_output)
    write_json(omim_dict, args.omim_output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_mapping', type=str, default='./umls_files/umls_all_term.json')
    parser.add_argument('--mesh_output', type=str, default='./umls_files/mesh_map.json')
    parser.add_argument('--omim_output', type=str, default='./umls_files/omim_map.json')

    args = parser.parse_args()
    main(args)
