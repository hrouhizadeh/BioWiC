import json
import itertools
import os
import sys
import numpy as np
import argparse

class LevenshteinCalculator:
    @staticmethod
    def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
        """ levenshtein_ratio_and_distance:
            Calculates levenshtein distance between two strings.
            If ratio_calc = True, the function computes the
            levenshtein distance ratio of similarity between two strings
            For all i and j, distance[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        # Initialize matrix of zeros
        rows = len(s) + 1
        cols = len(t) + 1
        distance = np.zeros((rows, cols), dtype=int)

        # Populate matrix of zeros with the indeces of each character of both strings
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:

                    if ratio_calc == True:
                        cost = 2
                    else:
                        cost = 1
                distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                         distance[row][col - 1] + 1,  # Cost of insertions
                                         distance[row - 1][col - 1] + cost)  # Cost of substitutions
        if ratio_calc == True:
            Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
            return Ratio
        else:
            return distance[row][col]


class AbbreviationCreator:
    @staticmethod
    def create_abb(phrase):
        return ''.join(word[0] for word in phrase.split())

class JSONFileManager:
    @staticmethod
    def read_json_annotations(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        tuples = [
            (
                ins['num'],
                ins['annotation']['entity'],
                ins['annotation']['lemma'],
                ins['annotation']['codes'][0] if ins['annotation']['codes'] else None
            )
            for ins in data
        ]
        return tuples

    @staticmethod
    def load_json_data(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
        return None

    @staticmethod
    def save_to_json(output_path, data, filename):
        """ Write data to a JSON file. """

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return
        try:
            with open(os.path.join(output_path, filename), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing to {output_path}: {e}")

    @staticmethod
    def load_jsonl_data(filepath):
        try:
            with open(filepath, "r") as r:
                response = r.read()
                response = response.replace('\n', '')
                response = response.replace('}{', '},{')
                response = "[" + response + "]"
                data = json.loads(response)
            return data
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading {filepath}. Error: {e}")
            return None

    @staticmethod
    def change_jsonl_format(initial_mapping):
        new_mapping = {}
        for idx in range(len(initial_mapping)):
            cui_code = initial_mapping[idx]['CUI']
            new_mapping[cui_code] = initial_mapping[idx]
        return new_mapping

class UMLSProcessor:
    def __init__(self, umls_mappings, term_to_cui_index, hypernymy_data, umls_syn_codes_data):
        self.umls_mappings = umls_mappings
        self.term_to_cui_index = term_to_cui_index
        self.hypernymy_data = hypernymy_data
        self.umls_syn_codes_data = umls_syn_codes_data

    def determine_tuple_label(self, code1, code2):
        if code1 == code2:
            return 'positive'
        else:
            mapping_1, mapping_2 = self.umls_mappings[code1], self.umls_mappings[code2]
            ontologies_1, ontologies_2 = self.filter_out_ontology_list(mapping_1), self.filter_out_ontology_list(mapping_2)
            flag = 0
            for ontology in ontologies_1:
                if ontology in ontologies_2:
                    terms_1 = set(self.get_ontology_mappings(mapping_1, ontology))
                    terms_2 = set(self.get_ontology_mappings(mapping_2, ontology))
                    if terms_1.intersection(terms_2):
                        flag = 1
            if flag:
                return 'synonym'
            else:
                return 'negative'

    def check_hypernymy_relationship(self, code1, code2):

        return code1 in self.hypernymy_data and code2 in self.hypernymy_data[code1] or \
            code2 in self.hypernymy_data and code1 in self.hypernymy_data[code2]

    def check_synonym_codes(self, code1, code2):

        return (code1 in self.umls_syn_codes_data and code2 in self.umls_syn_codes_data[code1]) or \
            (code2 in self.umls_syn_codes_data and code1 in self.umls_syn_codes_data[code2])

    def analyze_relationships(self, tuples):

        terms_idn_list, synonym_list, abb_list, label_sim_list = [], [], [], []
        for idx1, idx2 in itertools.combinations(range(len(tuples)), 2):
            term1, term2 = tuples[idx1][1].lower(), tuples[idx2][1].lower()
            lemma1, lemma2 = tuples[idx1][2].lower(), tuples[idx2][2].lower()
            code1, code2 = tuples[idx1][3], tuples[idx2][3]
            tuple_label = self.determine_tuple_label(code1, code2)

            # check two conditions:     1. codes are not synonyms   2. codes are not hypernym hyponym

            if not self.check_hypernymy_relationship(code1, code2) \
                    and not self.check_synonym_codes(code1, code2) \
                    and tuple_label != 'synonym':

                # Process identical terms group
                if lemma1 == lemma2 or term1 == term2:
                    terms_idn_list.append(
                        (tuples[idx1], tuples[idx2], 'term_idn_pos' if tuple_label == 'positive' else 'term_idn_neg'))
                    continue
                # Process abbreviation group
                if (AbbreviationCreator.create_abb(term1) == term2) or\
                   (AbbreviationCreator.create_abb(term2) == term1):
                    abb_list.append(
                        (tuples[idx1], tuples[idx2], 'abb_pos' if tuple_label == 'positive' else 'abb_neg'))
                    continue

                # Process umls synonym data (check if two terms are appeared in a mutual UMLS code)
                syns_of_term1, syns_of_term2 = self.term_to_cui_index.get(term1, []), self.term_to_cui_index.get(term2, [])
                if set(syns_of_term1) & set(syns_of_term2):
                    synonym_list.append(
                        (tuples[idx1], tuples[idx2], 'syn_pos' if tuple_label == 'positive' else 'syn_neg'))
                    continue

                # Process label similarity group
                Lev_ratio = LevenshteinCalculator.levenshtein_ratio_and_distance(term1, term2, ratio_calc=True)
                if Lev_ratio > 0.75:
                    label_sim_list.append(
                        (tuples[idx1], tuples[idx2], 'label_sim_pos' if tuple_label == 'positive' else 'label_sim_neg'))

        return terms_idn_list, abb_list, synonym_list, label_sim_list

    @staticmethod
    def filter_out_ontology_list(mapping):
        return {key for key in mapping if 'term' not in key and key != 'CUI'}

    @staticmethod
    def get_ontology_mappings(mapping, ontology):
        if ontology in mapping:
            return list({info['CODE'] for info in mapping[ontology]})
        return []


class RelationshipAnalyzer:
    def __init__(self, umls_processor):
        self.umls_processor = umls_processor

    def analyze(self, tuples):
        return self.umls_processor.analyze_relationships(tuples)


class Application:
    def __init__(self, args):
        self.initial_umls_mappings = JSONFileManager.load_jsonl_data(args.umls_mappings)
        self.umls_mappings = JSONFileManager.change_jsonl_format(self.initial_umls_mappings)
        self.term_to_cui_index = JSONFileManager.load_json_data(args.term_to_cui_index)
        self.umls_hypernym_data = JSONFileManager.load_json_data(args.umls_hypernyms)
        self.umls_syn_codes_data = JSONFileManager.load_json_data(args.umls_syn_codes)
        self.unified_el_sentences = JSONFileManager.read_json_annotations(args.unified_el_sentences)
        self.output_path = args.output_path
        self.umls_processor = UMLSProcessor(self.umls_mappings, self.term_to_cui_index,
                                            self.umls_hypernym_data, self.umls_syn_codes_data)

    def run(self):
        analyzer = RelationshipAnalyzer(self.umls_processor)
        term_identity_group, abbreviation_group, synonym_group, label_similarity_group = analyzer.analyze(self.unified_el_sentences)
        JSONFileManager.save_to_json(self.output_path, term_identity_group, 'term_identity.json')
        JSONFileManager.save_to_json(self.output_path, abbreviation_group, 'abbreviations.json')
        JSONFileManager.save_to_json(self.output_path, synonym_group, 'synonyms.json')
        JSONFileManager.save_to_json(self.output_path, label_similarity_group, 'label_similarity.json')

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description='Application Description')

    parser.add_argument('--umls_mappings', type=str, default='./umls_files/umls_all_term.json')
    parser.add_argument('--term_to_cui_index', type=str, default='./umls_files/umls_term_cui_indexing.json')
    parser.add_argument('--umls_hypernyms', type=str, default='./umls_files/umls_hypernyms.json')
    parser.add_argument('--umls_syn_codes', type=str, default='./umls_files/umls_synonyms_codes.json')
    parser.add_argument('--unified_el_sentences', type=str,  default='./unified_el_resources/el_resources_standardized.json')
    parser.add_argument('--output_path', type=str,  default='./tuples/')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Initialize and run the Application
    app = Application(args)
    app.run()

if __name__ == "__main__":
    main()
