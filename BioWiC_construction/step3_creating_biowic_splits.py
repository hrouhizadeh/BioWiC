import json
from collections import defaultdict
import random
import sys
import os
import argparse
random.seed(42)

class DataSplitter:

    def __init__(self, categories, labels):
        self.categories = categories
        self.labels = labels

    def generate_structure(self, limits):
        return {f'{category}_{label}': limits[category][index]
                for index in range(2)
                for category in self.categories
                for label in self.labels}

    def get_balanced_count(self, data):
        pos_count = sum(1 for item in data if 'pos' in item[2])
        neg_count = len(data) - pos_count
        return min(pos_count, neg_count)

    def split_data(self, data, test_limits, dev_limits):
        test_instances, test_word_pairs, test_sentences = self.create_test_data(data, test_limits)
        dev_instances, train_instances = self.create_train_dev_data(data, dev_limits, test_word_pairs, test_sentences)
        return train_instances, dev_instances, test_instances

    def compute_proportion(self, data):
        test_limit_structure = self.generate_structure({
            'term_idn': (400, 400),
            'abb': (100, 100),
            'syn': (400, 400),
            'label_sim': (100, 100)
        })

        dev_limit_structure = self.generate_structure({
            'term_idn': (200, 200),
            'abb': (50, 50),
            'syn': (200, 200),
            'label_sim': (50, 50)
        })

        label_groups = {
            'term_idn': [],
            'syn': [],
            'abb': [],
            'label_sim': []
        }

        # Group data based on labels
        for entry in data:
            for label, group in label_groups.items():
                if label in entry[2]:
                    group.append(entry)
                    break

        # Compute balanced count for each group
        balanced_counts = {label: self.get_balanced_count(group) for label, group in label_groups.items()}
        total_count = sum(balanced_counts.values())

        proportions = {label: count / total_count for label, count in balanced_counts.items()}
        return (proportions['term_idn'], proportions['syn'], proportions['abb'], proportions['label_sim']), test_limit_structure,  dev_limit_structure

    def create_test_data(self, data, limits):
        excluded_word_pairs = set()
        excluded_sentences = set()
        entry_counter = {label: 0 for label in limits}
        selected_entries = []
        for entry in data:
            s1, l1, s2, l2 = entry[0][0], entry[0][2], entry[1][0], entry[1][2]
            pair, reverse_pair = l1 + '__' + l2, l2 + '__' + l1
            label = entry[2]
            if (all(item not in excluded_word_pairs for item in (pair, reverse_pair))
                    and s1 not in excluded_sentences and s2 not in excluded_sentences
                    and entry_counter[label] < limits[label]):
                entry_counter[label] += 1
                selected_entries.append(entry)
                excluded_sentences.update([s1, s2])
                excluded_word_pairs.update([pair, reverse_pair])
        return selected_entries, excluded_word_pairs, excluded_sentences

    def create_train_dev_data(self, data, limits, test_word_pairs, test_sentences):
        sentence_LIMIT = 100
        entry_counter = {label: 0 for label in limits}
        selected_entries_train_dev = []
        selected_entries_train = []
        selected_entries_dev = []

        sentence_count = defaultdict(int)
        # select data for train and dev
        for entry in data:
            s1, l1, s2, l2 = entry[0][0], entry[0][2], entry[1][0], entry[1][2]
            pair, reverse_pair = l1 + '__' + l2, l2 + '__' + l1
            label = entry[2]

            if (all(item not in test_word_pairs for item in (pair, reverse_pair))
                    and s1 not in test_sentences and s2 not in test_sentences
                    and sentence_count[s1] < sentence_LIMIT and sentence_count[s2] < sentence_LIMIT
            ):
                sentence_count[s1] += 1
                sentence_count[s2] += 1
                selected_entries_train_dev.append(entry)
        # split dev and train data
        for entry in selected_entries_train_dev:
            label = entry[2]
            if entry_counter[label] < limits[label]:
                entry_counter[label] += 1
                selected_entries_dev.append(entry)
            else:
                selected_entries_train.append(entry)
        ## balace the intances of the training set
        selected_entries_train = self.balance_train_data(selected_entries_train)
        return selected_entries_dev, selected_entries_train

    def balance_train_data(self, data):
        """ Balance the data by ensuring equal representation of each category in the data. """

        # Group the data points by their label
        CATEGORIES = ['term_idn', 'abb', 'syn', 'label_sim']
        cat_dict = {label: [] for label in (f"{category}_{suffix}" for category in CATEGORIES for suffix in ['pos', 'neg'])}
        for entry in data:
            label = entry[2]
            cat_dict[label].append(entry)
        balanced_train = []
        # Balance the data for each category
        for category in CATEGORIES:
            pos_label = f'{category}_pos'
            neg_label = f'{category}_neg'
            balance_num = min(len(cat_dict[pos_label]), len(cat_dict[neg_label]))

            balanced_train.extend(cat_dict[pos_label][:balance_num])
            balanced_train.extend(cat_dict[neg_label][:balance_num])

        return balanced_train

class BioWiC_format:

    def __init__(self, source_file, train_tuples, dev_tuples, test_tuples):
        self.source_file = source_file
        self.train_tuples = train_tuples
        self.dev_tuples = dev_tuples
        self.test_tuples = test_tuples

    def build_sentence_dict (self,source_file ):
        sen_data = JSONFileHandler.load_json_file(self.source_file)
        sentence_dict = {}
        for entry in sen_data:
            num = entry.get('num', "")
            sentence_dict[num] = {
                'sen': entry['sen'],
                'start': entry['annotation']['start'],
                'end': entry['annotation']['end'],
                'code': entry['annotation']['codes'],
                'source': entry['source']
            }
        return sentence_dict

    def convert_to_BioWiC_format(self, split, sentence_dict):
        """
        Convert the given tuple pairs to BioWiC structure.
        """
        new_format = []
        for instance in split:

            first, sec, detailed_label = instance
            label = 0 if '_neg' in detailed_label else 1
            # Extract and format data from the given instance based on the sentence dictionary.
            t_w1, t_l1 = first[1], first[2]
            t_sen1 = sentence_dict[first[0]]['sen']
            char_start1 = sentence_dict[first[0]]['start']
            char_end1 = char_start1 + len(t_w1)
            code1 = sentence_dict[first[0]]['code']

            t_w2, t_l2 = sec[1], sec[2]
            t_sen2 = sentence_dict[sec[0]]['sen']
            char_start2 = sentence_dict[sec[0]]['start']
            char_end2 = char_start2 + len(t_w2)
            code2 = sentence_dict[sec[0]]['code']

            CAT_dict = {
                'term_idn_pos': 'term_identity',  'term_idn_neg': 'term_identity',
                'syn_pos': 'synonyms', 'syn_neg': 'synonyms',
                'abb_pos': 'abbreviations', 'abb_neg': 'abbreviations',
                'label_sim_pos': 'label_similarity', 'label_sim_neg': 'label_similarity',
            }
            detailed_label = CAT_dict[detailed_label]

            new_instance = {
                'term1': t_w1, 'term2': t_w2,
                'sentence1': t_sen1, 'sentence2': t_sen2,
                'start1': char_start1, 'start2': char_start2,
                'end1': char_end1, 'end2': char_end2,
                'label': label, 'cat': detailed_label
            }

            new_format.append(new_instance)
        return new_format

    def build_split(self):
        sentence_dict = self.build_sentence_dict(self.source_file)
        train_split = self.convert_to_BioWiC_format(self.train_tuples, sentence_dict)
        dev_split = self.convert_to_BioWiC_format(self.dev_tuples, sentence_dict)
        test_split = self.convert_to_BioWiC_format(self.test_tuples, sentence_dict)

        return train_split, dev_split, test_split


class JSONFileHandler:

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path

    def process_initial_tuples(self):
        json_files = JSONFileHandler.list_json_files(self.data_dir_path)

        all_data = []
        for file in json_files:
            data = JSONFileHandler.load_json_file(file)
            if data:
                all_data.extend(data)
        return all_data


    @staticmethod
    def load_json_file(filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
        except Exception as e:
            print(f"An error occurred while reading {filepath}. Error: {e}")
        return None

    @staticmethod
    def save_to_json(filepath, data):
        """
        Save data to a JSON file.
        """
        # Create the directory if it does not exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}. Error: {e}")
                return  # Exit the function as the directory could not be created
        # Save the data as a JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving to {filepath}. Error: {e}")
    @staticmethod
    def list_json_files(directory_path):
        return [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.json')]


class DataManager:
    def __init__(self, tuples_data_dir):
        self.tuples_data_dir = tuples_data_dir
        self.data_processor = JSONFileHandler(self.tuples_data_dir)

    def create_splits(self):
        CATEGORIES = ['term_idn', 'abb', 'syn', 'label_sim']
        labels = ('pos', 'neg')
        all_data = self.data_processor.process_initial_tuples()
        data_splitter = DataSplitter(CATEGORIES, labels)
        proportions, test_limit_structure, dev_limit_structure = data_splitter.compute_proportion(all_data)
        random.shuffle(all_data)
        return data_splitter.split_data(all_data, test_limit_structure, dev_limit_structure)


def main(args):
    data_manager = DataManager(args.tuples_data_dir)
    train_instances, dev_instances, test_instances = data_manager.create_splits()

    biowic_builder = BioWiC_format(args.unified_el_sentences, train_instances, dev_instances, test_instances)
    train_split, dev_split, test_split = biowic_builder.build_split()

    JSONFileHandler.save_to_json(f'{args.output_dir}/train.json', train_split)
    JSONFileHandler.save_to_json(f'{args.output_dir}/dev.json', dev_split)
    JSONFileHandler.save_to_json(f'{args.output_dir}/test.json', test_split)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tuples_data_dir', type=str, default='./tuples/')
    parser.add_argument('--unified_el_sentences', type=str, default='./unified_el_resources/el_resources_standardized.json')
    parser.add_argument('--output_dir', type=str, default='./BioWiC/')
    args = parser.parse_args()
    main(args)


