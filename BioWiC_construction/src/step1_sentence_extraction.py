import json
import random
import spacy
import pysbd
import sys
import os
import argparse

nlp = spacy.load('en_core_web_sm')
seg = pysbd.Segmenter(language="en", clean=False)
random.seed(42)

class FileManager:
    @staticmethod
    def read_json_file(filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading {filepath}. Error: {e}")
            return None
    @staticmethod
    def read_jsonl_file(filepath):
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
    def save_to_directory(filename, content, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            json.dump(content, file)

    @staticmethod
    def change_jsonl_format(initial_mapping):
        new_mapping = {}
        for idx in range(len(initial_mapping)):
            cui_code = initial_mapping[idx]['CUI']
            new_mapping[cui_code] = initial_mapping[idx]

        return new_mapping

class DocumentPreprocessor:
    TITLE_MARKER = '|t|'
    ABSTRACT_MARKER = '|a|'

    def __init__(self, filepath, source):
        self.filepath = filepath
        self.source = source

    def process_initial_documents(self):

        annotations = []
        documents = []
        current_annotations = []
        current_document = ''

        with open(self.filepath, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    if self.TITLE_MARKER in line:
                        title = line.split(self.TITLE_MARKER)[1]
                        current_document += title
                    elif self.ABSTRACT_MARKER in line:
                        abstract = line.split(self.ABSTRACT_MARKER)[1]
                        current_document += abstract
                    else:
                        annotation = self._parse_annotation_line(line, self.source)
                        if annotation:
                            current_annotations.append(annotation)
                else:
                    if current_document or current_annotations:
                        annotations.append(current_annotations)
                        documents.append(current_document)
                        current_annotations = []
                        current_document = ''

        final_docs_flat = self.make_flat(documents, annotations)
        return final_docs_flat

    def _parse_annotation_line(self, line, source):

      if source == 'NCBI':
        parts = line.strip().split('\t')[:6]
        if '|' not in parts[5]:
            parts[:3] = map(int, parts[:3])
            parts.append(parts[3].replace('.', '*'))
            parts.append(self.source)
            return parts

      elif source == 'BC5CDR':
        parts = line.strip().split('\t')[:6]
        if parts[1] != 'CID' and '|' not in parts[5]:
            parts[:3] = map(int, parts[:3])
            parts.append(parts[3].replace('.', '*'))
            parts.append(source)
            return parts

      elif source == 'Medmentions':
        parts = line.strip().split('\t')
        parts[:3] = map(int, parts[:3])
        parts.append(parts[3].replace('.', '*'))
        parts.append(self.source)
        return parts

    @staticmethod
    def make_flat(documents, annotations):
        all_info = []
        for doc_id, (document, doc_annotations) in enumerate(zip(documents, annotations)):
            processed_document = AnnotationHandler.annotation_replacement(document, doc_annotations)
            sentences = DocumentSegmenter.segment_document(processed_document)
            sentence_annotations = AnnotationHandler.extract_sentence_annotations(processed_document, doc_annotations,
                                                                                  sentences)
            all_info.append(sentence_annotations)
        flat_list = [item for sublist in all_info for item in sublist]
        return flat_list


class AnnotationHandler:
    @staticmethod
    def annotation_replacement(document, annotations):
        for annotation in annotations:
            if annotation[3] != annotation[6]:
                start = int(annotation[1])
                end = int(annotation[2])
                document = document[:start] + annotation[6] + document[end:]
        return document

    @staticmethod
    def extract_sentence_annotations(document, annotations, sentences):
        sentence_level_annotations = []
        for sen in sentences:
            start_index_of_sentence = document.find(sen)
            end_index_of_sentence = start_index_of_sentence + len(sen)
            sentence_annotations = [
                annotation for annotation in annotations
                if AnnotationHandler.is_annotation_within_boundaries(annotation, start_index_of_sentence,
                                                                     end_index_of_sentence)
            ]
            for annotation in sentence_annotations:
                annotation[1] -= start_index_of_sentence
                annotation[2] -= start_index_of_sentence

            if sentence_annotations:
                sentence_level_annotations.append({'sen': sen, 'annotations': sentence_annotations})

        return sentence_level_annotations

    @staticmethod
    def is_annotation_within_boundaries(annotation, start_idx, end_idx):
        start = int(annotation[1])
        end = int(annotation[2])
        return start_idx <= start and end <= end_idx


class Lemmatizer:
    @staticmethod
    def get_lemma (entity):
        doc = nlp(entity)
        lemma = [token.lemma_ for token in doc]
        lemma = ' '.join(lemma)
        return lemma


    
class DocumentSegmenter:
    @staticmethod
    def segment_document(document):
        sentences = seg.segment(document)
        return sentences

class FormattingDocuments:

    def __init__(self, data, source, umls_mapping_data, mesh_map_dict, omim_map_dict):
        self.data = data
        self.source = source
        self.umls_mapping_data = umls_mapping_data
        self.mesh_map_dict = mesh_map_dict
        self.omim_map_dict = omim_map_dict

    def segment_and_map_sentences (self):
        mapped_data = []
        total_sentence_count = 0

        for entry in self.data:

            sentence = entry['sen']
            annotations = entry['annotations']

            if self.source == 'BC5CDR':
                umls_mapped_annotations = self.map_annotations_BC5CDR(annotations,self.mesh_map_dict, self.umls_mapping_data)
            elif self.source == 'NCBI':
                umls_mapped_annotations = self.map_annotations_NCBI(annotations,self.mesh_map_dict,self.omim_map_dict, self.umls_mapping_data)
            elif self.source == 'Medmentions':
                umls_mapped_annotations = self.map_annotations_medmentions(annotations, self.umls_mapping_data)

            # Filter to keep only the sentences with at-least one annotation
            valid_annotations = [annotation for annotation in umls_mapped_annotations
                                 if len(annotation) > 0 and annotation[5] in self.umls_mapping_data]
            if valid_annotations:
                chosen_annotation = random.choice(valid_annotations)
                real_entity = chosen_annotation[3]
                placeholder_entity = chosen_annotation[6]
                sentence = sentence.replace(placeholder_entity, real_entity)
                sentence = self.remove_start_in_annotation(annotations,sentence)
                lemma = Lemmatizer.get_lemma(real_entity)
                instance = {
                    'num': total_sentence_count,
                    'sen': sentence,
                    'annotation': {
                    'entity': chosen_annotation[3],
                    'lemma' : lemma,
                    'start': chosen_annotation[1],
                    'end': chosen_annotation[2],
                    'codes': [chosen_annotation[5]],
                    'type': chosen_annotation[4]
                    },
                    'source': self.source
                }
                total_sentence_count +=1
                mapped_data.append(instance)
        return mapped_data

    def map_annotations_BC5CDR(self, annotations, mesh_map_dict, umls_mapping_data):
        INDEX = 5
        mapped_annotations = []
        for annotation in annotations:
            original_code = annotation[INDEX]
            # If code is in MeSH mapping and has only one mapping
            if original_code in mesh_map_dict and len(mesh_map_dict[original_code]) == 1:
                annotation[INDEX] = mesh_map_dict[original_code][0]
                if annotation[INDEX] in umls_mapping_data:
                    mapped_annotations.append(annotation)
        return mapped_annotations

    def map_annotations_medmentions(self, annotations, umls_mapping_data):
        UMLS_PREFIX = 'UMLS:'
        INDEX = 5
        new_annotations = []
        for annotation in annotations:
            annotation[INDEX] = annotation[INDEX].replace(UMLS_PREFIX, '')
            if annotation[INDEX] in umls_mapping_data:
                new_annotations.append(annotation)
        return new_annotations

    def map_annotations_NCBI (self, annotations, mesh_map_dict, omim_map_dict, umls_mapping_data):

        INDEX = 5
        OMIM_PREFIX = 'OMIM:'
        mapped_annotations = []
        for annotation in annotations:
            original_code = annotation[INDEX]
            # Check and replace based on OMIM mapping
            if OMIM_PREFIX in original_code:
                original_code = original_code.replace(OMIM_PREFIX, '')
                mapped_code = omim_map_dict.get(original_code)
                if mapped_code and len(mapped_code) == 1:
                    annotation[INDEX] = mapped_code[0]
            # Directly check in MeSH mapping if not OMIM
            elif original_code in mesh_map_dict:
                mapped_code = mesh_map_dict.get(original_code)
                if mapped_code and len(mapped_code) == 1:
                    annotation[INDEX] = mapped_code[0]
            # Append if mapped code is in UMLS data
            if annotation[INDEX] in umls_mapping_data:
                mapped_annotations.append(annotation)
        return mapped_annotations

    @staticmethod
    def remove_start_in_annotation(annotations, sentence):
        for annotation in annotations:
            if annotation[3] != annotation[6]:
                start = int(annotation[1])
                end = int(annotation[2])
                sentence = sentence.replace(annotation[6], annotation[3])
        return sentence


def run_(annotation_paths, sources, umls_mapping_data, mesh_map_dict, omim_map_dict, output_dir):
    all_three_sources = []
    for idx, filepath in enumerate(annotation_paths):
        document_preprocessor = DocumentPreprocessor(filepath, sources[idx])
        pre_processed_documents = document_preprocessor.process_initial_documents()
        formatting_documents = FormattingDocuments(pre_processed_documents, sources[idx], umls_mapping_data, mesh_map_dict, omim_map_dict)
        formatted_docs = formatting_documents.segment_and_map_sentences()
        all_three_sources.append(formatted_docs)
        # here you can save the content of each file separately
        # FileManager.save_to_directory(sources[idx]+'_standardized.json', formatted_docs, output_dir)

    flated_all_three_sources = [item for sublist in all_three_sources for item in sublist]
    for counter, instance in enumerate(flated_all_three_sources, 0):
        instance['num'] = counter
    FileManager.save_to_directory('el_resources_standardized.json', flated_all_three_sources, output_dir)
    return None

def main():

    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument('--umls_mappings', type=str, default = '../UMLS/umls_files/umls_all_term.json')

    parser.add_argument('--omim_file', type=str, default='../UMLS/umls_files/omim_map.json')
    parser.add_argument('--mesh_file', type=str, default='../UMLS/umls_files/mesh_map.json')
    parser.add_argument('--output_dir', type=str, default='./unified_el_resources/')
    parser.add_argument('--entity_linking_files_path', type=str, nargs='+',
                        default=['./entity_linking_source_files/NCBI.txt', './entity_linking_source_files/BC5CDR.txt', './entity_linking_source_files/st21pv.txt'],
                        )
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['NCBI', 'BC5CDR', 'Medmentions'])

    args = parser.parse_args()

    umls_mapping_data = FileManager.read_jsonl_file(args.umls_mappings)
    omim_map_dict = FileManager.read_json_file(args.omim_file)
    mesh_map_dict = FileManager.read_json_file(args.mesh_file)
    umls_mapping_data = FileManager.change_jsonl_format(umls_mapping_data)
    output_dir = args.output_dir
    annotation_filepath = args.entity_linking_files_path
    sources = args.sources

    run_(annotation_filepath, sources, umls_mapping_data, mesh_map_dict, omim_map_dict, output_dir)

if __name__ == "__main__":
    main()

