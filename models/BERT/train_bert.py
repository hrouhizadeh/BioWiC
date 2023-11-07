import os
import json
import logging
import argparse

from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


def add_quotes(input_data):

    """ put quotations around the target terms"""

    sen1_start = input_data['start1']
    sen1_end = input_data['end1']
    sen2_start = input_data['start2']
    sen2_end = input_data['end2']

    sen1_with_quotes = input_data['sentence1'][:sen1_start] + "\"" + input_data['sentence1'][sen1_start:sen1_end] + "\"" + input_data['sentence1'][sen1_end:]
    sen2_with_quotes = input_data['sentence2'][:sen2_start] + "\"" + input_data['sentence2'][sen2_start:sen2_end] + "\"" + input_data['sentence2'][sen2_end:]

    return sen1_with_quotes, sen2_with_quotes


def main (args):



    ## load the model
    model = SentenceTransformer(args.model_name)
    os.makedirs(args.model_save_path, exist_ok=True)
    
    ######### Read train data  ##########
    # Read train data
    train_samples = []
    with open(args.dataset_path + 'train.json', 'r', encoding='utf-8') as f_train:
        data_train = json.load(f_train)
        for input_ins in data_train:
            sen1, sen2 = add_quotes(input_ins)
            train_samples.append(InputExample(texts=[sen1, sen2], label=input_ins['label']))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, margin=args.margin)

    ###### Classification ######
    # Given (sentence1_term1, sentence2_term2), do term1 and term2 have the same meaning?
    # The evaluator will compute the embeddings for both sentences and then compute
    # a cosine similarity. If the similarity is above a threshold, terms have the same meaning.
    evaluators = []
    dev_sentences1 = []
    dev_sentences2 = []
    dev_labels = []
    with open(args.dataset_path + 'dev.json', 'r', encoding='utf-8') as f_dev:
        data_dev = json.load(f_dev)
        for input_ins in data_dev:
            sen1, sen2 = add_quotes(input_ins)
            dev_sentences1.append(sen1)
            dev_sentences2.append(sen2)
            dev_labels.append(input_ins['label'])


    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
    evaluators.append(binary_acc_evaluator)

    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    logger.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=args.model_save_path)


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=seq_evaluator,
              epochs=args.num_epochs,
              warmup_steps=1000,
              output_path=args.model_save_path
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on the test set dataset
    #
    ##############################################################################

    test_samples = []
    with open(args.dataset_path + 'test.json', 'r', encoding='utf-8') as f_test:
        data_test = json.load(f_test)
        for input_ins in data_test:
            sen1, sen2 = add_quotes(input_ins)
            test_samples.append(InputExample(texts=[sen1, sen2], label=input_ins['label']))

    model = SentenceTransformer(args.model_save_path)
    test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, batch_size=args.train_batch_size, name='sts-test')

    test_evaluator(model, output_path=args.model_save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--dataset_path", default="./BioWiC/", type=str)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--model_save_path", default="./BioWiC_model/", type=str)
    parser.add_argument("--margin", default=0.5, type=float)

    args = parser.parse_args()
    os.makedirs(args.model_save_path, exist_ok=True)
    main(args)



##############################################################################
# The code is mostly taken from the official repository of sentence-bert:
# https://www.sbert.net/
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/quora_duplicate_questions/training_OnlineContrastiveLoss.py
##############################################################################