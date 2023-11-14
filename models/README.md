# BioWiC baseline models 

This directory contains codes for training and evaluating baseline models on the BioWiC dataset.


**BERT Models**

To train a BERT based model, run the following script:

   ```bash
python BERT/train_bert.py --model_name [model name] --dataset_path [BioWiC dataset path] --num_epochs [number of epochs] --train_batch_size [batch size] --model_save_path [save directory]
   ```
Evaluate the trained BERT model using:
   ```bash
python BERT/inference_bert.py --model_path [model path] --test_set_path [BioWiC test set path]
   ```

**Llama-2 Models**

For training Llama-2 models, use the following command:

   ```bash
python Llama-2/train_llama.py --model_name [Llama-2 model name] --dataset_name [training set path] --val_dataset_name [BioWiC dev set path] --num_train_epochs [number of epochs] --use_auth_token [Huggingface token]
   ```

Evaluate the Llama-2 models with:
   ```bash
python Llama-2/inference_llama.py --model_path [Llama-2 model path] --data_address [BioWiC test set path] --access_token [Huggingface token]
   ```


