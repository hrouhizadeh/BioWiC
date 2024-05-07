
# A Dataset for Evaluating Contextualized Representation of Biomedical Concepts in Language Models

## Introduction
In this paper, we present the BioWiC benchmark, a new dataset designed to assess how well language models represent biomedical concepts according to their corresponding context.
BioWiC is formulated as a binary classification task where each instance involves a pair of biomedical terms along with their corresponding sentences. 
The task is to classify each instance as True if the target terms carry the same meaning across both sentences or False if they do not.

For further details refer to the article [BioWiC paper](https://www.nature.com/articles/s41597-024-03317-w).


## Installation
Follow these instructions to install the necessary dependencies for the project
```bash
git clone https://github.com/hrouhizadeh/BioWiC
cd BioWiC
pip install -r requirements.txt
```

## Reconstruct BioWiC 
To reproduce the construction of BioWiC, you need to perform the following steps:
1. **Extracting UMLS information**: In the [UMLS](https://github.com/hrouhizadeh/BioWiC/tree/main/UMLS) directory, detailed steps are provided to extract UMLS information needed for the BioWiC dataset development.
2. **Building BioWiC dataset**:  Follow the instructions in the [BioWiC_construction](https://github.com/hrouhizadeh/BioWiC/tree/main/BioWiC_construction) directory to reconstruct the BioWiC dataset.
3. **Train and evaluate models**: The [models](https://github.com/hrouhizadeh/BioWiC/tree/main/models) folder contains scripts that enable you to train and test various discriminative and generative large language models using the BioWiC dataset.

Additionally, the official release of the BioWiC dataset is available for direct download in the [data](https://github.com/hrouhizadeh/BioWiC/tree/main/data) folder.


<a name="hugging-face"></a>
## Accessing BioWiC on Hugging Face 🤗

1. **Install the Hugging Face `datasets` Library:**
   If not already installed, you can add the `datasets` library from Hugging Face.
   ```bash
   pip install datasets
   ```

2. **Load the BioWiC Dataset:**
   To load the BioWiC dataset, execute the following Python code.
   ```python
   from datasets import load_dataset

   dataset = load_dataset("hrouhizadeh/BioWiC")
   ```
   This command will automatically download and cache the dataset.

## Citation
If you use the BioWiC dataset in your research, please cite the following paper:
   ```bash
@article{rouhizadeh2024dataset,
  title={A Dataset for Evaluating Contextualized Representation of Biomedical Concepts in Language Models},
  author={Rouhizadeh, Hossein and Nikishina, Irina and Yazdani, Anthony and Bornet, Alban and Zhang, Boya and Ehrsam, Julien and Gaudet-Blavignac, Christophe and Naderi, Nona and Teodoro, Douglas},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={1--13},
  year={2024},
  publisher={Nature Publishing Group}
}
```


## Contact
Should you have any inquiries, feel free to contact us at _hossein.rouhizadeh@unige.ch_.
