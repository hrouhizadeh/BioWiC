# BioWiC construction

The folder contains scripts for assembling the BioWiC dataset.
Before beginning, confirm you've run the scripts in the UMLS folder and stored the outputs in the `umls_files` folder.
Required files are: `umls_all_term.json`, `omim_map.json`, `mesh_map.json`, `umls_term_cui_indexing.json`, `umls_hypernyms.json`,
and `umls_synonyms_codes.json`. 
Moreover, for constructing BioWiC, scripts leverage three English entity linking datasets: Medmentions, NCBI Disease, and BC5CDR, located in the `entity_linking_resources` folder.
The construction process of the BioWiC dataset involves three following steps. Make sure to execute the steps in the specified sequence, as each step relies on the output generated by the preceding one.


**Sentence extraction**

Use this script to segment the sentences in the entity linking datasets, map MeSH and OMIM codes to UMLS, and randomly choose one mention per sentence linked to UMLS, if available.
The result, `el_resources_standardized.json`, will be saved in the `unified_el_resources` folder.

```bash
python  src/step1_sentence_extraction.py
   ```

**Tuple creation**

Execute this command to generate all possible tuple combinations, each serving as the foundation for creating a BioWiC instance. 
The execution of this script will result in the creation of four distinct files: `term_identity.json`, `abbreviations.json`, `synonyms.json` and `label_similarity.json`. 
These files will be saved in the `tuples` directory.

```bash
python  src/step2_tuple_creation.py
   ```


**BioWiC construction**

This process will create the training, validation, and test sets of BioWiC, which are saved in the `BioWiC` folder.
   ```bash
python  src/step3_creating_biowic_splits.py
   ```





