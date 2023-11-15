# Extract UMLS information

This directory is consist of scripts that preprocess Unified Medical Language System (UMLS) data.
Note that UMLS raw data are not included in this repository. For downloading UMLS data, follow the instructions available at this [link](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html).
Once you have obtained the data, ensure that both MRCONSO.RRF and MRREL.RRF are located in the same directory as the scripts in this folder.
The scripts should be executed sequentially as they may depend on the outputs from preceding scripts.

**Note on Script Outputs**

By default, the outputs from these scripts will be saved in the `umls_files` directory. 
If you modify the output file names or the directory, ensure to adjust the arguments accordingly when running BioWiC construction codes in the [biowic_construction](https://github.com/hrouhizadeh/BioWiC/tree/main/BioWiC_construction) directory.



**Constructing UMLS indexing file**

To create an index file containing UMLS information, execute the script below. The output, `umls_synonyms_codes.json`, will be stored as `umls_files` directory.
   ```bash
python  umls_preprocess_all_term.py
   ```


**Extracting UMLS synonym CUIs**

For extracting synonymous Concept Unique Identifiers (CUIs) from UMLS, use the following scrip. The resulting file, `umls_synonyms_codes.json`, will be created.

   ```bash
python extract_synonym_cuis.py 
   ```

**Mapping MeSH and OMIM to UMLS**

Execute this command to map MeSH and OMIM codes to UMLS. It will generate two files:`mesh_map.json` and `omim_map.json`
   ```bash
python extract_umls_mappings_from_mesh_omim.py
   ```

**Extracting CUI pairs with hypernym-hyponym relationship**

This script extracts CUI pairs where one term is a hypernym of the other. The output file will be saved as `umls_hypernyms.json`.
   ```bash
python extract_cui_hypernyms.py 
   ```

**Extracting all CUIs per concept**

Use this script to compile all possible CUIs for each UMLS concept. This will produce the file `umls_term_cui_indexing.json`. 

```bash
python extract_all_cuis_per_umls_concept.py  
```


