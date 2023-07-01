# Utility Scripts

---

There is one utility script -- `compile_ontologies.py`.
This script reads all files from the `../data/ontologies` directory.
It assumes that every file contains the words for an ontology class,
with a separate word on every line. It then combines all files 
together in a single file which can be used by the deep learning manager.
The output file is generated in
`../dl_manager/feature_generators/util/ontologies.json`