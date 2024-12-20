# Coreference-based Ontology Building

This repository is the implementation of the paper "Data-driven Coreference-based Ontology Building".
It focuses on constructing biomedical ontology automatically by analyzing coreference resolution graph and applying heuristics.

## Table of Contents

- [Demo](#Demo)
- [Quick Start](#quick-start)
- [Citation](#citation)

## Demo

The ontology is available in the web demo:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/biu-nlp/Data-driven_Coreference-based_Ontology)


## Quick Start

1. Install the repository:
```
git clone git@github.com:ShirApp/Coreference-based-Ontology-Building.git
cd Coreference-based-Ontology-Building
```

2. Create a virtual environment and activate it:
```
python -m venv coref-venv
source coref-venv/bin/activate
```

3. Install the requirements:
```
pip install -r requirements.txt
```

4. Run Coreference Resolution algorithm (we recommend [fastcoref](https://github.com/shon-otmazgin/fastcoref)) on your corpus.

5. Run the "create_clean_nodes" script with the path to your coreference chains file:
```
python create_clean_nodes.py --file_path coref_chains.csv
```
The script will create a folder named "data" that contains the following files:
   - "weighted_pairs.csv" - The pairs of clean phrases occurrences with their frequency.
   - "phrase2clean.pkl" - A dict of the clean phrases and their original occurrences.

** The input to the script is a file where each line contains a coreference chain and its phrases are seperated by TAB.

6. Define the following path (in the jupyter notebook) and run the notebook:
   - data_path - path where the data files from the step above are located.

## Citation

```
@misc{ashurytahan2024datadrivencoreferencebasedontologybuilding,
      title={Data-driven Coreference-based Ontology Building}, 
      author={Shir Ashury-Tahan and Amir David Nissan Cohen and Nadav Cohen and Yoram Louzoun and Yoav Goldberg},
      year={2024},
      eprint={2410.17051},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.17051}, 
}
```
