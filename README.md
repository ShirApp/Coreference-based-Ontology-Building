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

4. Define the following things (in the jupyter notebook) and run the notebook:
  - weighted_graph_clean: CSV file containing phrases relations along with weights (columns: "node1", "node2", "weight")
  - clean2phrases: JSON file mapping clean phrases to their original forms and occurrences. This file should be created as a part of the cleaning.
  - data_path: Path where the above data files are located.

## Citation