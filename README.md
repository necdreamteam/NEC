#NEC 

"Interpretable Prediction of Necrotizing Enterocolitis from Machine Learning Analysis of Premature Infant Stool Microbiota", by Yun Chao Lin, Ansaf Salleb-Aouissi and Thomas A. Hooven

## Setup
This code uses Python 3.7 virtual environment can be use with requirement.txt. 

```mkdir venv
python3 -m venv ~/venv/nec
source ~/venv/nec/bin/activate
pip install -r requirements.txt
```

## Running the model
The main script are `src/main.py`. All script codes are stored in `src`

Example of usage:
`python src/main.py -d <dataset_path> -ne <number_of_epochs> -lr <learning_rate>`

The main.py runs the MIL model and produces the ROC and Precision and Recall Curves. 

## Karken2 and Hierarchical Feature Engineering

Data is preprocessed using Karken2 (<https://ccb.jhu.edu/software/kraken2/>) and HFE (<https://github.com/HenschelLab/HierarchicalFeatureEngineering>). Codes are available on link provided. 

## Dataset and Target Label 
Each sample must contain a SUBJECTID. A patient can be represented by multiple samples with same SUBJECTID. Target label is whether a patient has NEC or not. A sample with same SUBJECTID has the same target label. 
 
## Model
Model are implemented in pytorch which originates from (<https://github.com/AMLab-Amsterdam/AttentionDeepMIL>). Codes are stored in `src/model`.


