# XAI Project 1 
## Project Module: Models that Exaplin Themselves

# Task Data

The below link provide direct access to the task the task dataset.
- [Train data](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Train.csv)
- [Test data](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Test.csv)
- [Test data w/ ground-truth labels](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Test-GS.csv)

# Training and Testing Setup(Linux SetUp)

1. Clone the repository

```git clone [git clone https URL]```

2. Create a Python virtual environment

```
# Update and upgrade
sudo apt update
sudo apt -y upgrade

# check for python version "ideal: 3.8.2"
python3 -V

# install python3-pip
sudo apt install -y python3-pip

# install-venv
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv my_env


# Activate virtual environment
source my_env/bin/activate
```

3. Install project dependent files

```
pip install requirements.txt
```

4. Run main.py

```
python3 main.py
```

# Project Directory Tree

```
└── Project1/
    ├── data/
    │   ├── biocreative_dataset.py
    │   ├── BC7-LitCovid-Test-GS.csv
    │   ├── BC7-LitCovid-Test.csv
    │   └── BC7-LitCovid-Train.csv
    ├── saved_models/
    │   └── 100_RFC.pkl
    ├── config.yaml
    ├── Decision_Tree_XAI.ipynb
    ├── main.py
    ├── models.py
    ├── utils.py
    ├── XAI.py
    └── requirements.txt
```

# NOTE

```
If there are any dependency issues related to SHAP or LIME not compatibile on local environment try using the .ipynb notebook instead. 
```
## Tree Algorithms
overall performance and the explainability aspect (supported with visualizations) of the model predictions.
### Tasks:
1.	Choose a dataset of your choice (language, vision, medical, ...)
2.	Implement a model that performs on the selected task of the dataset
3.	Perform hyper-parameter search to find the optimal model configuration(s)
4.	Perform global and local feature importance analysis
5.	Generate visualizations with selected some examples (3-5 correct and 3-5 failed failed samples) - think about adding counter factual examples and show the performance change
6.	Present all the findings (trees, features, examples, etc.) by showing the explainability aspect of the learned model
