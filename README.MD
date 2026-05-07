# GraOmicSynergy

## Setup

Use Python 3.13 and install the project dependencies from `requirements.txt`.
Also for additional dependencies run install.sh
## Resources

+ `data/`: GDSC dataset and processed splits
+ `process_split_data.ipynb`: splits the data into training, validation, and test sets
+ `create_data.ipynb`: builds the standard 2D PyTorch-format dataset
+ `utils.py`: utilities for the standard pipeline
+ `models/gin_add_at_cls.ipynb`, `models/gin_add_at_rmse.ipynb`: standard classification and regression training notebooks

Gene expression data can be downloaded from:
https://drive.google.com/file/d/1orMkX_4xC6N8fah98nUFJSuZueHOadX4/view?usp=sharing

## Workflow

### 1. Process and split the data

Run `process_split_data.ipynb` first.

This step only needs to be run once. It prepares the train/validation/test splits reused by the standard pipeline.

### 2. Create the dataset

After `process_split_data.ipynb`, run the standard data creation notebook:

+ `create_data.ipynb`

The generated files are in `.pkl` format and include the training, validation, and test sets, with drug information converted into graphs and Gene expression, Mutation and Copy number alteration, and Methylation information converted.

### 3. Train the models

Standard notebooks:

+ Regression: `models/gin_add_at_rmse.ipynb`
+ Classification: `models/gin_add_at_cls.ipynb`

## Notes

The explanations for the runtime improvements are documented inside the training notebooks in the `models/` directory.

These notebooks present the proposed models, namely GINConvNet, for drug combination classification and regression prediction tasks
