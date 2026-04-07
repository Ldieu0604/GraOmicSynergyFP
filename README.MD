# GraOmicSynergy

## Setup

Use Python `3.12.3` and install the project dependencies from `requirements.txt`.

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Resources

+ `data/`: GDSC dataset and processed splits
+ `process_split_data.ipynb`: splits the data into training, validation, and test sets
+ `create_data.ipynb`: builds the standard 2D PyTorch-format dataset
+ `create_data_with_3d.ipynb`: builds the 2D + 3D PyTorch-format dataset
+ `utils.py`: utilities for the standard pipeline
+ `utils_with_3d.py`: utilities for the 2D + 3D pipeline
+ `models/gin_add_at_cls.ipynb`, `models/gin_add_at_rmse.ipynb`: standard classification and regression training notebooks
+ `models/gin_add_at_cls_with_3d.ipynb`, `models/gin_add_at_rmse_with_3d.ipynb`: classification and regression training notebooks using the 3D-enhanced input

Gene expression data can be downloaded from:
https://drive.google.com/file/d/1orMkX_4xC6N8fah98nUFJSuZueHOadX4/view?usp=sharing

## Workflow

### 1. Process and split the data

Run `process_split_data.ipynb` first.

This step only needs to be run once. It prepares the train/validation/test splits that are reused by both the standard pipeline and the 3D pipeline.

### 2. Choose a data creation path

After `process_split_data.ipynb`, you can continue with either of these paths:

+ Standard path: run `create_data.ipynb`, then train with the regular GIN notebooks in `models/`
+ 3D path: run `create_data_with_3d.ipynb`, then train with the `*_with_3d.ipynb` notebooks in `models/`

The 3D implementation is a simple concatenation of the 3D features with the existing 2D representation, so it uses the same processed split produced by `process_split_data.ipynb`. Note that the files created by the create_data notebooks are in .pkl format, which  will include the training, validation, and test sets, with drug information converted into graphs and Gene expression, Mutation and Copy number alteration, and Methylation information converted

### 3. Train the models

Standard notebooks:

+ Regression: `models/gin_add_at_rmse.ipynb`
+ Classification: `models/gin_add_at_cls.ipynb`

3D notebooks:

+ Regression: `models/gin_add_at_rmse_with_3d.ipynb`
+ Classification: `models/gin_add_at_cls_with_3d.ipynb`

## Notes

The explanations for the runtime improvements are documented inside the training notebooks in the `models/` directory.

These notebooks present the proposed models, namely GINConvNet, for drug combination classification and regression prediction tasks
