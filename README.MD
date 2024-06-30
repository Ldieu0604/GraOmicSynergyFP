# install requirements
chmod +x install.sh

./install.sh

# Resources:
+ README.md: this file.
+ data: GDSC dataset
+ download gen expression from this following link: https://drive.google.com/file/d/1orMkX_4xC6N8fah98nUFJSuZueHOadX4/view?usp=sharing

### Source codes:
+ split_process_data.ipynb: This notebook is used to split the data into training, validation, and test sets. It employs the Kolmogorov-Smirnov statistical test (Zwillinger and Kokoska, 1999) to validate the synergy distribution between each set.
+ utils.py: This file includes the TestbedDataset, which is used by create_data.py to generate data. It also contains performance measures and functions to plot the loss and Pearson correlation coefficient by epoch.
+ models/gin_add_at_cls.ipynb, gin_add_at_rmse.ipynb: These notebooks present the proposed models, namely GINConvNet, for drug classification and regression prediction tasks. The models receive graphs as input, and the notebooks provide details about the training process.


## Dependencies
+ [Tensorflow](https://www.tensorflow.org/)
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Rdkit](https://www.rdkit.org/)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)

# Step-by-step running:

## 1. Process and split data
Run process_split_data.ipynb.

## 2. Prepair Pytorch format data
Run create_data.ipynb

This will generate a file in PyTorch format (.pt) stored at data/all_test/processed. The file will include the training, validation, and test sets. It converts the drug information into a graph and converts the gene, mutation, and methylation information into a tensor.

## 3. Train a GraphDRP model
Train regression task with GIN using gin_add_at_rmse.ipynb.

Train classification task with GIN using gin_add_at_cls.ipynb.

## 4. Run Deepsynergy and DeepDDS experiment
Deepsynergy: models/DeepSynergy/cv_example/deepsynergy.py

DeepDDS: Deepdds.ipynb
