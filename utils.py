import os
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

class DrugCombination(DATA.Data):
    def __init__(self, edge_index_1=None, x_1=None, edge_index_2=None, x_2=None, y=None):
        super(DrugCombination, self).__init__()
        self.edge_index_1 = edge_index_1
        self.x_1 = x_1 
        self.edge_index_2 = edge_index_2
        self.x_2 = x_2
        self.y = y
    def __inc__(self, key, value):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        else:
            return super().__inc__(key, value)
        
        
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd_1=None, xd_pt_1 =None, xd_2=None, xd_pt_2 = None, xt_mut=None, xt_meth=None, xt_ge=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,saliency_map=False):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph):
        data_list = []
        data_len = len(xt_ge)
        for i in range(data_len):
#             print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            os.system('cls')

            smiles_1 = xd_1[i]
            smiles_2 = xd_2[i]
            smiles_pt_1 = xd_pt_1[i]
            smiles_pt_2 = xd_pt_2[i]
            target_mut = xt_mut[i]
            target_meth = xt_meth[i]
            target_ge = xt_ge[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size_1, features_1, edge_index_1 = smile_graph[smiles_1]
            c_size_2, features_2, edge_index_2 = smile_graph[smiles_2]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DrugCombination(
                edge_index_1=torch.LongTensor(edge_index_1).transpose(1, 0),
                x_1=torch.Tensor(features_1),
                edge_index_2=torch.LongTensor(edge_index_2).transpose(1, 0),
                x_2=torch.Tensor(features_2),                
                y=torch.FloatTensor([labels]),
                               )
            
            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target_mut = torch.tensor([target_mut], dtype=torch.float, requires_grad=True)
                GCNData.target_meth = torch.tensor([target_meth], dtype=torch.float, requires_grad=True)
                GCNData.target_ge = torch.tensor([target_ge], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target_mut = torch.FloatTensor([target_mut])
                GCNData.target_meth = torch.FloatTensor([target_meth])
                GCNData.target_ge = torch.FloatTensor([target_ge])
                
            GCNData.xd_pt_1 = torch.FloatTensor([smiles_pt_1])
            GCNData.xd_pt_2 = torch.FloatTensor([smiles_pt_2])
            
            GCNData.__setitem__('c_size_1', torch.LongTensor([c_size_1]))
            GCNData.__setitem__('c_size_2', torch.LongTensor([c_size_2]))
            
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd

def bce(y,f):
    f = np.clip(f, 1e-7, 1 - 1e-7)
    term_0 = (1-y) * np.log(1-f + 1e-7)
    term_1 = y * np.log(f + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method
    plt.close()

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method
    plt.close()


def plot_confusion_matrix(y_true, y_pred,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=10, rotation=90)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_path)
    plt.close()