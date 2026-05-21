import os
import shutil
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch_geometric import data as DATA
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

class DrugCombination(DATA.Data):
    def __init__(self, edge_index_1=None, x_1=None, edge_index_2=None, x_2=None, 
                 edge_index_f1=None, x_f1=None, edge_index_f2=None, x_f2=None, 
                 frag_atom_mapping_1=None, frag_atom_mapping_2=None, y=None):
        super(DrugCombination, self).__init__()
        
        # Đồ thị Nguyên tử
        self.edge_index_1 = edge_index_1
        self.x_1 = x_1 
        self.edge_index_2 = edge_index_2
        self.x_2 = x_2
        
        # Đồ thị Mảnh
        self.edge_index_f1 = edge_index_f1
        self.x_f1 = x_f1
        self.edge_index_f2 = edge_index_f2
        self.x_f2 = x_f2
        
        # Ánh xạ Mảnh - Nguyên tử
        self.frag_atom_mapping_1 = frag_atom_mapping_1 # Tensor [2, N_links]
        self.frag_atom_mapping_2 = frag_atom_mapping_2
        
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        # 1. Offset cho đồ thị nguyên tử (Giữ nguyên)
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
            
        # 2. Offset cho đồ thị mảnh (Giữ nguyên)
        if key == 'edge_index_f1':
            return self.x_f1.size(0)
        if key == 'edge_index_f2':
            return self.x_f2.size(0)
        
        # 3. SỬA LỖI: Offset cho ma trận ánh xạ (Bipartite Mapping)
        # Bipartite mapping có shape [2, N_links]. 
        # Hàng 0 là index mảnh (cần offset theo số mảnh), hàng 1 là index nguyên tử (offset theo số nguyên tử).
        if key == 'frag_atom_mapping_1':
            # Trả về tensor 1D: [offset_mảnh, offset_nguyên_tử]
            return torch.tensor([self.x_f1.size(0), self.x_1.size(0)])
                
        if key == 'frag_atom_mapping_2':
            # Trả về tensor 1D: [offset_mảnh, offset_nguyên_tử]
            return torch.tensor([self.x_f2.size(0), self.x_2.size(0)])
            
        return super().__inc__(key, value, *args, **kwargs)
    
        
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

torch.serialization.add_safe_globals([DrugCombination, DataEdgeAttr, DataTensorAttr, GlobalStorage])

def _row_float_tensor(row, requires_grad=False):
    tensor = torch.as_tensor(np.asarray(row, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor

class TestbedDataset(InMemoryDataset):
    # Thêm biến smile_graph_frag vào hàm khởi tạo
    def __init__(self, root='/tmp', dataset='davis', 
                 xd_1=None, xd_pt_1 =None, xd_2=None, xd_pt_2 = None, 
                 xt_mut=None, xt_meth=None, xt_ge=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, smile_graph_frag=None, saliency_map=False):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map
        
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            # Truyền thêm smile_graph_frag vào hàm process
            self.process(xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph, smile_graph_frag)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph, smile_graph_frag):
        data_list = []
        data_len = len(xt_ge)
        for i in range(data_len):
            smiles_1 = xd_1[i]
            smiles_2 = xd_2[i]
            smiles_pt_1 = xd_pt_1[i]
            smiles_pt_2 = xd_pt_2[i]
            
            target_mut = xt_mut[i]
            target_meth = xt_meth[i]
            target_ge = xt_ge[i]
            labels = y[i]
            
            # Trích xuất Đồ thị Nguyên tử
            c_size_1, features_1, edge_index_1 = smile_graph[smiles_1]
            c_size_2, features_2, edge_index_2 = smile_graph[smiles_2]
            
            # Trích xuất Đồ thị Mảnh (THÊM MỚI)
            # smile_graph_frag[smiles] phải trả về: (c_size_f, features_f, edge_index_f, frag_atom_mapping)
            # frag_atom_mapping: Tensor [2, N_edges] với hàng 0 = frag indices, hàng 1 = atom indices
            if len(smile_graph_frag[smiles_1]) == 4:
                c_size_f1, features_f1, edge_index_f1, frag_atom_mapping_1 = smile_graph_frag[smiles_1]
            else:
                # Fallback nếu không có frag_atom_mapping, tạo mapping rỗng
                c_size_f1, features_f1, edge_index_f1 = smile_graph_frag[smiles_1]
                frag_atom_mapping_1 = torch.LongTensor([[], []])  # Rỗng: [2, 0]
                
            if len(smile_graph_frag[smiles_2]) == 4:
                c_size_f2, features_f2, edge_index_f2, frag_atom_mapping_2 = smile_graph_frag[smiles_2]
            else:
                c_size_f2, features_f2, edge_index_f2 = smile_graph_frag[smiles_2]
                frag_atom_mapping_2 = torch.LongTensor([[], []])  # Rỗng: [2, 0]

            # KHẮC PHỤC: Đảm bảo frag_atom_mapping có đúng định dạng
            if not isinstance(frag_atom_mapping_1, torch.Tensor):
                frag_atom_mapping_1 = torch.LongTensor(frag_atom_mapping_1)
            if not isinstance(frag_atom_mapping_2, torch.Tensor):
                frag_atom_mapping_2 = torch.LongTensor(frag_atom_mapping_2)

            GCNData = DrugCombination(
                edge_index_1=torch.LongTensor(edge_index_1).t(), # .t() là viết tắt của .transpose(0, 1)
                x_1=torch.Tensor(features_1),
                edge_index_2=torch.LongTensor(edge_index_2).t(),
                x_2=torch.Tensor(features_2),
                
                # Đối với đồ thị mảnh:
                # Nếu edge_index_f1 CHƯA transpose, thì dùng .t()
                # Nếu nó ĐÃ transpose ở bước tiền xử lý, hãy BỎ .t() đi
                edge_index_f1=torch.LongTensor(edge_index_f1) if edge_index_f1.shape[0] == 2 else torch.LongTensor(edge_index_f1).t(),
                x_f1=torch.Tensor(features_f1),
                edge_index_f2=torch.LongTensor(edge_index_f2) if edge_index_f2.shape[0] == 2 else torch.LongTensor(edge_index_f2).t(),
                x_f2=torch.Tensor(features_f2),
                
                frag_atom_mapping_1=frag_atom_mapping_1,
                frag_atom_mapping_2=frag_atom_mapping_2,
                y=torch.FloatTensor([labels]),
            )
            # Saliency map processing
            if self.saliency_map == True:
                GCNData.target_mut = _row_float_tensor(target_mut, requires_grad=True)
                GCNData.target_meth = _row_float_tensor(target_meth, requires_grad=True)
                GCNData.target_ge = _row_float_tensor(target_ge, requires_grad=True)
            else:
                GCNData.target_mut = _row_float_tensor(target_mut)
                GCNData.target_meth = _row_float_tensor(target_meth)
                GCNData.target_ge = _row_float_tensor(target_ge)
                
            GCNData.xd_pt_1 = _row_float_tensor(smiles_pt_1)
            GCNData.xd_pt_2 = _row_float_tensor(smiles_pt_2)
            
            GCNData.__setitem__('c_size_1', torch.LongTensor([c_size_1]))
            GCNData.__setitem__('c_size_2', torch.LongTensor([c_size_2]))
            
            # Lưu kích thước của đồ thị mảnh
            GCNData.__setitem__('c_size_f1', torch.LongTensor([c_size_f1]))
            GCNData.__setitem__('c_size_f2', torch.LongTensor([c_size_f2]))
            
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
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

def draw_loss(train_losses, eval_losses, title, train_label='train loss', eval_label='validation loss', y_label='Loss'):
    plt.figure()
    plt.plot(train_losses, label=train_label)
    plt.plot(eval_losses, label=eval_label)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method
    plt.close()

def draw_pearson(pearsons, title, label='validation pearson'):
    plt.figure()
    plt.plot(pearsons, label=label)

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
