import itertools
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class DrugCombinationWith3D(DATA.Data):
    def __init__(
        self,
        edge_index_1=None,
        x_1=None,
        pos_1=None,
        edge_index_2=None,
        x_2=None,
        pos_2=None,
        y=None,
    ):
        super().__init__()
        self.edge_index_1 = edge_index_1
        self.x_1 = x_1
        self.pos_1 = pos_1
        self.edge_index_2 = edge_index_2
        self.x_2 = x_2
        self.pos_2 = pos_2
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_1":
            return self.x_1.size(0)
        if key == "edge_index_2":
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class TestbedDatasetWith3D(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="davis_with_3d",
        xd_1=None,
        xd_pt_1=None,
        xd_2=None,
        xd_pt_2=None,
        xt_mut=None,
        xt_meth=None,
        xt_ge=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
        saliency_map=False,
    ):
        super().__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print(f"Pre-processed data found: {self.processed_paths[0]}, loading ...")
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print(f"Pre-processed data {self.processed_paths[0]} not found, doing pre-processing...")
            self.process(xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + ".pkl"]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd_1, xd_pt_1, xd_2, xd_pt_2, xt_mut, xt_meth, xt_ge, y, smile_graph):
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

            graph_1 = smile_graph.get(smiles_1)
            graph_2 = smile_graph.get(smiles_2)
            if graph_1 is None or graph_2 is None:
                continue

            c_size_1, features_1, edge_index_1, pos_1 = self._unpack_graph(graph_1)
            c_size_2, features_2, edge_index_2, pos_2 = self._unpack_graph(graph_2)
            graph_data = DrugCombinationWith3D(
                edge_index_1=torch.LongTensor(edge_index_1).transpose(1, 0),
                x_1=torch.tensor(np.asarray(features_1), dtype=torch.float32),
                pos_1=torch.tensor(np.asarray(pos_1), dtype=torch.float32),
                edge_index_2=torch.LongTensor(edge_index_2).transpose(1, 0),
                x_2=torch.tensor(np.asarray(features_2), dtype=torch.float32),
                pos_2=torch.tensor(np.asarray(pos_2), dtype=torch.float32),
                y=torch.FloatTensor([labels]),
            )

            if self.saliency_map:
                graph_data.target_mut = torch.tensor([target_mut], dtype=torch.float32, requires_grad=True)
                graph_data.target_meth = torch.tensor([target_meth], dtype=torch.float32, requires_grad=True)
                graph_data.target_ge = torch.tensor([target_ge], dtype=torch.float32, requires_grad=True)
            else:
                graph_data.target_mut = torch.FloatTensor([target_mut])
                graph_data.target_meth = torch.FloatTensor([target_meth])
                graph_data.target_ge = torch.FloatTensor([target_ge])

            graph_data.xd_pt_1 = torch.FloatTensor([smiles_pt_1])
            graph_data.xd_pt_2 = torch.FloatTensor([smiles_pt_2])
            graph_data.__setitem__("c_size_1", torch.LongTensor([c_size_1]))
            graph_data.__setitem__("c_size_2", torch.LongTensor([c_size_2]))
            data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Graph construction done. Saving to file.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def _unpack_graph(graph):
        if len(graph) == 4:
            return graph
        if len(graph) >= 5:
            c_size, features, edge_index, pos, *_ = graph
            return c_size, features, edge_index, pos
        raise ValueError("3D graph data must contain at least 4 elements")


def bce(y, f):
    f = np.clip(f, 1e-7, 1 - 1e-7)
    term_0 = (1 - y) * np.log(1 - f + 1e-7)
    term_1 = y * np.log(f + 1e-7)
    return -np.mean(term_0 + term_1, axis=0)


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    s = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    s = s + 1
                elif u == 0:
                    s = s + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return s / z


def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(title + ".png")
    plt.close()


def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label="test pearson")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Pearson")
    plt.legend()
    plt.savefig(title + ".png")
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    target_names,
    title="Confusion matrix",
    cmap=None,
    normalize=True,
    save_path=None,
):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=10, rotation=90)
        plt.yticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.2f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
