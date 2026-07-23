import os
import pickle
import numpy as np
import scipy.sparse as sp


SHARED_DATA_PATH = os.environ.get(
    "AHC_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
)


path_to_data = ""

def load(dataset):
    global path_to_data

    path_to_data = os.path.join(SHARED_DATA_PATH, dataset)

    isNPZ = os.path.exists(os.path.join(path_to_data, 'hypergraph.npz'))

    if isNPZ:
        data_dict = load_npz(dataset)
    else:
        ps = parser(dataset)

        try:
            with open(os.path.join(path_to_data, 'hypergraph.pickle'), 'rb') as handle:
                hypergraph = pickle.load(handle)

            with open(os.path.join(path_to_data, 'features.pickle'), 'rb') as handle:
                features = pickle.load(handle)

            with open(os.path.join(path_to_data, 'labels.pickle'), 'rb') as handle:
                labels = ps._1hot(pickle.load(handle))
        except FileNotFoundError as e:
            raise e

        adj = sp.lil_matrix((len(hypergraph), features.shape[0]), dtype=np.int8)

        for index, edge in enumerate(hypergraph):
            hypergraph[edge] = list(hypergraph[edge])
            adj[index, hypergraph[edge]] = 1

        adj_sp = adj.tocsr()

        data_dict = {
            'hypergraph': hypergraph,
            'features': features,
            'labels': labels,
            'name': dataset,
            'adj': adj_sp
        }

    return data_dict


def load_npz(dataset):
    path_to_data = os.path.join(SHARED_DATA_PATH, dataset)

    try:
        hg_adj = sp.load_npz(os.path.join(path_to_data, 'hypergraph.npz'))

        np.clip(hg_adj.data, 0, 1, out=hg_adj.data)

        features = sp.load_npz(os.path.join(path_to_data, 'features.npz'))
        labels = np.load(os.path.join(path_to_data, 'labels.npy'))
    except FileNotFoundError as e:
        raise e

    hypergraph = {}
    for index, edge in enumerate(hg_adj):
        hypergraph[index] = list(edge.indices)

    return {
        'hypergraph': hypergraph,
        'features': features,
        'labels': labels,
        'name': dataset,
        'adj': hg_adj
    }


class parser(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def parse(self):
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self):
        path_to_data = os.path.join(SHARED_DATA_PATH, self.dataset)

        with open(os.path.join(path_to_data, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(path_to_data, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(path_to_data, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels}

    def _1hot(self, labels):
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)
