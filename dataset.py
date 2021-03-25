import pandas as pd
import torch as th
import dgl
from dgl.data import DGLDataset


class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')

    def process(self):
        # Loading data
        df_interactions = pd.read_csv('data/interactions.csv')
        df_members = pd.read_csv('data/members.csv')

        node_features = th.from_numpy(df_members['Age'].to_numpy())
        node_labels = th.from_numpy(df_members['Club'].astype('category').cat.codes.to_numpy())
        edge_features = th.from_numpy(df_interactions['Weight'].to_numpy())
        edges_src = th.from_numpy(df_interactions['Src'].to_numpy())
        edges_dst = th.from_numpy(df_interactions['Dst'].to_numpy())

        # Creating Graph Model
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=df_members.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # Marking train-test portions
        n_nodes = df_members.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = th.zeros(n_nodes, dtype=th.bool)
        val_mask = th.zeros(n_nodes, dtype=th.bool)
        test_mask = th.zeros(n_nodes, dtype=th.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1