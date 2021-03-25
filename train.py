#!/usr/bin/env python

import os, json

import torch as th
import torch.nn.functional as F 

from dgl.data import CoraFullDataset
from dgl.data.utils import split_dataset, save_graphs

from model import GraphConvolutionalNetwork


def main():
    # Setup Variables
    config_dir = '/opt/ml/input/config'
    graph_dir = '/opt/ml/input/data'
    model_dir = '/opt/ml/model'

    with open(os.path.join(config_dir, 'hyperparameters.json'), 'r') as file:
        parameters_dict = json.load(file)

        learning_rate = float(parameters_dict['learning-rate'])
        epochs = int(parameters_dict['epochs'])

    # Getting dataset
    dataset = CoraFullDataset()
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']

    # Spliting dataset
    train_mask, val_mask = split_dataset(graph, [0.8, 0.2])

    # Creating Model
    model = GraphConvolutionalNetwork(features.shape[1], 16, dataset.num_classes)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        pred = model(graph, features)
        loss = F.cross_entropy(pred[train_mask.indices], labels[train_mask.indices].to(th.long))

        train_acc = (labels[train_mask.indices] == pred[train_mask.indices].argmax(1)).float().mean()
        val_acc = (labels[val_mask.indices] == pred[val_mask.indices].argmax(1)).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}/{epochs} | Loss: {loss.item()}, train_accuracy: {train_acc}, val_accuracy: {val_acc}')

    
    # Saving Graph
    save_graphs(os.path.join(graph_dir, 'dgl-citation-network.bin'), graph)

    # Saving Model
    th.save(model.state_dict(), os.path.join(model_dir, 'dgl-citation-network.pt'))

if __name__ == '__main__':
    main()