import torch
import numpy as np
from torch_geometric.data import Data

# Tạo đồ thị
def create_graph(data, tokenizer, model, device='cpu'):
    nodes = {concept for item in data for concept in item['title'] + item['abstract']}
    node_index = {node: idx for idx, node in enumerate(nodes)}
    
    def edge_generator():
        for item in data:
            concepts = item['title'] + item['abstract']
            for i in range(len(concepts) - 1):
                for j in range(i + 1, len(concepts)):
                    yield node_index[concepts[i]], node_index[concepts[j]]
                    
    node_features = {
        node_index[node]: model(**tokenizer(node, return_tensors='pt').to(device)).last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
        for node in nodes
    }

    edge_index = torch.tensor(list(edge_generator()), dtype=torch.long).t().contiguous().to(device)
    x = torch.tensor(np.array([node_features[idx] for idx in range(len(nodes))]), dtype=torch.float).to(device)
    return Data(x=x, edge_index=edge_index), len(nodes), node_index