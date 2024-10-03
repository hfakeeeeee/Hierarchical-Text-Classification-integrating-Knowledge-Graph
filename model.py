import json
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertModel
from torch.utils.checkpoint import checkpoint

class KHTCModel(nn.Module):
    def __init__(self, num_labels, num_nodes):
        super(KHTCModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.gcn = GCNConv(768, 64)
        self.label_embedding = nn.Embedding(num_labels, 768)
        self.linear = nn.Linear(768, 64)

    def forward(self, input_ids, attention_mask, graph):
        # Mã hóa BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        transformed_output = self.linear(pooled_output)

        # Mã hóa văn bản dựa trên kiến thức (KTE)
        concept_gcn = checkpoint(self.gcn, graph.x, graph.edge_index, use_reentrant=False)
        combined_representation = transformed_output + concept_gcn.mean(dim=0)

        # Attention phân cấp dựa trên kiến thức (KHLA)
        label_rep = self.label_embedding.weight
        valid_edge_index = graph.edge_index[:, graph.edge_index.max(dim=0)[0] < label_rep.size(0)]
        label_gcn = checkpoint(self.gcn, label_rep, valid_edge_index, use_reentrant=False)
        attention_weights = torch.matmul(combined_representation, label_gcn.T)
        attention_output = torch.matmul(attention_weights, label_gcn)

        # Kết hợp biểu diễn (KCL)
        output = self.dropout(attention_output + combined_representation)
        logits = self.classifier(output)
        return self.sigmoid(logits)
    
    def summary(self):
        def get_layer_info(layer, indent=0):
            lines = []
            for name, module in layer.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                output_shape = [list(p.size()) for p in module.parameters()]
                output_shape = output_shape[0] if output_shape else 'No parameters'
                lines.append(f"{' ' * indent}{name} ({module.__class__.__name__}):")
                lines.append(f"{' ' * (indent + 2)}Output Shape: {output_shape}")
                lines.append(f"{' ' * (indent + 2)}Param #    : {num_params}")
                lines.extend(get_layer_info(module, indent + 2))
            return lines
        
        lines = ["Model: " + self.__class__.__name__, "=" * 60]
        lines.extend(get_layer_info(self))
        lines.append("=" * 60)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        lines.append(f"Total params: {total_params}")
        lines.append(f"Trainable params: {trainable_params}")
        lines.append(f"Non-trainable params: {non_trainable_params}")
        
        print("\n".join(lines))

# Đếm số dòng trong các tập tin
def count_lines_in_files(directory):
    return sum(
        len(json.load(open(os.path.join(directory, file_name), 'r', encoding='utf-8')))
        for file_name in os.listdir(directory) if file_name.endswith('.json')
    )
