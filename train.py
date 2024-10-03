import time
import numpy as np 
import torch
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from config import DEVICE
from dataset import CustomDataset, get_num_labels, read_json_from_chunks
from model import KHTCModel, count_lines_in_files
from graph import create_graph
from transformers import BertTokenizer, BertModel

# Tính toán độ chính xác của tập huấn luyện
def calculate_accuracy(predictions, labels):
    preds = (predictions > 0.5).float()
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

# Đánh giá mô hình
def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask, graph_data)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_predictions = (all_predictions > 0.5).astype(int)

    precision = precision_score(all_labels, all_predictions, average='micro')
    recall = recall_score(all_labels, all_predictions, average='micro')
    f1 = f1_score(all_labels, all_predictions, average='micro')
    accuracy = accuracy_score(all_labels, all_predictions)

    return total_loss / len(dataloader), precision, recall, f1, accuracy

# Huấn luyện mô hình
def train_model(model, train_loader, validate_loader, test_loader, optimizer, loss_fn, num_epochs=10, save_path='model.pth', patience=3):
    train_samples = count_lines_in_files(train_chunk_dir)
    validate_samples = count_lines_in_files(validate_chunk_dir)
    test_samples = count_lines_in_files(test_chunk_dir)

    print(f"Train on {train_samples} samples, Validate on {validate_samples} samples, Test on {test_samples} samples")

    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        total_accuracy = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            with autocast():
                outputs = model(input_ids, attention_mask, graph_data)
                loss = loss_fn(outputs, labels)
                accuracy = calculate_accuracy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        val_loss, _, _, _, val_accuracy = evaluate_model(model, validate_loader, loss_fn)
        test_loss, _, _, _, test_accuracy = evaluate_model(model, test_loader, loss_fn)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} \n"
              f"{train_samples}/{train_samples} "
              f"[{'=' * 25}] - "
              f"{int(epoch_time*1000)}ms/step - "
              f"loss: {avg_loss:.4f} - acc: {avg_accuracy:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f} - "
              f"test_loss: {test_loss:.4f} - test_acc: {test_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model.pth')
            print(f"Model saved at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    # Đường dẫn đến các tập dữ liệu
    train_chunk_dir = 'Train_Samples'
    validate_chunk_dir = 'Validate_Samples'
    test_chunk_dir = 'Test_Samples'

    # Đếm số nhãn và khởi tạo tokenizer
    num_labels = get_num_labels([train_chunk_dir, validate_chunk_dir, test_chunk_dir])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Khởi tạo tập dữ liệu và DataLoader
    train_dataset = CustomDataset(train_chunk_dir, num_labels, tokenizer)
    validate_dataset = CustomDataset(validate_chunk_dir, num_labels, tokenizer)
    test_dataset = CustomDataset(test_chunk_dir, num_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    # Đọc dữ liệu và tạo đồ thị
    train_data = read_json_from_chunks(train_chunk_dir)
    validate_data = read_json_from_chunks(validate_chunk_dir)
    test_data = read_json_from_chunks(test_chunk_dir)
    combined_data = train_data + validate_data + test_data
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    print("Creating graph")
    graph_data, num_nodes, node_index = create_graph(combined_data, tokenizer, bert_model, device=DEVICE)

    # Khởi tạo mô hình và optimizer
    model = KHTCModel(num_labels=num_labels, num_nodes=num_nodes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    # Huấn luyện mô hình
    print('Start training')
    train_model(model, train_loader, validate_loader, test_loader, optimizer, loss_fn, num_epochs=10, save_path='model.pth', patience=3)

    # Đánh giá mô hình
    val_loss, val_precision, val_recall, val_f1, val_accuracy = evaluate_model(model, validate_loader, loss_fn)
    print(f"Validation Accuracy: {val_accuracy:.4f} - Validation Precision: {val_precision:.4f} - Validation Recall: {val_recall:.4f} - Validation F1: {val_f1:.4f}")

    test_loss, test_precision, test_recall, test_f1, test_accuracy = evaluate_model(model, test_loader, loss_fn)
    print(f"Test Accuracy: {test_accuracy:.4f} - Test Precision: {test_precision:.4f} - Test Recall: {test_recall:.4f} - Test F1: {test_f1:.4f}")

    # Xuất ra thông tin chi tiết của model
    model.summary()