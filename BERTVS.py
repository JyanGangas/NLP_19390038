import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


df = pd.read_csv('D:\Σχολή\Επεξεργασία Φυσικής Γλώσσας και Σημασιολογικού Ιστού (8ο εξάμηνο)/vgsales.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

inputs = df['Name'].apply(tokenize_text)
input_ids = torch.cat([input_dict['input_ids'] for input_dict in inputs], dim=0)
attention_masks = torch.cat([input_dict['attention_mask'] for input_dict in inputs], dim=0)
labels = torch.tensor(df['Global_Sales'].values)

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Genre'].unique()))
optimizer = AdamW(model.parameters(), lr=2e-5)

train_dataset = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model.train()
for epoch in range(5): 
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Average Loss: {total_loss/len(train_dataloader)}")

test_dataset = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
total_correct = 0
total_samples = 0
for batch in test_dataloader:
    batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        total_correct += (predicted_labels == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy}")

with torch.no_grad():
    model.eval()
    test_loss = 0
    predictions = []
    true_labels = []

    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        _, predicted_labels = torch.max(logits, dim=1)

        predictions.extend(predicted_labels.detach().cpu().numpy())
        true_labels.extend(labels.detach().cpu().numpy())

        loss = outputs.loss
        test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    test_accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    cm = confusion_matrix(true_labels, predictions)

    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.2%}".format(test_accuracy))
    print("Precision: {:.2%}".format(precision))
    print("Recall: {:.2%}".format(recall))
    print("F1-Score: {:.2%}".format(f1_score))
    print("Confusion Matrix:")
    print(cm)

