import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix
import torch

data = pd.read_csv('D:\Σχολή\Επεξεργασία Φυσικής Γλώσσας και Σημασιολογικού Ιστού (8ο εξάμηνο)/vgsales.csv')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
encoded_data = data['Name'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))
encoded_data = encoded_data.apply(lambda x: x[:128]) 
encoded_data = encoded_data.tolist()
labels = data['Genre'].values

train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_data, labels, test_size=0.2, random_state=42)

train_inputs = torch.tensor(train_inputs, dtype=torch.long)
test_inputs = torch.tensor(test_inputs, dtype=torch.long)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

config = GPT2Config.from_pretrained('gpt2', num_labels=len(data['Genre'].unique()))

model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_inputs) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0

    for i in range(len(train_inputs)):
        inputs = train_inputs[i].unsqueeze(0).to(device)
        labels = train_labels[i].unsqueeze(0).to(device)

        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(train_inputs)}")

model.eval()
predicted_labels = []
with torch.no_grad():
    for i in range(len(test_inputs)):
        inputs = test_inputs[i].unsqueeze(0).to(device)
        labels = test_labels[i].unsqueeze(0).to(device)

        outputs = model(inputs, labels=labels)
        logits = outputs.logits
        predicted_label = logits.argmax().item()
        predicted_labels.append(predicted_label)

accuracy = accuracy_score(test_labels, predicted_labels)
confusion_matrix = confusion_matrix(test_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix)
