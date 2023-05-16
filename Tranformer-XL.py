import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import TransfoXLTokenizer, TransfoXLModel


data = pd.read_csv('D:\Σχολή\Επεξεργασία Φυσικής Γλώσσας και Σημασιολογικού Ιστού (8ο εξάμηνο)/vgsales.csv')
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')


encoded_data = data['Name'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
train_data, test_data = train_test_split(encoded_data, test_size=0.2, random_state=42)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

train_dataset = CustomDataset(train_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters())

num_epochs = 10  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        outputs = model(batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(train_dataloader)}")


model.eval()
test_dataset = CustomDataset(test_data)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

all_predictions = []

for batch in test_dataloader:
    batch = batch.to(device)
    outputs = model(batch)
    predictions = outputs.logits.argmax(dim=-1).tolist()
    all_predictions.extend(predictions)

model.eval()
all_predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        outputs = model(batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.extend(predictions.cpu().tolist())
