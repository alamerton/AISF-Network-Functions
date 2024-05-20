#%%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

# %%
# Load the pre-trained BERT model and tokenizer.
tokeniser = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

#%%
from datasets import load_dataset

dataset = load_dataset("Alamerton/Synthetic-Palindromes")

#%%
# Tokenize data
def tokenize_function(examples):
    return tokeniser(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenised_dataset = dataset.map(tokenize_function, batched=True)
tokenised_dataset = tokenised_dataset.remove_columns(["text"])
# tokenised_dataset = tokenised_dataset.set_format('torch')

print(tokenised_dataset)
    
dataloader = DataLoader(tokenised_dataset["train"], batch_size=32, shuffle=True)

print(dataloader)
#%%
model.train()

model.eval()

print("Training complete.")