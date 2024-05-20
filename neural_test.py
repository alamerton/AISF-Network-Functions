#%%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from transformers import Trainer
from datasets import load_dataset
from transformers import TrainingArguments
from datasets import load_metric

# %%
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokeniser = AutoTokenizer.from_pretrained('bert-base-uncased')

#%%
dataset = load_dataset("Alamerton/Synthetic-Palindromes")

#%%
tokenised_dataset = dataset.map(lambda examples: tokeniser(examples['train'], padding='max_length', truncation=True), batched=True)

# %%
train_dataset, eval_dataset = train_test_split(tokenised_dataset, test_size=0.2, random_state=42)

# %%
metric = load_metric("metric_name")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    return metric.compute(predictions=predictions, references=labels)

# %%

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall,
#     }

#%%
training_args = TrainingArguments(
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    learning_rate='1e-4',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir='./results',
    overwrite_output_dir=True,
    remove_unused_columns=False,
    save_total_limit=1
)


# %%

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
print("Training complete.")

# %%
