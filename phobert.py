import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import evaluate
import numpy as np
from sklearn.metrics import classification_report
import torch

xls = pd.ExcelFile("lần 2.xlsx")
df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]

df_all = pd.concat(df_list, ignore_index=True)
df_all = df_all.dropna(subset=['Claim', 'Evidence', 'Label'])

label_map = {"SUPPORTED": 0, "REFUTED": 1, "NOT ENOUGH INFO": 2, "NEI": 2}
df_all['Label'] = df_all['Label'].astype(str).str.strip().str.upper().map(label_map)
df_all = df_all.dropna(subset=['Label'])
df_all['Label'] = df_all['Label'].astype(int)

def preprocess_text(claim, evidence):
    return str(claim) + " </s></s> " + str(evidence)

df_all['text'] = df_all.apply(lambda row: preprocess_text(row['Claim'], row['Evidence']), axis=1)

train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['Label'])
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Label'])

train_data = Dataset.from_pandas(train_df[['text', 'Label']].rename(columns={"Label": "labels"}))
dev_data = Dataset.from_pandas(dev_df[['text', 'Label']].rename(columns={"Label": "labels"}))
test_data = Dataset.from_pandas(test_df[['text', 'Label']].rename(columns={"Label": "labels"}))

print(f"Tổng số mẫu: {len(df_all)} | Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Khởi tạo thợ đệm
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

train_tokenized = train_data.map(tokenize_function, batched=True)
dev_tokenized = dev_data.map(tokenize_function, batched=True)
test_tokenized = test_data.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3)

metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_eval=True,
    logging_strategy="no",  
    report_to="none"        
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print(torch.cuda.is_available())

print("\n Training...")
trainer.train()

print("\n--- TEST ---")
results = trainer.evaluate(test_tokenized)
print(results)

print("\n--- REPORT ---")
predictions = trainer.predict(test_tokenized)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

target_names = ["SUPPORTED (0)", "REFUTED (1)", "NEI (2)"]
print(classification_report(y_true, y_pred, target_names=target_names))