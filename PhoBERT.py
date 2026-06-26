import pandas as pd
import numpy as np
import torch
import evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)


set_seed(42)

file_name = "lần 4.xlsx"

xls = pd.ExcelFile(file_name)
df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]

df_all = pd.concat(df_list, ignore_index=True)

df_all = df_all.dropna(subset=["Claim", "Evidence", "Label"])

label_map = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "NOT ENOUGH INFO": 2,
    "NEI": 2
}

id2label = {
    0: "SUPPORTED",
    1: "REFUTED",
    2: "NEI"
}

label2id = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "NEI": 2
}

df_all["Label"] = (
    df_all["Label"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map(label_map)
)

df_all = df_all.dropna(subset=["Label"])
df_all["Label"] = df_all["Label"].astype(int)


def preprocess_text(claim, evidence):
    return str(claim) + " </s></s> " + str(evidence)


df_all["text"] = df_all.apply(
    lambda row: preprocess_text(row["Claim"], row["Evidence"]),
    axis=1
)

print("========== THỐNG KÊ DỮ LIỆU ==========")
print(f"Tổng số mẫu hợp lệ: {len(df_all)}")

print("\nPhân phối nhãn toàn bộ dữ liệu:")
print(df_all["Label"].value_counts().sort_index())


unique_evidences = df_all["Evidence"].unique()

ev_train, ev_temp = train_test_split(
    unique_evidences,
    test_size=0.30,
    random_state=42
)

ev_dev, ev_test = train_test_split(
    ev_temp,
    test_size=0.50,
    random_state=42
)

train_df = df_all[df_all["Evidence"].isin(ev_train)].reset_index(drop=True)
dev_df = df_all[df_all["Evidence"].isin(ev_dev)].reset_index(drop=True)
test_df = df_all[df_all["Evidence"].isin(ev_test)].reset_index(drop=True)

print("\n========== CHIA DỮ LIỆU THEO EVIDENCE ==========")
print(f"Train: {len(train_df)} ({len(train_df) / len(df_all) * 100:.1f}%)")
print(f"Dev  : {len(dev_df)} ({len(dev_df) / len(df_all) * 100:.1f}%)")
print(f"Test : {len(test_df)} ({len(test_df) / len(df_all) * 100:.1f}%)")

print("\nPhân phối nhãn Train:")
print(train_df["Label"].value_counts().sort_index())

print("\nPhân phối nhãn Dev:")
print(dev_df["Label"].value_counts().sort_index())

print("\nPhân phối nhãn Test:")
print(test_df["Label"].value_counts().sort_index())


train_data = Dataset.from_pandas(
    train_df[["text", "Label"]].rename(columns={"Label": "labels"})
)

dev_data = Dataset.from_pandas(
    dev_df[["text", "Label"]].rename(columns={"Label": "labels"})
)

test_data = Dataset.from_pandas(
    test_df[["text", "Label"]].rename(columns={"Label": "labels"})
)


model_name = "vinai/phobert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )


train_tokenized = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

dev_tokenized = dev_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

test_tokenized = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)


f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = acc_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )

    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1["f1"]
    }


print(f"\nGPU available: {torch.cuda.is_available()}")

training_args = TrainingArguments(
    output_dir="./phobert_results",
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    save_total_limit=2,
    report_to="none",

    seed=42,
    data_seed=42,

    fp16=torch.cuda.is_available()
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("\n========== TRAINING ==========")
trainer.train()


print("\n========== DEV SET ==========")
dev_results = trainer.evaluate(dev_tokenized)
print(dev_results)


print("\n========== TEST SET ==========")
test_results = trainer.evaluate(test_tokenized)
print(test_results)


print("\n========== CLASSIFICATION REPORT ==========")
predictions = trainer.predict(test_tokenized)

y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

target_names = [
    "SUPPORTED (0)",
    "REFUTED (1)",
    "NEI (2)"
]

print(
    classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )
)
