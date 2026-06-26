import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from underthesea import word_tokenize


file_name = "lần 4.xlsx"

xls = pd.ExcelFile(file_name)

df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]
df_all = pd.concat(df_list, ignore_index=True)

df_all = df_all.dropna(subset=["Claim", "Evidence", "Label"]).reset_index(drop=True)


label_map = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "NOT ENOUGH INFO": "NEI",
    "NEI": "NEI"
}

df_all["Label"] = (
    df_all["Label"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map(label_map)
)

df_all = df_all.dropna(subset=["Label"]).reset_index(drop=True)


def preprocess(claim, evidence):
    claim_tokenized = word_tokenize(str(claim), format="text")
    evidence_tokenized = word_tokenize(str(evidence), format="text")

    return claim_tokenized + " [SEP] " + evidence_tokenized


print("Đang tách từ tiếng Việt...")

df_all["Text"] = df_all.apply(
    lambda row: preprocess(row["Claim"], row["Evidence"]),
    axis=1
)


unique_evidences = df_all["Evidence"].unique()

ev_train, ev_temp = train_test_split(
    unique_evidences,
    test_size=0.20,
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
print(f"Train: {len(train_df)}")
print(f"Dev  : {len(dev_df)}")
print(f"Test : {len(test_df)}")


X_train = train_df["Text"]
y_train = train_df["Label"]

X_dev = dev_df["Text"]
y_dev = dev_df["Label"]

X_test = test_df["Text"]
y_test = test_df["Label"]


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_dev_tfidf = vectorizer.transform(X_dev)
X_test_tfidf = vectorizer.transform(X_test)


def evaluate_model(model_name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)

    print(f"\n========== {model_name} ==========")
    print("Phân phối nhãn thật:", dict(Counter(y_eval)))
    print("Phân phối dự đoán  :", dict(Counter(y_pred)))

    print(f"Accuracy: {accuracy_score(y_eval, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_eval, y_pred, average='macro'):.4f}")
    print()

    print(classification_report(y_eval, y_pred, zero_division=0))
    print("-" * 70)


lr_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    class_weight="balanced"
)

lr_model.fit(X_train_tfidf, y_train)

evaluate_model(
    "TF-IDF + Logistic Regression - DEV",
    lr_model,
    X_dev_tfidf,
    y_dev
)

evaluate_model(
    "TF-IDF + Logistic Regression - TEST",
    lr_model,
    X_test_tfidf,
    y_test
)


svm_model = LinearSVC(
    C=1.0,
    random_state=42,
    class_weight="balanced"
)

svm_model.fit(X_train_tfidf, y_train)

evaluate_model(
    "TF-IDF + SVM - DEV",
    svm_model,
    X_dev_tfidf,
    y_dev
)

evaluate_model(
    "TF-IDF + SVM - TEST",
    svm_model,
    X_test_tfidf,
    y_test
)
