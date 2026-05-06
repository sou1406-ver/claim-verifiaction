import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from underthesea import word_tokenize
from collections import Counter

xls = pd.ExcelFile("lần 3.xlsx")
df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]
df_all = pd.concat(df_list, ignore_index=True)

df_all = df_all.dropna(subset=['Claim', 'Evidence', 'Label'])

label_map = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT ENOUGH INFO": "NEI", "NEI": "NEI"}
df_all['Label'] = df_all['Label'].astype(str).str.strip().str.upper().map(label_map)
df_all = df_all.dropna(subset=['Label'])

print("Đang tách từ...")

def preprocess(claim, evidence):
    claim_tokenized = word_tokenize(str(claim), format="text")
    evidence_tokenized = word_tokenize(str(evidence), format="text")
    return claim_tokenized + " [SEP] " + evidence_tokenized

X = df_all.apply(lambda row: preprocess(row['Claim'], row['Evidence']), axis=1)
y = df_all['Label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Tổng số mẫu: {len(X)} | Train: {len(X_train)} | Dev: {len(X_dev)} | Test: {len(X_test)}\n")

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

def evaluate_model(model_name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    print(f"=== KẾT QUẢ: {model_name} ===")
    print("Phân phối dự đoán:", dict(Counter(y_pred)))
    print(f"Accuracy : {accuracy_score(y_eval, y_pred):.4f}")
    print(f"Macro-F1 : {f1_score(y_eval, y_pred, average='macro'):.4f}")
    print("Chi tiết:\n", classification_report(y_eval, y_pred, zero_division=0))
    print("-" * 50)

lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
lr_model.fit(X_train_tfidf, y_train)

evaluate_model("TF-IDF + Logistic Regression", lr_model, X_test_tfidf, y_test)

svm_model = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)

evaluate_model("TF-IDF + SVM", svm_model, X_test_tfidf, y_test)