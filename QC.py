import re
import os
import glob
import pandas as pd
from IPython.display import display


INPUT_PATTERN = "/content/lần 4*.xlsx"

REQUIRED_COLUMNS = ["Claim", "Evidence", "Label"]
LABELS = ["SUPPORTED", "REFUTED", "NEI"]

CLAIM_MIN = 10
CLAIM_MAX = 25

EVIDENCE_MIN = 50
EVIDENCE_MAX = 205

STRICT_GROUP_WORD_COUNT = False

FORBIDDEN_REFUTED_WORDS = [
    "không", "khong",
    "chưa", "chua",
    "chẳng", "chang",
    "phi"
]

FILLERS = [
    "trong bối cảnh",
    "trong đoạn văn",
    "được nêu là",
    "được mô tả là",
    "như đã đề cập",
    "theo đoạn văn",
    "theo evidence",
    "trong bằng chứng",
    "đoạn văn cho thấy",
    "đoạn văn nói rằng"
]


def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


def normalize_space(text):
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text):
    text = normalize_text(text)
    text = re.sub(r"[.,;:!?()\[\]\"“”‘’]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return 0

    return len(text.split())


def has_ellipsis(text):
    text = normalize_text(text)
    return "..." in text or "…" in text


def has_filler(text):
    text = normalize_text(text).lower()
    return any(filler in text for filler in FILLERS)


def has_forbidden_refuted(row):
    if row["Label"] != "REFUTED":
        return False

    claim = normalize_text(row["Claim"]).lower()

    claim = claim.replace("không khí", "không_khí")
    claim = claim.replace("khong khi", "khong_khi")
    claim = claim.replace("không gian", "không_gian")
    claim = claim.replace("khong gian", "khong_gian")

    claim = re.sub(r"[.,;:!?()\[\]\"“”‘’]", " ", claim)
    claim = re.sub(r"\s+", " ", claim).strip()

    words = claim.split()

    return any(w in words for w in FORBIDDEN_REFUTED_WORDS)


def standardize_columns(df):
    temp = df.copy()
    temp.columns = [str(c).strip() for c in temp.columns]

    temp = temp.loc[:, ~temp.columns.astype(str).str.startswith("Unnamed")]

    rename_map = {}

    for col in temp.columns:
        low = str(col).strip().lower()

        if low == "id":
            rename_map[col] = "ID"
        elif low == "claim":
            rename_map[col] = "Claim"
        elif low == "evidence":
            rename_map[col] = "Evidence"
        elif low == "label":
            rename_map[col] = "Label"
        elif low == "group_id":
            rename_map[col] = "Group_ID"
        elif low == "split":
            rename_map[col] = "Split"

    temp = temp.rename(columns=rename_map)

    return temp


def is_valid_dataset_sheet(df):
    return all(col in df.columns for col in REQUIRED_COLUMNS)


xlsx_files = glob.glob(INPUT_PATTERN)

if len(xlsx_files) == 0:
    print("Không thấy file theo pattern:", INPUT_PATTERN)
    print("Fallback sang tất cả file .xlsx trong /content.")
    xlsx_files = glob.glob("/content/*.xlsx")

if len(xlsx_files) == 0:
    raise FileNotFoundError("Không tìm thấy file .xlsx nào trong /content.")

xlsx_files = sorted(xlsx_files, key=os.path.getmtime, reverse=True)

INPUT_FILE = None
all_sheets = None

for file_path in xlsx_files:
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, sheet_df in sheets.items():
            temp = standardize_columns(sheet_df)

            if is_valid_dataset_sheet(temp):
                INPUT_FILE = file_path
                all_sheets = sheets
                break

        if INPUT_FILE is not None:
            break

    except Exception as e:
        print(f"Bỏ qua file {file_path} vì đọc lỗi: {e}")

if INPUT_FILE is None:
    raise ValueError("Không tìm thấy file Excel nào có đủ cột Claim, Evidence, Label.")

print("Đang dùng file:", INPUT_FILE)


sheet_names = list(all_sheets.keys())
sheet_map = {name.lower().replace(" ", "_"): name for name in sheet_names}

preferred_sheet = None

for key in ["dataset_all", "all", "data_all", "dataset"]:
    if key in sheet_map:
        preferred_sheet = sheet_map[key]
        break

if preferred_sheet is not None:
    sheets_to_use = {preferred_sheet: all_sheets[preferred_sheet]}
    print("Phát hiện sheet tổng, chỉ QC sheet:", preferred_sheet)
else:
    sheets_to_use = all_sheets
    print("Không có sheet tổng, sẽ gộp các sheet hợp lệ để QC.")


dfs = []

for sheet_name, sheet_df in sheets_to_use.items():
    temp = standardize_columns(sheet_df)

    if not is_valid_dataset_sheet(temp):
        print(f"Bỏ qua sheet '{sheet_name}' vì thiếu Claim/Evidence/Label.")
        continue

    for col in temp.columns:
        temp[col] = temp[col].apply(normalize_text)

    temp = temp[~(
        (temp["Claim"] == "") &
        (temp["Evidence"] == "") &
        (temp["Label"] == "")
    )].copy()

    header_mask = (
        temp["Claim"].str.lower().eq("claim") |
        temp["Evidence"].str.lower().eq("evidence") |
        temp["Label"].str.lower().eq("label")
    )

    if "ID" in temp.columns:
        header_mask = header_mask | temp["ID"].str.lower().eq("id")

    before = len(temp)
    temp = temp[~header_mask].copy()
    after = len(temp)

    temp["Source_Sheet"] = sheet_name

    dfs.append(temp)

    print(f"Sheet '{sheet_name}': {before} dòng -> {after} dòng sau khi làm sạch header.")

if len(dfs) == 0:
    raise ValueError("Không có sheet nào hợp lệ để QC.")

df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

print("\nTổng số dòng sau khi gộp:", len(df))
print("Các cột:", df.columns.tolist())


for col in REQUIRED_COLUMNS:
    df[col] = df[col].apply(normalize_space)

df["Label"] = df["Label"].str.upper().str.strip()

if "ID" not in df.columns:
    df.insert(0, "ID", [f"row_{i + 1:04d}" for i in range(len(df))])


if "Group_ID" in df.columns and df["Group_ID"].apply(normalize_text).replace("", pd.NA).notna().any():
    df["QC_Group_ID"] = df["Group_ID"].apply(normalize_text)
    print("Dùng Group_ID có sẵn trong file để QC group.")
else:
    df["QC_Group_ID"] = df.index // 3 + 1
    print("Không có Group_ID, tự tạo QC_Group_ID theo mỗi block 3 dòng.")


df["Claim_Words"] = df["Claim"].apply(word_count)
df["Evidence_Words"] = df["Evidence"].apply(word_count)

df["Empty_Claim"] = df["Claim"].eq("")
df["Empty_Evidence"] = df["Evidence"].eq("")

df["Invalid_Label"] = ~df["Label"].isin(LABELS)

df["Claim_Length_Error"] = ~df["Claim_Words"].between(CLAIM_MIN, CLAIM_MAX)
df["Evidence_Length_Error"] = ~df["Evidence_Words"].between(EVIDENCE_MIN, EVIDENCE_MAX)

df["Evidence_Ellipsis"] = df["Evidence"].apply(has_ellipsis)
df["Duplicate_Claim"] = df.duplicated("Claim", keep=False)

df["Forbidden_REFUTED"] = df.apply(has_forbidden_refuted, axis=1)
df["Filler_Artifact"] = df["Claim"].apply(has_filler)

group_size = df.groupby("QC_Group_ID")["Claim"].transform("count")
df["Group_Size_Error"] = group_size != 3

group_label_ok = df.groupby("QC_Group_ID")["Label"].transform(
    lambda x: set(x) == set(LABELS)
)
df["Group_Label_Error"] = ~group_label_ok

group_evidence_unique = df.groupby("QC_Group_ID")["Evidence"].transform("nunique")
df["Group_Evidence_Mismatch"] = group_evidence_unique != 1

group_word_count_unique = df.groupby("QC_Group_ID")["Claim_Words"].transform("nunique")
df["Group_Word_Count_Mismatch"] = group_word_count_unique != 1


error_cols = [
    "Empty_Claim",
    "Empty_Evidence",
    "Invalid_Label",
    "Claim_Length_Error",
    "Evidence_Length_Error",
    "Evidence_Ellipsis",
    "Duplicate_Claim",
    "Forbidden_REFUTED",
    "Filler_Artifact",
    "Group_Size_Error",
    "Group_Label_Error",
    "Group_Evidence_Mismatch",
]

if STRICT_GROUP_WORD_COUNT:
    error_cols.append("Group_Word_Count_Mismatch")

df["Need_Review"] = df[error_cols].any(axis=1)


def safe_min(series):
    if len(series) == 0:
        return 0
    return int(series.min())


def safe_max(series):
    if len(series) == 0:
        return 0
    return int(series.max())


summary = {
    "Tổng số mẫu": len(df),

    "SUPPORTED": int((df["Label"] == "SUPPORTED").sum()),
    "REFUTED": int((df["Label"] == "REFUTED").sum()),
    "NEI": int((df["Label"] == "NEI").sum()),

    "Claim words min--max": f"{safe_min(df['Claim_Words'])}--{safe_max(df['Claim_Words'])}",
    "Evidence words min--max": f"{safe_min(df['Evidence_Words'])}--{safe_max(df['Evidence_Words'])}",

    "Label không hợp lệ": int(df["Invalid_Label"].sum()),

    "Claim rỗng": int(df["Empty_Claim"].sum()),
    "Evidence rỗng": int(df["Empty_Evidence"].sum()),

    "Claim dưới 10 hoặc trên 25 từ": int(df["Claim_Length_Error"].sum()),
    f"Evidence dưới {EVIDENCE_MIN} hoặc trên {EVIDENCE_MAX} từ": int(df["Evidence_Length_Error"].sum()),

    "Claim trùng hoàn toàn": int(df["Duplicate_Claim"].sum()),
    "Evidence chứa dấu ...": int(df["Evidence_Ellipsis"].sum()),

    "REFUTED chứa từ phủ định bị cấm": int(df["Forbidden_REFUTED"].sum()),
    "Claim chứa filler artifact": int(df["Filler_Artifact"].sum()),

    "Group không có đúng 3 claim": int(df["Group_Size_Error"].sum()),
    "Group không đủ 3 nhãn": int(df["Group_Label_Error"].sum()),
    "Group không cùng một Evidence": int(df["Group_Evidence_Mismatch"].sum()),

    "Lệch word count trong group": int(df["Group_Word_Count_Mismatch"].sum()),

    "Dòng cần review": int(df["Need_Review"].sum())
}

summary_df = pd.DataFrame(
    [{"Tiêu chí QC": k, "Kết quả": v} for k, v in summary.items()]
)


print("\n===== QC SUMMARY =====")
display(summary_df)

print("\n===== PHÂN BỐ NHÃN =====")
label_dist = df["Label"].value_counts().reset_index()
label_dist.columns = ["Label", "Số lượng"]
display(label_dist)

print("\n===== CÁC CẢNH BÁO KHÔNG TÍNH VÀO NEED_REVIEW =====")
warning_summary = pd.DataFrame([
    {
        "Cảnh báo": "Lệch word count trong group",
        "Số dòng": int(df["Group_Word_Count_Mismatch"].sum()),
        "Ghi chú": "Không tính vào Need_Review nếu STRICT_GROUP_WORD_COUNT = False"
    }
])
display(warning_summary)

review_df = df[df["Need_Review"]].copy()

print("\n===== DÒNG CẦN REVIEW =====")
print("Số dòng cần review:", len(review_df))

show_cols = [
    "Source_Sheet",
    "ID",
    "QC_Group_ID",
    "Claim",
    "Evidence",
    "Label",
    "Claim_Words",
    "Evidence_Words"
] + error_cols

show_cols = [c for c in show_cols if c in review_df.columns]

if len(review_df) == 0:
    print("QC PASS: Không có dòng cần review theo bộ tiêu chí chính.")
else:
    display(review_df[show_cols].head(100))
