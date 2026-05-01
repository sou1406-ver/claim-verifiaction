import pandas as pd
import google.generativeai as genai
import json
import time
import os
import sys

# 1. Cấu hình
API_KEY = "AIzaSyClq-pHfA_EsFlI5xtmuf4pqZsek8pKuQE".strip()
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

input_file = "Dataset_Claim_Verification_Template.xlsx"
output_file = "Final_Dataset_Auto_V3.csv"

# 2. Đọc và Checkpoint
if not os.path.exists(input_file):
    print("M chưa có file Template rồi Phúc ơi!")
    sys.exit()

df_input = pd.read_excel(input_file)
unique_evidences = df_input['Evidence'].dropna().unique().tolist()

final_results = []
start_idx = 0

if os.path.exists(output_file):
    df_temp = pd.read_csv(output_file)
    final_results = df_temp.to_dict('records')
    start_idx = len(final_results) // 3
    print(f"Phát hiện chạy dở. Đang tiếp tục từ đoạn {start_idx + 1}...")

def safe_save():
    if final_results:
        pd.DataFrame(final_results).to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n>> Đã lưu an toàn {len(final_results)} mẫu vào {output_file}")

# 3. Vòng lặp chính với "Giáp bảo vệ"
try:
    for idx in range(start_idx, len(unique_evidences)):
        evidence = unique_evidences[idx]
        print(f"[{idx+1}/{len(unique_evidences)}] Đang xin AI sinh claim...")

        # Prompt ép đúng chuẩn đồ án [cite: 30, 45]
        prompt = f"""
        Bằng chứng: "{evidence}"
        Nhiệm vụ: Tạo 3 câu khẳng định (Claim) tiếng Việt (5-25 từ).
        - SUPPORTED: Đúng, dùng từ đồng nghĩa, KHÔNG copy nguyên văn[cite: 14, 45].
        - REFUTED: Sai hoàn toàn (đổi mốc thời gian/địa danh)[cite: 15].
        - NOT ENOUGH INFO: Liên quan nhưng không thể kết luận[cite: 16].
        Trả về JSON duy nhất: {{"SUPPORTED": "...", "REFUTED": "...", "NOT ENOUGH INFO": "..."}}
        """

        success = False
        for retry in range(3):
            try:
                response = model.generate_content(prompt)
                res = json.loads(response.text)
                
                for label in ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]:
                    final_results.append({
                        "ID": f"vi_claim_{len(final_results)+1:05d}", # 
                        "Claim": res.get(label).strip(),
                        "Evidence": evidence.strip(),
                        "Label": label
                    })
                success = True
                break
            except Exception as e:
                wait = (retry + 1) * 20
                print(f"Lỗi rồi, đợi {wait}s... ({e})")
                time.sleep(wait)

        if (idx + 1) % 5 == 0:
            safe_save()

        # Nghỉ 10 giây để không bị khóa mỏm (429)
        time.sleep(10)

except KeyboardInterrupt:
    print("\n\nPhúc bấm dừng à? Đợi tí để lưu nốt chỗ vừa chạy đã...")
    safe_save()
    sys.exit()  

safe_save()
print("XONG RỒI! Đi ráp Gundam thôi m!")