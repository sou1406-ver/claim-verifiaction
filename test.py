import google.generativeai as genai
import pandas as pd
import time
import json

# --- KIỂM TRA KỸ API KEY ---
# Ông vào https://aistudio.google.com/ lấy key mới nếu cần
API_KEY = "DÁN_KEY_CỦA_ÔNG_VÀO_ĐÂY" 

genai.configure(api_key=API_KEY)

# Sử dụng gemini-1.5-flash (nhanh và ít lỗi 400 hơn)
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

def generate_data(evidence_text):
    prompt = f"""
    Dựa trên đoạn văn: "{evidence_text}"
    Hãy tạo 3 câu Claim (5-25 từ) cho 3 nhãn: SUPPORTED, REFUTED, NOT ENOUGH INFO.
    Yêu cầu: Câu văn tự nhiên, không copy nguyên văn.
    Trả về danh sách JSON gồm các trường: claim, label.
    """
    try:
        response = model.generate_content(prompt)
        # Ép về kiểu dữ liệu list trong Python
        return json.loads(response.text)
    except Exception as e:
        print(f"Lỗi tại dòng: {e}")
        return None

try:
    # Đọc tối đa 100 dòng để test trước
    df_raw = pd.read_csv("evidence_raw.csv")
    df_test = df_raw.head(100) 
    
    final_results = []
    current_id = 1

    print("Đang chạy... Nếu thấy đứng im là do đang nghỉ 4s tránh khóa API.")

    for index, row in df_test.iterrows():
        evidence = row['evidence']
        # Xử lý trường hợp evidence bị trống
        if pd.isna(evidence) or str(evidence).strip() == "":
            continue
            
        claims = generate_data(str(evidence))
        
        if claims:
            for c in claims:
                final_results.append({
                    "ID": f"vi_claim_{str(current_id).zfill(5)}",
                    "Claim": c.get('claim', ''),
                    "Evidence": evidence,
                    "Label": c.get('label', '')
                })
                current_id += 1
            print(f"--> Xong đoạn {index + 1}/100")
        
        # Nghỉ 4 giây để không bị lỗi 429 (quá tải) hoặc 400 (do gửi quá nhanh)
        time.sleep(4) 

    # Lưu kết quả
    df_final = pd.DataFrame(final_results)
    df_final.to_excel("dataset_300_mau.xlsx", index=False)
    print("\n✅ XONG! Đã có file dataset_300_mau.xlsx")

except Exception as e:
    print(f"Lỗi hệ thống: {e}")