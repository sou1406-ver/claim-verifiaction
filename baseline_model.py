import pandas as pd
import google.generativeai as genai
import time
import json

# 1. Cấu hình API Key (Lấy miễn phí tại Google AI Studio)
API_KEY = "AIzaSyClq-pHfA_EsFlI5xtmuf4pqZsek8pKuQE".strip() # ĐIỀN KEY CỦA M VÀO ĐÂY
genai.configure(api_key=API_KEY)

# Dùng model Gemini Flash, ép nó phải trả về định dạng JSON
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

def generate_claims_from_ai(evidence):
    prompt = f"""
    Bạn là chuyên gia Xử lý Ngôn ngữ Tự nhiên tiếng Việt.
    Đọc đoạn bằng chứng sau: "{evidence}"

    Nhiệm vụ: Tạo 3 câu phát biểu (Claim) dựa trên bằng chứng trên.
    Yêu cầu bắt buộc:
    - Độ dài mỗi câu Claim phải từ 5 đến 25 từ.
    - SUPPORTED: Phát biểu đúng theo bằng chứng [cite: 14], diễn đạt tự nhiên, tuyệt đối KHÔNG copy nguyên văn.
    - REFUTED: Phát biểu sai theo bằng chứng [cite: 15] (đảo ngược ý nghĩa, sai số liệu, sai địa điểm...).
    - NOT ENOUGH INFO: Không đủ thông tin để kết luận [cite: 16] (có chứa từ khóa trong bài nhưng nhắc đến chi tiết không có trong đoạn văn).

    Chỉ trả về ĐÚNG cấu trúc JSON sau, không kèm văn bản nào khác:
    {{
        "SUPPORTED": "nội dung claim...",
        "REFUTED": "nội dung claim...",
        "NOT ENOUGH INFO": "nội dung claim..."
    }}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Lỗi gọi API: {e}")
        return None

# 2. Chạy thử nghiệm với 1 đoạn Evidence
sample_evidence = "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam, cũng là kinh đô của hầu hết các vương triều phong kiến tại Việt Nam trước đây. Do đó, lịch sử Hà Nội gắn liền với sự thăng trầm của lịch sử Việt Nam qua các thời kỳ."

print("Đang nhờ AI nghĩ Claim...")
result = generate_claims_from_ai(sample_evidence)

if result:
    print("\n--- KẾT QUẢ AI SINH RA ---")
    print(f"SUPPORTED: {result.get('SUPPORTED')}")
    print(f"REFUTED: {result.get('REFUTED')}")
    print(f"NOT ENOUGH INFO: {result.get('NOT ENOUGH INFO')}")