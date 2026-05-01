import wikipediaapi
import pandas as pd
import re

# 1. Cấu hình Wikipedia tiếng Việt
# Thay 'YourName (your@email.com)' bằng thông tin của bạn để tuân thủ quy định Wikipedia
wiki = wikipediaapi.Wikipedia(
    user_agent='MyDataCollector/1.0 (phuc123vodanh@example.com)',
    language='vi',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def get_evidence_from_wiki(topics, limit_per_topic=100):
    data = []
    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            continue
        
        # Chia văn bản thành các đoạn (paragraphs)
        paragraphs = page.text.split('\n')
        
        count = 0
        for p in paragraphs:
            # Làm sạch văn bản cơ bản
            p = re.sub(r'\[\d+\]', '', p).strip() 
            words = p.split()
            word_count = len(words)
            
            # Lọc độ dài từ 50 đến 200 từ theo yêu cầu 
            if 50 <= word_count <= 200:
                data.append(p)
                count += 1
                if count >= limit_per_topic:
                    break
    return data

# 2. Danh sách các chủ đề đa dạng để tránh trùng lặp 
topics = [
    "Việt Nam", "Lịch sử Việt Nam", "Hà Nội", "Hồ Chí Minh (thành phố)", 
    "Biến đổi khí hậu", "Trí tuệ nhân tạo", "Hệ Mặt Trời", "Chiến tranh Thế giới thứ hai",
    "Ẩm thực Việt Nam", "Kinh tế Việt Nam", "Vịnh Hạ Long", "Năng lượng tái tạo"
]

# 3. Chạy chương trình
print("Đang lấy dữ liệu...")
raw_evidence = get_evidence_from_wiki(topics, limit_per_topic=150)

# 4. Lưu vào DataFrame và loại bỏ trùng lặp hoàn toàn 
df = pd.DataFrame(raw_evidence, columns=['evidence'])
df = df.drop_duplicates().reset_index(drop=True)

# 5. Xuất file CSV
df.to_csv('evidence_raw.csv', index=False, encoding='utf-8-sig')
print(f"Hoàn thành! Đã lấy được {len(df)} đoạn Evidence.")
print("File đã lưu: evidence_raw.csv")