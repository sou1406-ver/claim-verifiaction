import pandas as pd

# Định nghĩa các cột theo yêu cầu đề bài
columns = ['ID', 'Claim', 'Evidence', 'Label'] 

# Tạo DataFrame trống
df_empty = pd.DataFrame(columns=columns)

# Xuất ra file Excel trắng
df_empty.to_excel("dataset_mau.xlsx", index=False)
print("Đã tạo file dataset_mau.xlsx thành công!")