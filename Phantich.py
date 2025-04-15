import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Giả lập danh sách ngành với điểm chuẩn
majors = pd.read_csv('Nganh.csv')  # Bạn cần đảm bảo Nganh.csv tồn tại và có đầy đủ dữ liệu

# Giả lập dữ liệu sinh viên
np.random.seed(42)
students = []
for _, row in majors.iterrows():
    for _ in range(int(row['Chi_tieu'] * 2)):  # Mỗi ngành có gấp đôi chỉ tiêu sinh viên đăng ký
        score = np.random.uniform(20, 30)  # Sinh viên có điểm thi ngẫu nhiên từ 20 đến 30
        label = 1 if score >= row['Diem_thi_tot_nghiep'] else 0  # Nếu điểm >= điểm chuẩn, đậu vào ngành
        students.append([score, row['Ma'], label])  # Lưu điểm thi, mã ngành và nhãn (đậu/không)

df_students = pd.DataFrame(students, columns=["Score", "Major", "Admitted"])

# One-hot encode ngành học
df_encoded = pd.get_dummies(df_students, columns=["Major"])

X = df_encoded.drop("Admitted", axis=1)  # Các đặc trưng đầu vào (điểm thi và ngành)
y = df_encoded["Admitted"]  # Nhãn (đậu hoặc không)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train mô hình KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Kiểm tra độ chính xác trên tập kiểm tra
accuracy = model.score(X_test, y_test)
print(f"Độ chính xác của mô hình trên tập kiểm tra: {accuracy * 100:.2f}%")

# Dự đoán xác suất đậu cho điểm thi = 26.2 với từng ngành
results = []
for _, row in majors.iterrows():
    # Tạo vector đầu vào cho điểm thi = 26.2 cho mỗi ngành
    input_data = {"Score": 26.2}
    for major in majors['Ma']:
        input_data[f"Major_{major}"] = 1 if major == row['Ma'] else 0
    
    input_df = pd.DataFrame([input_data])  # Chuyển thành DataFrame để dự đoán
    prob = model.predict_proba(input_df)[0][1]  # Xác suất đậu vào ngành này
    results.append((row['Ma'], prob))  # Lưu lại mã ngành và xác suất

# Sắp xếp kết quả theo xác suất đậu từ cao đến thấp
results = sorted(results, key=lambda x: x[1], reverse=True)

# In kết quả
print("\n📈 Xác suất đậu vào các ngành với điểm thi 26.2:")
for major_code, p in results:
    print(f"Ngành mã {major_code}: {p * 100:.2f}%")
