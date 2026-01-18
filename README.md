# Decision Tree from Scratch (Python)

Dự án này là **cài đặt thuật toán Decision Tree (Classification) từ đầu bằng Python**, không dùng `sklearn.tree`. Mục tiêu là hiểu rõ cách hoạt động của cây quyết định: entropy, information gain, chọn threshold và xây dựng cây đệ quy (đc học từ coursera).

---

##  Tính năng

* Decision Tree cho **bài toán phân loại**
* Tiêu chí chia: **Entropy / Information Gain**
* Hỗ trợ:

  * `max_depth`
  * Dừng khi node thuần nhất
* Tự cài đặt:

  * Tìm feature tốt nhất
  * Tìm threshold tốt nhất
  * Chia dữ liệu trái / phải

---

## Cấu trúc thư mục

```
Thuattoan/
│
├── DecisionTree.py   # Cài đặt Decision Tree từ đầu
├── test.py           # Train & test với sklearn dataset
└── README.md         # Tài liệu dự án
```

---

## Ý tưởng thuật toán

### Entropy

Đo độ hỗn loạn của nhãn:
<img width="544" height="187" alt="image" src="https://github.com/user-attachments/assets/40dab1fa-4489-47da-8c41-f9071f5dd3ce" />


### Information Gain

Đo mức giảm entropy sau khi chia:

[ IG = H(node) - H(l,r) ]

### Chọn split tốt nhất

* Duyệt từng feature
* Lấy các **threshold = unique values** của feature
* Tính Information Gain
* Chọn `(feature, threshold)` cho IG lớn nhất

---

## Cách chạy

### Cài thư viện cần thiết

```bash
pip install numpy scikit-learn
```

### Chạy test

```
python test.py
```

---

## Ví dụ sử dụng (`test.py`)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Train model
clf = DecisionTree(max_depth=36)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
```

---

## Lưu ý quan trọng

* `feature` **luôn là index (int)**
* `threshold` **luôn là giá trị so sánh (float/int)**
* Không truyền xác suất (`p_left`) vào hàm entropy
* `_entropy()` chỉ nhận **mảng nhãn y**

---

## Kết quả

* Dataset: `sklearn.datasets.load_breast_cancer`
* Accuracy thường đạt: **~90% – 95%** (tùy `max_depth`)

---

## Mục tiêu học tập

* Hiểu rõ Decision Tree hoạt động như thế nào
* Tránh học máy kiểu “black box”
* Chuẩn bị nền tảng cho:

  * Random Forest
  * Gradient Boosting

---

## 
* Viết để **học và hiểu thuật toán**, không tối ưu hiệu năng
---


