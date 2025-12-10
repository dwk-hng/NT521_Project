"""
NPM (JavaScript) Mono-Language Decision Tree Training and Testing
==================================================================

Script này thực hiện:
1. Đọc dữ liệu đã được gán nhãn (Labelled_Dataset.csv)
2. Lọc chỉ lấy dữ liệu NPM packages (JavaScript)
3. Đánh giá hiệu suất Decision Tree với cross-validation (10 folds)
4. Tìm hyperparameters tối ưu bằng Bayesian Optimization
5. Huấn luyện mô hình cuối cùng với toàn bộ dữ liệu NPM
6. Lưu mô hình đã train vào file .pkl

Yêu cầu:
    - File Labelled_Dataset.csv phải có trong cùng thư mục
    - Đã cài đặt các thư viện: pandas, joblib, scikit-learn, psutil
"""

import pandas as pd
import joblib
import datetime
import time
import psutil
import os
from utilities_functions import evaluation_decision_tree
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================================

print("="*70)
print("NPM MONO-LANGUAGE DECISION TREE - TRAINING & TESTING")
print("="*70)
print("\n[*] Đang đọc dữ liệu từ Labelled_Dataset.csv...")

try:
    data = pd.read_csv('./Labelled_Dataset.csv', sep=',')
    print(f"[✓] Đọc dữ liệu thành công: {len(data)} samples (tất cả repositories)")
    
    # Lọc chỉ lấy NPM packages
    data = data[data['Package Repository'] == 'NPM']
    
    print(f"[✓] Đã lọc chỉ lấy NPM packages: {len(data)} samples")
    print(f"    - Số features: {len(data.columns) - 3}")  # trừ 3 cột: Malicious, Package Repository, Package Name
    print(f"    - Malicious (NPM): {data['Malicious'].sum()}")
    print(f"    - Benign (NPM): {len(data) - data['Malicious'].sum()}")
    print(f"    - Malicious ratio: {data['Malicious'].sum() / len(data) * 100:.2f}%")
    
    if len(data) == 0:
        print("[✗] ERROR: Không có dữ liệu NPM trong dataset!")
        exit(1)
        
except FileNotFoundError:
    print("[✗] ERROR: Không tìm thấy file 'Labelled_Dataset.csv'")
    print("    Vui lòng đảm bảo file CSV nằm trong cùng thư mục với script này.")
    exit(1)
except Exception as e:
    print(f"[✗] ERROR khi đọc file: {e}")
    exit(1)

# ============================================================================
# 2. ĐÁNH GIÁ VÀ TỐI ƯU HÓA DECISION TREE
# ============================================================================

print("\n" + "="*70)
print("BƯỚC 1: ĐÁNH GIÁ VÀ TÌM KIẾM HYPERPARAMETERS TỐI ƯU")
print("="*70)
print("[*] Bắt đầu quá trình cross-validation (10 folds)...")
print("[*] Chỉ sử dụng dữ liệu NPM (JavaScript packages)...")
print("[*] Sử dụng Bayesian Optimization để tìm hyperparameters tối ưu...")
print()

# Bắt đầu đo thời gian và memory
start_time = time.time()
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 ** 2)  # MB

# Chạy evaluation (10-fold cross-validation + Bayesian Optimization)
performance, hyperparams = evaluation_decision_tree(data)

# Tính toán resource usage
final_memory = process.memory_info().rss / (1024 ** 2)  # MB
memory_usage = final_memory - initial_memory
elapsed_time = time.time() - start_time

print("\n" + "="*70)
print("KẾT QUẢ ĐÁNH GIÁ Từ 10-FOLD CROSS-VALIDATION")
print("="*70)
print("[!] LƯU Ý: Các kết quả dưới đây là ước lượng hiệu suất của mô hình")
print("    trên dữ liệu mới (chưa thấy) dựa trên 10 lần train-test khác nhau.")
print("    Đây KHÔNG phải kết quả của mô hình cuối cùng được lưu.")
print("="*70)

# In thông tin về resources
print(f"\n[*] Thời gian thực thi: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"[*] Memory usage: {memory_usage:.2f} MB")

# In hyperparameters tối ưu
print(f"\n[*] Hyperparameters tối ưu tìm được:")
for key, value in hyperparams.items():
    print(f"    - {key}: {value}")

# ============================================================================
# 3. IN KẾT QUẢ CHI TIẾT (CHỈ NPM)
# ============================================================================

print("\n" + "-"*70)
print("METRICS CHO NPM PACKAGES (Mono-Language)")
print("-"*70)
print(f"Test Precision       : {round(performance.iloc[-16,0],2)} ± {round(performance.iloc[-16,1],2)}%")
print(f"Test Recall          : {round(performance.iloc[-15,0],2)} ± {round(performance.iloc[-15,1],2)}%")
print(f"Test F1-Score        : {round(performance.iloc[-14,0],2)} ± {round(performance.iloc[-14,1],2)}%")
print(f"Test Accuracy        : {round(performance.iloc[-13,0],2)} ± {round(performance.iloc[-13,1],2)}%")

print("\n" + "-"*70)
print("CONFUSION MATRIX COMPONENTS")
print("-"*70)
print(f"False Positive (benign → malicious): {round(performance.iloc[-12,0],0)} ± {round(performance.iloc[-12,1],0)}")
print(f"False Negative (malicious → benign): {round(performance.iloc[-11,0],0)} ± {round(performance.iloc[-11,1],0)}")
print(f"True Negative  (benign → benign)   : {round(performance.iloc[-10,0],0)} ± {round(performance.iloc[-10,1],0)}")
print(f"True Positive  (malicious → mal.)  : {round(performance.iloc[-9,0],0)} ± {round(performance.iloc[-9,1],0)}")

# ============================================================================
# 4. HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG VỚI TOÀN BỘ DỮ LIỆU NPM
# ============================================================================

print("\n" + "="*70)
print("BƯỚC 2: HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG")
print("="*70)
print("[*] Đang train mô hình Decision Tree với toàn bộ dữ liệu NPM...")

# Chuẩn bị dữ liệu
X = data.drop(labels=['Package Repository', 'Malicious', 'Package Name'], axis=1).values
Y = data['Malicious'].astype('int').values

print(f"    - Training set size: {len(X)} samples (NPM only)")
print(f"    - Number of features: {X.shape[1]}")
print(f"    - Positive class (Malicious): {Y.sum()}")
print(f"    - Negative class (Benign): {len(Y) - Y.sum()}")

# Khởi tạo và train classifier với hyperparameters tối ưu
classifier_DT = DecisionTreeClassifier(
    random_state=123,
    criterion=hyperparams['criterion'],
    max_depth=hyperparams['max_depth'],
    max_features=hyperparams['max_features'],
    min_samples_leaf=hyperparams['min_sample_leaf'],
    min_samples_split=hyperparams['min_sample_split']
)

classifier_DT.fit(X=X, y=Y)

print("[✓] Hoàn tất huấn luyện mô hình!")

# ============================================================================
# 5. LƯU MÔ HÌNH
# ============================================================================

print("\n" + "="*70)
print("BƯỚC 3: LƯU MÔ HÌNH")
print("="*70)

# Tạo thư mục models/ nếu chưa tồn tại
models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"[*] Đã tạo thư mục: {models_dir}")

# Tạo tên file với timestamp
dt = str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(":", '_')
joblib_file = os.path.join(models_dir, f'JS_Monolanguage_DT_{dt}.pkl')

# Lưu model
joblib.dump(classifier_DT, joblib_file)

print(f"[✓] Đã lưu mô hình vào: {joblib_file}")
print(f"    - Model type: Decision Tree Classifier (NPM Mono-Language)")
print(f"    - Training samples: {len(X)} NPM packages")
print(f"    - Hyperparameters: {hyperparams}")
print(f"    - Training accuracy: {classifier_DT.score(X, Y)*100:.2f}%")

# ============================================================================
# HOÀN TẤT
# ============================================================================

print("\n" + "="*70)
print("HOÀN TẤT!")
print("="*70)
print(f"[✓] Tổng thời gian thực thi: {elapsed_time:.2f} seconds")
print(f"[✓] File model đã được lưu tại: {joblib_file}")
print(f"[✓] Model này chỉ được train trên NPM (JavaScript) packages")
print("="*70)
