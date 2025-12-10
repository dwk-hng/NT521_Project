"""
Script Huấn Luyện và Kiểm Tra Decision Tree Đơn Ngôn Ngữ Python
Huấn luyện mô hình phân loại Decision Tree chỉ trên các gói PyPI (Python)
Sử dụng Bayesian Optimization để tối ưu hóa siêu tham số
"""

import pandas as pd
import joblib
import datetime
import time
import psutil
import os
from utilities_functions import *

def main():
    """
    Hàm chính để huấn luyện và đánh giá Decision Tree trên các gói PyPI
    """
    print("=" * 80)
    print("HUẤN LUYỆN VÀ KIỂM TRA DECISION TREE ĐƠN NGÔN NGỮ PYTHON")
    print("=" * 80)
    print()
    
    # Tải tập dữ liệu
    print("✓ Đang tải tập dữ liệu...")
    try:
        data = pd.read_csv('./Labelled_Dataset.csv', sep=',')
        print(f"  Tổng số gói đã tải: {len(data)}")
    except FileNotFoundError:
        print("✗ Lỗi: Không tìm thấy file Labelled_Dataset.csv!")
        print("  Vui lòng đảm bảo file tồn tại trong thư mục cross-language-detection-artifacts/")
        return
    
    # Lọc chỉ các gói PyPI
    print("\n✓ Đang lọc chỉ các gói PyPI...")
    data = data[data['Package Repository'] == 'PyPI']
    print(f"  Số gói PyPI: {len(data)}")
    print(f"  Độc hại: {len(data[data['Malicious'] == 1])}")
    print(f"  Lành tính: {len(data[data['Malicious'] == 0])}")
    
    if len(data) == 0:
        print("✗ Lỗi: Không tìm thấy gói PyPI nào trong tập dữ liệu!")
        return
    
    # Huấn luyện và đánh giá hiệu suất
    print("\n" + "=" * 80)
    print("HUẤN LUYỆN DECISION TREE VỚI BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print()
    
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 2)  # Bộ nhớ ban đầu tính bằng MB
    
    print("Bắt đầu cross-validation 10 fold với Bayesian Optimization...")
    performance, hyperparams = evaluation_decision_tree(data)
    
    final_memory = process.memory_info().rss / (1024 ** 2)    # Bộ nhớ cuối cùng tính bằng MB
    memory_usage = final_memory - initial_memory
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH HUẤN LUYỆN")
    print("=" * 80)
    print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")
    print(f"Bộ nhớ sử dụng: {memory_usage:.2f} MB")
    print()
    
    # Hiển thị các siêu tham số tốt nhất
    print("=" * 80)
    print("SIÊU THAM SỐ TỐI ƯU")
    print("=" * 80)
    for param, value in hyperparams.items():
        print(f"  {param}: {value}")
    print()
    
    # Hiển thị các chỉ số hiệu suất
    print("=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ TỪ 10-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print("[!] LƯU Ý: Các kết quả dưới đây là ước lượng hiệu suất của mô hình")
    print("    trên dữ liệu mới (chưa thấy) dựa trên 10 lần train-test khác nhau.")
    print("    Đây KHÔNG phải kết quả của mô hình cuối cùng được lưu.")
    print("=" * 80)
    print(f"Độ chính xác (Precision)  : {performance.iloc[-16, 0]:.2f} ± {performance.iloc[-16, 1]:.2f}%")
    print(f"Độ thu hồi (Recall)       : {performance.iloc[-15, 0]:.2f} ± {performance.iloc[-15, 1]:.2f}%")
    print(f"Điểm F1 (F1-Score)        : {performance.iloc[-14, 0]:.2f} ± {performance.iloc[-14, 1]:.2f}%")
    print(f"Độ chính xác (Accuracy)   : {performance.iloc[-13, 0]:.2f} ± {performance.iloc[-13, 1]:.2f}%")
    print()
    print("Chỉ số Ma trận nhầm lẫn:")
    print(f"  Dương tính giả (False Positive) : {performance.iloc[-12, 0]:.0f} ± {performance.iloc[-12, 1]:.0f}")
    print(f"  Âm tính giả (False Negative)    : {performance.iloc[-11, 0]:.0f} ± {performance.iloc[-11, 1]:.0f}")
    print(f"  Âm tính thật (True Negative)    : {performance.iloc[-10, 0]:.0f} ± {performance.iloc[-10, 1]:.0f}")
    print(f"  Dương tính thật (True Positive) : {performance.iloc[-9, 0]:.0f} ± {performance.iloc[-9, 1]:.0f}")
    print()
    
    # Huấn luyện mô hình cuối cùng trên toàn bộ tập dữ liệu
    print("=" * 80)
    print("HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG")
    print("=" * 80)
    print()
    
    X = data.drop(labels=['Package Repository', 'Malicious', 'Package Name'], axis=1).values
    Y = data['Malicious'].astype('int').values
    
    print(f"✓ Đang huấn luyện Decision Tree trên toàn bộ tập dữ liệu ({len(X)} mẫu)...")
    classifier_DT = DecisionTreeClassifier(
        random_state=123,
        criterion=hyperparams['criterion'],
        max_depth=hyperparams['max_depth'],
        max_features=hyperparams['max_features'],
        min_samples_leaf=hyperparams['min_sample_leaf'],
        min_samples_split=hyperparams['min_sample_split']
    )
    classifier_DT.fit(X=X, y=Y)
    print("  Hoàn thành huấn luyện mô hình!")
    
    # Lưu mô hình
    models_dir = './models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"[*] Đã tạo thư mục: {models_dir}")
    
    dt = str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(":", '_')
    joblib_file = os.path.join(models_dir, f'Py_Monolanguage_DT_{dt}.pkl')
    
    print(f"\n✓ Đang lưu mô hình vào: {joblib_file}")
    joblib.dump(classifier_DT, joblib_file)
    print("  Lưu mô hình thành công!")
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH TẤT CẢ CÁC THAO TÁC")
    print("=" * 80)

if __name__ == "__main__":
    main()
