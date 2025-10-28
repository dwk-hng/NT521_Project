# run_npm_test.py
import os
# Do file test và file extractor ở cùng thư mục, ta import trực tiếp
from npm_feature_extractor import NPM_Feature_Extractor

# Tên thư mục chứa các package mẫu (vẫn là thư mục con so với vị trí hiện tại)
SAMPLE_DIR = "npm_samples"

if not os.path.exists(SAMPLE_DIR):
    print(f"Lỗi: Không tìm thấy thư mục '{SAMPLE_DIR}'.")
    print("Hãy chắc chắn bạn đã tạo nó và tải package mẫu vào đó.")
else:
    print("="*50)
    print("Bắt đầu quá trình 'Smoke Test'...")
    print("="*50)
    
    # 1. Khởi tạo Feature Extractor
    npm_fe = NPM_Feature_Extractor()
    
    # 2. Chạy pipeline trích xuất đặc trưng
    try:
        print(f"[*] Đang quét các package trong thư mục: '{SAMPLE_DIR}'")
        features_df = npm_fe.extract_features(SAMPLE_DIR)
        
        print("\n[THÀNH CÔNG] Quá trình trích xuất đặc trưng đã hoàn tất không có lỗi.")
        # File output sẽ được tạo ra ngay trong thư mục feature_extraction
        print(f"[*] Kết quả đã được lưu vào file: npm_feature_extracted.csv")
        
        # 3. In ra một vài thông tin để kiểm tra nhanh
        print("\n[*] Thông tin DataFrame kết quả:")
        features_df.info()
        
        print("\n[*] 5 dòng đầu tiên của dữ liệu:")
        print(features_df.head())
        
    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra sự cố trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("Smoke Test kết thúc.")
print("="*50)
