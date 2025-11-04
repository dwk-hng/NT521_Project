import os
import pandas as pd
from pypi_feature_extractor import PyPI_Feature_Extractor # Import lớp extractor

# Thư mục chứa các package mẫu PyPI
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "pypi_samples")

def run_pypi_extraction_test():
    """
    Hàm để chạy quá trình trích xuất đặc trưng cho các package PyPI mẫu.
    """
    if not os.path.exists(SAMPLE_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{SAMPLE_DIR}'.")
        print("Hãy chắc chắn bạn đã tạo nó và tải package mẫu PyPI vào đó.")
        return

    print("="*50)
    print("Bắt đầu quá trình 'Smoke Test' cho PyPI Feature Extraction...")
    print("="*50)
    
    # 1. Khởi tạo Feature Extractor cho PyPI
    pypi_fe = PyPI_Feature_Extractor()
    
    # 2. Chạy pipeline trích xuất đặc trưng
    try:
        print(f"[*] Đang quét và trích xuất đặc trưng từ các package trong thư mục: '{SAMPLE_DIR}'")
        features_df = pypi_fe.extract_features(SAMPLE_DIR)
        
        print("\n[THÀNH CÔNG] Quá trình trích xuất đặc trưng PyPI đã hoàn tất không có lỗi.")
        
        # File output sẽ được tạo ra ngay trong thư mục cha của feature_extraction
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pypi_feature_extracted.csv")
        print(f"[*] Kết quả đã được lưu vào file: {os.path.abspath(output_path)}")
        
        # 3. In ra một vài thông tin để kiểm tra nhanh
        print("\n[*] Thông tin DataFrame kết quả PyPI:")
        features_df.info(verbose=True, show_counts=True) # verbose=True để hiển thị tất cả cột
        
        print("\n[*] 5 dòng đầu tiên của dữ liệu PyPI:")
        print(features_df.head())
        
        print("\n[*] Thống kê mô tả các cột số của dữ liệu PyPI:")
        print(features_df.describe())
        
    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra sự cố trong quá trình chạy PyPI extractor: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("Smoke Test PyPI kết thúc.")
    print("="*50)

if __name__ == "__main__":
    run_pypi_extraction_test()