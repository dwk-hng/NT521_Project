import pandas as pd
import joblib
import os
import glob
import xgboost as xgb
import numpy as np

def load_all_models(models_dir):
    """
    Load tất cả các model trong thư mục models và phân loại chúng.
    Trả về dict: {model_name: model_object}
    """
    models = {}
    pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    
    if not pkl_files:
        print(f"[!] Không tìm thấy model nào trong {models_dir}")
        return models
    
    print(f"[*] Tìm thấy {len(pkl_files)} model(s)")
    
    for model_path in pkl_files:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        print(f"    - Loading: {model_name}")
        
        try:
            model = joblib.load(model_path)
            models[model_name] = model
        except Exception as e:
            print(f"      [!] Lỗi khi load {model_name}: {e}")
            
    return models

def prepare_data(df):
    """
    Xử lý dữ liệu DataFrame để chuẩn bị cho model.
    Trả về: (package_info, X)
    """
    # Lưu lại thông tin gói và repository (nếu có)
    info_columns = ['Package Name']
    if 'repository' in df.columns:
        info_columns.append('repository')
    package_info = df[info_columns].copy()
    
    # Tạo bản sao để xử lý
    X = df.copy()
    
    # Bỏ cột Package Name
    if 'Package Name' in X.columns:
        X = X.drop('Package Name', axis=1)
        
    # Bỏ cột repository (nếu có)
    if 'repository' in X.columns:
        X = X.drop('repository', axis=1)
        
    # Đảm bảo không còn cột nào khác không phải feature
    if 'Malicious' in X.columns:
        X = X.drop('Malicious', axis=1)
    if 'Package Repository' in X.columns:
        X = X.drop('Package Repository', axis=1)

    # Fill NaN nếu có
    X = X.fillna(0)
    
    return package_info, X

def predict_with_model(model, X):
    """
    Thực hiện dự đoán với model.
    Trả về: (predictions, probabilities)
    """
    predictions = model.predict(X.values)
    
    # Lấy xác suất (nếu model hỗ trợ)
    try:
        probabilities = model.predict_proba(X.values)[:, 1]
    except:
        probabilities = [None] * len(predictions)
    
    return predictions, probabilities

def process_and_predict(model, model_name, data_df, output_path, data_type=""):
    """
    Xử lý dữ liệu và thực hiện dự đoán với một model cụ thể.
    """
    try:
        print(f"    - Đang dự đoán với {model_name} ({data_type})...")
        
        # Chuẩn bị dữ liệu
        package_info, X = prepare_data(data_df)
        
        # Dự đoán
        predictions, probabilities = predict_with_model(model, X)
        
        # Tạo DataFrame kết quả
        result_df = package_info.copy()
        result_df['Prediction'] = ['Malicious' if p == 1 else 'Benign' for p in predictions]
        result_df['Malicious_Probability'] = probabilities
        
        # Thêm cột repository type nếu có
        if 'repository' in result_df.columns:
            result_df['Package_Type'] = result_df['repository'].apply(
                lambda x: 'NPM' if x == 1 else ('PyPI' if x == 2 else 'Unknown')
            )
        
        # Lưu kết quả
        result_df.to_csv(output_path, index=False)
        
        # Thống kê
        malicious_count = result_df[result_df['Prediction'] == 'Malicious'].shape[0]
        benign_count = len(result_df) - malicious_count
        print(f"      ✓ Lưu: {output_path}")
        print(f"      ✓ Kết quả: {malicious_count} Malicious, {benign_count} Benign")
        
        return True
        
    except Exception as e:
        print(f"      [!] Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_npm_pypi_data(npm_file, pypi_file):
    """
    Gộp dữ liệu NPM và PyPI thành một DataFrame duy nhất.
    Thêm cột 'repository' để phân biệt: 1=NPM, 2=PyPI
    """
    data_frames = []
    
    if os.path.exists(npm_file):
        npm_df = pd.read_csv(npm_file)
        npm_df['repository'] = 1  # NPM
        data_frames.append(npm_df)
        print(f"    - Đọc NPM: {len(npm_df)} gói")
    else:
        print(f"    [!] Không tìm thấy file NPM: {npm_file}")
    
    if os.path.exists(pypi_file):
        pypi_df = pd.read_csv(pypi_file)
        pypi_df['repository'] = 2  # PyPI
        data_frames.append(pypi_df)
        print(f"    - Đọc PyPI: {len(pypi_df)} gói")
    else:
        print(f"    [!] Không tìm thấy file PyPI: {pypi_file}")
    
    if not data_frames:
        return None
    
    merged_df = pd.concat(data_frames, ignore_index=True)
    print(f"    - Tổng cộng: {len(merged_df)} gói")
    
    return merged_df

def main():
    # Đường dẫn cơ sở
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    feature_dir = os.path.join(base_dir, 'feature_extraction')
    
    # Tạo thư mục results nếu chưa có
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    print("="*70)
    print("CHƯƠNG TRÌNH ĐÁNH GIÁ ĐỘC HẠI CỦA GÓI PHẦN MỀM")
    print("="*70)
    
    # 1. Load tất cả models
    print("\n[BƯỚC 1] Load tất cả models...")
    models = load_all_models(models_dir)
    
    if not models:
        print("[!] Không có model nào để chạy. Dừng chương trình.")
        return
    
    # 2. Đọc dữ liệu
    print("\n[BƯỚC 2] Đọc dữ liệu features...")
    npm_file = os.path.join(feature_dir, 'npm_feature_extracted.csv')
    pypi_file = os.path.join(feature_dir, 'pypi_feature_extracted.csv')
    
    npm_df = pd.read_csv(npm_file) if os.path.exists(npm_file) else None
    pypi_df = pd.read_csv(pypi_file) if os.path.exists(pypi_file) else None
    
    if npm_df is not None:
        print(f"    - NPM: {len(npm_df)} gói")
    if pypi_df is not None:
        print(f"    - PyPI: {len(pypi_df)} gói")
    
    # 3. Gộp dữ liệu cho CrossLanguage models
    print("\n[BƯỚC 3] Chuẩn bị dữ liệu cho CrossLanguage models...")
    merged_df = merge_npm_pypi_data(npm_file, pypi_file)
    
    # 4. Chạy dự đoán với từng model
    print("\n[BƯỚC 4] Thực hiện dự đoán với tất cả models...")
    print("="*70)
    
    total_success = 0
    total_models = len(models)
    
    for model_name, model in models.items():
        print(f"\n[MODEL {total_success + 1}/{total_models}] {model_name}")
        
        # Xác định loại model
        is_crosslanguage = 'CrossLanguage' in model_name or 'Crosslanguage' in model_name
        is_npm = 'JS' in model_name or 'NPM' in model_name
        is_pypi = 'Py' in model_name or 'PyPI' in model_name or 'Python' in model_name
        
        if is_crosslanguage:
            # CrossLanguage: dùng dữ liệu đã gộp
            if merged_df is not None:
                output_file = os.path.join(results_dir, f'{model_name}_results.csv')
                success = process_and_predict(model, model_name, merged_df, output_file, "NPM + PyPI")
                if success:
                    total_success += 1
            else:
                print("    [!] Không có dữ liệu để xử lý")
                
        elif is_npm:
            # MonoLanguage NPM: chỉ dùng dữ liệu NPM
            if npm_df is not None:
                output_file = os.path.join(results_dir, f'{model_name}_results.csv')
                success = process_and_predict(model, model_name, npm_df, output_file, "NPM only")
                if success:
                    total_success += 1
            else:
                print("    [!] Không có dữ liệu NPM")
                
        elif is_pypi:
            # MonoLanguage PyPI: chỉ dùng dữ liệu PyPI
            if pypi_df is not None:
                output_file = os.path.join(results_dir, f'{model_name}_results.csv')
                success = process_and_predict(model, model_name, pypi_df, output_file, "PyPI only")
                if success:
                    total_success += 1
            else:
                print("    [!] Không có dữ liệu PyPI")
        else:
            # Model không xác định: thử với dữ liệu gộp
            print("    [?] Không xác định được loại model, thử với dữ liệu gộp...")
            if merged_df is not None:
                output_file = os.path.join(results_dir, f'{model_name}_results.csv')
                success = process_and_predict(model, model_name, merged_df, output_file, "All data")
                if success:
                    total_success += 1
    
    # 5. Tổng kết
    print("\n" + "="*70)
    print("HOÀN TẤT!")
    print("="*70)
    print(f"✓ Đã xử lý thành công: {total_success}/{total_models} models")
    print(f"✓ Kiểm tra thư mục '{results_dir}' để xem chi tiết kết quả.")
    print("="*70)

if __name__ == "__main__":
    main()
