import os
import re
import json
import tarfile
import zipfile
import shutil
import warnings
from pathlib import Path
from functools import reduce
from typing import Optional
import ast

import pandas as pd
import numpy as np
import statistics

from pygments.lexers import PythonLexer
from pygments.token import Token
from urlextract import URLExtract

import utilities_functions

detect_encoding = utilities_functions.detect_encoding
contains_base64 = utilities_functions.contains_base64
contains_IPAddress = utilities_functions.contains_IPAddress
contains_URL = utilities_functions.contains_URL
contains_dangerous_token = utilities_functions.contains_dangerous_token
shannon_entropy = utilities_functions.shannon_entropy
obfuscation = utilities_functions.obfuscation
gen_language_4 = utilities_functions.gen_language_4
find_files_of_ext = utilities_functions.find_files_of_ext

class PyPI_Feature_Extractor:
    """
    Lớp chính thực hiện pipeline trích xuất features cho tập PyPI packages.
    Các phương thức tương tự NPM_Feature_Extractor để giữ tính nhất quán.
    """

    # Định nghĩa các phần mở rộng của file nén ở cấp class
    ARCHIVE_EXTENSIONS = (".tar.gz", ".zip", ".whl")

    def __init__(self) -> None:
        # Danh sách extension đầy đủ cho PyPI (tương thích với cross-language dataset)
        self.classes = ['bat',	'bz2',	'c', 'cert','conf','cpp' ,'crt', 'css',	'csv', 'deb' ,'erb','gemspec', 'gif', 'gz', 'h', 'html', 'ico' ,'ini' ,'jar', 'java', 'jpg', 'js', 'json', 'key' ,'m4v' ,'markdown' ,'md' ,'pdf', 'pem', 'png', 'ps', 'py',	'rb', 'rpm', 'rst','sh'	,'svg',	'toml',	'ttf',	'txt','xml', 'yaml', 'yml', 'eot', 'exe', 'jpeg', 'properties',	'sql',	'swf',	'tar',	'woff', 'woff2', 'aac','bmp', 'cfg' ,'dcm', 'dll', 'doc', 'flac','flv',	'ipynb', 'm4a', 'mid', 'mkv', 'mp3', 'mp4', 'mpg', 'ogg','otf', 'pickle', 'pkl' ,'psd',	'pxd' ,'pxi', 'pyc', 'pyx', 'r', 'rtf',	'so', 'sqlite' ,'tif',	'tp', 'wav', 'webp' ,'whl', 'xcf', 'xz', 'zip' ,'mov' ,'wasm', 'webm']

        # Đường dẫn gốc để quét (được gán khi gọi extract_features)
        self.path_to_scan: Optional[str] = None

        # Tải danh sách dangerous token (nếu file không tồn tại thì dùng danh sách rỗng)
        try:
            here = Path(__file__).parent
            with open(here / 'resources' / 'dangerous_tokens.json', 'r', encoding='utf-8') as fh:
                self.dangerous_token = json.load(fh)
        except Exception as e:
            print(f"[WARN] Could not load dangerous_tokens.json: {e}. Using empty list.")
            self.dangerous_token = []

        self.url_extractor = URLExtract()

    def extract_features(self, path: str) -> pd.DataFrame:
        """
        Entrypoint chính để trích xuất feature cho các package PyPI.
        - path: thư mục chứa các file nén (.tar.gz, .zip, .whl) hoặc đã giải nén packages.
        - Trả về pandas.DataFrame với tất cả các feature đã tính và cũng lưu file CSV.
        """
        self.path_to_scan = path
        print(f"[*] PyPI extractor starting. Scanning directory: '{path}'")

        # Giải nén .tar.gz, .zip, .whl (không ghi đè nếu đã giải nén)
        self.unzip_packages()

        # Trích xuất feature từ mã nguồn Python (.py files)
        py_files_df = self.extract_feature_from_py()

        # Trích xuất feature từ các file metadata (setup.py, pyproject.toml, setup.cfg)
        metadata_df = self.extract_feature_from_setup()

        # Đếm số lượng file theo extension cho mỗi package
        extensions_files_df = self.count_package_files_extension()

        # Gộp các DataFrame theo 'Package Name' (outer join để giữ tất cả các package)
        dfs = [py_files_df, metadata_df, extensions_files_df]
        dfs = [df for df in dfs if not df.empty]  # Chỉ giữ các DataFrame không rỗng
        
        if len(dfs) > 1:
            final_df = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), dfs)
        elif len(dfs) == 1:
            final_df = dfs[0]
        else:
            # Trường hợp không có dữ liệu nào
            final_df = pd.DataFrame(columns=['Package Name'])
        
        # Fill NA cho các cột numeric (không fill cho strings/identifiers)
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

        final_df = self.extraction(final_df, gen_language_4, 4, gen_language_4, 4)

        out_csv_path = Path(self.path_to_scan).parent / "pypi_feature_extracted.csv"
        final_df.to_csv(out_csv_path, encoding='utf-8', index=False)
        print(f"[*] PyPI features extracted successfully. Saved to: '{out_csv_path}'")
        return final_df

    def unzip_packages(self) -> None:
        """
        Duyệt cây thư mục self.path_to_scan, tìm file nén (.tar.gz, .zip, .whl) và giải nén.
        - Chỉ giải nén khi file thực sự tồn tại và có kích thước > 0.
        - Tạo thư mục con cùng tên file (bỏ extension) để giải nén vào đó.
        """
        if not self.path_to_scan:
            return

        for root, dirs, files in os.walk(self.path_to_scan):
            for fname in files:
                lower_fname = fname.lower()
                if lower_fname.endswith(self.ARCHIVE_EXTENSIONS): # Sử dụng ARCHIVE_EXTENSIONS
                    fullpath = os.path.join(root, fname)
                    try:
                        if os.path.getsize(fullpath) > 0:
                            dest_dir_name = Path(fname).stem
                            if dest_dir_name.endswith(".tar"): # Xử lý .tar.gz
                                dest_dir_name = Path(dest_dir_name).stem
                            dest_dir = Path(fullpath).parent / dest_dir_name
                            
                            if not os.path.exists(dest_dir):
                                os.makedirs(dest_dir, exist_ok=True)
                                print(f"[*] Extracting '{fullpath}' to '{dest_dir}'")
                                if lower_fname.endswith(".tar.gz"):
                                    with tarfile.open(fullpath, 'r:gz') as pkg:
                                        pkg.extractall(path=dest_dir)
                                else:
                                    with zipfile.ZipFile(fullpath, 'r') as pkg:
                                        pkg.extractall(path=dest_dir)
                    except Exception as e:
                        print(f"[WARN] Error extracting '{fullpath}': {e}")
                        continue

    def extract_feature_from_py(self) -> pd.DataFrame:
        """
        Trích xuất feature từ tất cả file .py bao gồm cả các ratio (plus, equal, bracket).
        """
        files_path = find_files_of_ext(self.path_to_scan, ".py")

        Package = []
        pyfile = []
        strings_list = []
        identifiers_list = []
        suspicious_tokens = []
        lines = []
        num_chars = []
        base64_counts = []
        ip_counts = []
        url_counts_in_strings = []
        email_counts_in_file = []
        plus_ratio = []
        equal_ratio = []
        square_ratio = []
        code_list = []

        print(f"[*] Found {len(files_path)} Python files.")

        for fp in files_path:
            p = Path(fp)
            # BỎ QUA CÁC FILE NÉN GỐC Ở ĐÂY
            if p.parent == Path(self.path_to_scan) and p.name.lower().endswith(self.ARCHIVE_EXTENSIONS):
                continue

            try:
                try:
                    package_name = str(p.relative_to(self.path_to_scan).parts[0])
                except ValueError:
                    package_name = p.parent.name
                
                py_name = p.name

                with open(fp, "rb") as f:
                    raw_data = f.read()

                encoding = detect_encoding(raw_data)
                data = raw_data.decode(encoding, errors='ignore')

                lexer = PythonLexer(stripnl=False, ensurenl=False)
                token_source = lexer.get_tokens(data)

                ids = []
                strs = []
                operator = []
                punctuation = []

                for tok_type, tok_val in token_source:
                    if tok_type in Token.Operator:
                        operator.append(tok_val)
                    elif tok_type in Token.Punctuation:
                        punctuation.append(tok_val)
                    elif tok_type in Token.Name:
                        if tok_val and not tok_val.isspace():
                            ids.append(tok_val)
                    elif tok_type in (Token.Literal.String.Single, Token.Literal.String.Double,
                                      Token.Literal.String.Heredoc, Token.Literal.String):
                        if tok_val and not tok_val.isspace():
                            strs.append(tok_val)

                num_lines = data.count("\n") + 1
                size_chars = len(data) if len(data) > 0 else 1

                # Tính toán các ratio cho operators và punctuation
                equalities = operator.count('=') / size_chars
                plus = operator.count('+') / size_chars
                Lbrackets = punctuation.count('[') / size_chars

                # Làm sạch ids và strs
                ids = [s.replace("'", '').replace('"', '') for s in ids]
                strs = [s.replace("'", '').replace('"', '') for s in strs]

                count_base64_in_strs = sum(len(contains_base64(s)) for s in strs)
                count_ip_in_strs = sum(len(contains_IPAddress(s)) for s in strs)
                count_url_in_strs = 0  # Sẽ tính trong extraction()
                count_emails = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', data))
                count_dangerous = len(contains_dangerous_token(data, self.dangerous_token))

                ids_join = ' '.join(ids)
                string_join = ' '.join(strs)

                Package.append(package_name)
                pyfile.append(py_name)
                suspicious_tokens.append(count_dangerous)
                lines.append(num_lines)
                num_chars.append(size_chars)
                base64_counts.append(count_base64_in_strs)
                ip_counts.append(count_ip_in_strs)
                url_counts_in_strings.append(count_url_in_strs)
                email_counts_in_file.append(count_emails)
                plus_ratio.append(plus)
                equal_ratio.append(equalities)
                square_ratio.append(Lbrackets)
                identifiers_list.append(ids_join)
                strings_list.append(string_join)
                code_list.append(data)

            except Exception as e:
                print(f"[WARN] Error processing Python file '{fp}': {e}")
                continue

        db = pd.DataFrame({
            'Package Name': Package,
            'py_file_name': pyfile,
            'suspicious_tokens_py': suspicious_tokens,
            'lines_py': lines,
            'num_chars_py': num_chars,
            'base64_py': base64_counts,
            'IP_py': ip_counts,
            'URLs_in_strings_py': url_counts_in_strings,
            'emails_in_file_py': email_counts_in_file,
            'plus_ratio': plus_ratio,
            'equal_ratio': equal_ratio,
            'bracket_ratio': square_ratio,
            'identifiers_py': identifiers_list,
            'strings_py': strings_list,
            'code_py': code_list
        })

        return self.merge_py_of_same_package(db)

    def merge_py_of_same_package(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp nhiều file .py của cùng 1 package thành 1 record package-level với đầy đủ ratio features.
        """
        if database.empty:
            cols = ['Package Name','Number of words','lines','plus ratio mean','plus ratio max','plus ratio std','plus ratio q3',
                    'eq ratio mean','eq ratio max','eq ratio std','eq ratio q3','bracket ratio mean','bracket ratio max',
                    'bracket ratio std','bracket ratio q3','base64','IP','sospicious token','strings','identifiers']
            return pd.DataFrame(columns=cols)

        p_database_code = database.groupby(['Package Name'], as_index=False)['code_py'].agg('\n'.join)
        p_database_code['Number of words'] = p_database_code["code_py"].apply(lambda n: len(n.split()))

        l_database = database.groupby(['Package Name'], as_index=False)['lines_py'].sum()
        
        # Tính toán mean, max, std, q3 cho plus ratio
        plus_mean = database.groupby(['Package Name'], as_index=False)['plus_ratio'].mean()
        plus_mean = plus_mean.rename(columns={"plus_ratio": "plus ratio mean"})
        plus_max = database.groupby(['Package Name'], as_index=False)['plus_ratio'].max()
        plus_max = plus_max.rename(columns={"plus_ratio": "plus ratio max"})
        plus_std = database.groupby(['Package Name'], as_index=False)['plus_ratio'].std()
        plus_std = plus_std.rename(columns={"plus_ratio": "plus ratio std"})
        plus_q3 = database.groupby(['Package Name'], as_index=False)['plus_ratio'].quantile(0.75)
        plus_q3 = plus_q3.rename(columns={"plus_ratio": "plus ratio q3"})
        
        # Tính toán mean, max, std, q3 cho equal ratio
        eq_mean = database.groupby(['Package Name'], as_index=False)['equal_ratio'].mean()
        eq_mean = eq_mean.rename(columns={"equal_ratio": "eq ratio mean"})
        eq_max = database.groupby(['Package Name'], as_index=False)['equal_ratio'].max()
        eq_max = eq_max.rename(columns={"equal_ratio": "eq ratio max"})
        eq_std = database.groupby(['Package Name'], as_index=False)['equal_ratio'].std()
        eq_std = eq_std.rename(columns={"equal_ratio": "eq ratio std"})
        eq_q3 = database.groupby(['Package Name'], as_index=False)['equal_ratio'].quantile(0.75)
        eq_q3 = eq_q3.rename(columns={"equal_ratio": "eq ratio q3"})
        
        # Tính toán mean, max, std, q3 cho bracket ratio
        bracket_mean = database.groupby(['Package Name'], as_index=False)['bracket_ratio'].mean()
        bracket_mean = bracket_mean.rename(columns={"bracket_ratio": "bracket ratio mean"})
        bracket_max = database.groupby(['Package Name'], as_index=False)['bracket_ratio'].max()
        bracket_max = bracket_max.rename(columns={"bracket_ratio": "bracket ratio max"})
        bracket_std = database.groupby(['Package Name'], as_index=False)['bracket_ratio'].std()
        bracket_std = bracket_std.rename(columns={"bracket_ratio": "bracket ratio std"})
        bracket_q3 = database.groupby(['Package Name'], as_index=False)['bracket_ratio'].quantile(0.75)
        bracket_q3 = bracket_q3.rename(columns={"bracket_ratio": "bracket ratio q3"})
        
        base = database.groupby(['Package Name'], as_index=False)['base64_py'].sum()
        ip = database.groupby(['Package Name'], as_index=False)['IP_py'].sum()
        sospicious = database.groupby(['Package Name'], as_index=False)['suspicious_tokens_py'].sum()
        
        string = database.groupby(['Package Name'], as_index=False)['strings_py'].agg(' '.join)
        identifier = database.groupby(['Package Name'], as_index=False)['identifiers_py'].agg(' '.join)

        data_to_merge = [p_database_code, l_database, plus_mean, plus_max, plus_std, plus_q3,
                        eq_mean, eq_max, eq_std, eq_q3, bracket_mean, bracket_max, bracket_std, bracket_q3,
                        base, ip, sospicious, string, identifier]
        final_database = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), data_to_merge)

        if 'code_py' in final_database.columns:
            final_database.drop('code_py', axis=1, inplace=True)
        
        final_database.columns = ['Package Name','Number of words','lines','plus ratio mean','plus ratio max','plus ratio std','plus ratio q3',
                                  'eq ratio mean','eq ratio max','eq ratio std','eq ratio q3','bracket ratio mean','bracket ratio max',
                                  'bracket ratio std','bracket ratio q3','base64','IP','sospicious token','strings','identifiers']
        return final_database

    def extract_feature_from_setup(self) -> pd.DataFrame:
        """
        Trích xuất features từ file setup.py (tương tự extract_feature_from_py nhưng chỉ cho setup.py).
        """
        files_path = find_files_of_ext(self.path_to_scan, ".py")
        
        # Lọc chỉ lấy file setup.py
        setup_files = [fp for fp in files_path if Path(fp).name == 'setup.py']

        Package = []
        pyfile = []
        strings_list = []
        identifiers_list = []
        suspicious_tokens = []
        lines = []
        num_chars = []
        base64_counts = []
        ip_counts = []
        url_counts_in_strings = []
        email_counts_in_file = []
        plus_ratio = []
        equal_ratio = []
        square_ratio = []
        code_list = []

        print(f"[*] Found {len(setup_files)} setup.py files.")

        for fp in setup_files:
            p = Path(fp)
            # BỎ QUA CÁC FILE NÉN GỐC
            if p.parent == Path(self.path_to_scan) and p.name.lower().endswith(self.ARCHIVE_EXTENSIONS):
                continue

            try:
                try:
                    package_name = str(p.relative_to(self.path_to_scan).parts[0])
                except ValueError:
                    package_name = p.parent.name
                
                py_name = p.name

                with open(fp, "rb") as f:
                    raw_data = f.read()

                encoding = detect_encoding(raw_data)
                data = raw_data.decode(encoding, errors='ignore')

                lexer = PythonLexer(stripnl=False, ensurenl=False)
                token_source = lexer.get_tokens(data)

                ids = []
                strs = []
                operator = []
                punctuation = []

                for tok_type, tok_val in token_source:
                    if tok_type in Token.Operator:
                        operator.append(tok_val)
                    elif tok_type in Token.Punctuation:
                        punctuation.append(tok_val)
                    elif tok_type in Token.Name:
                        if tok_val and not tok_val.isspace():
                            ids.append(tok_val)
                    elif tok_type in (Token.Literal.String.Single, Token.Literal.String.Double,
                                      Token.Literal.String.Heredoc, Token.Literal.String):
                        if tok_val and not tok_val.isspace():
                            strs.append(tok_val)

                num_lines = data.count("\n") + 1
                size_chars = len(data) if len(data) > 0 else 1

                # Tính toán các ratio
                equalities = operator.count('=') / size_chars
                plus = operator.count('+') / size_chars
                Lbrackets = punctuation.count('[') / size_chars

                # Làm sạch
                ids = [s.replace("'", '').replace('"', '') for s in ids]
                strs = [s.replace("'", '').replace('"', '') for s in strs]

                count_base64_in_strs = sum(len(contains_base64(s)) for s in strs)
                count_ip_in_strs = sum(len(contains_IPAddress(s)) for s in strs)
                count_url_in_strs = 0
                count_emails = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', data))
                count_dangerous = len(contains_dangerous_token(data, self.dangerous_token))

                ids_join = ' '.join(ids)
                string_join = ' '.join(strs)

                Package.append(package_name)
                pyfile.append(py_name)
                suspicious_tokens.append(count_dangerous)
                lines.append(num_lines)
                num_chars.append(size_chars)
                base64_counts.append(count_base64_in_strs)
                ip_counts.append(count_ip_in_strs)
                url_counts_in_strings.append(count_url_in_strs)
                email_counts_in_file.append(count_emails)
                plus_ratio.append(plus)
                equal_ratio.append(equalities)
                square_ratio.append(Lbrackets)
                identifiers_list.append(ids_join)
                strings_list.append(string_join)
                code_list.append(data)

            except Exception as e:
                print(f"[WARN] Error processing setup.py file '{fp}': {e}")
                continue

        db = pd.DataFrame({
            'Package Name': Package,
            'py_file_name': pyfile,
            'suspicious_tokens_py': suspicious_tokens,
            'lines_py': lines,
            'num_chars_py': num_chars,
            'base64_py': base64_counts,
            'IP_py': ip_counts,
            'URLs_in_strings_py': url_counts_in_strings,
            'emails_in_file_py': email_counts_in_file,
            'plus_ratio': plus_ratio,
            'equal_ratio': equal_ratio,
            'bracket_ratio': square_ratio,
            'identifiers_py': identifiers_list,
            'strings_py': strings_list,
            'code_py': code_list
        })

        return self.merge_setup_of_same_package(db)

    def merge_setup_of_same_package(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp các file setup.py của cùng 1 package (không có ratio features như .py files).
        """
        if database.empty:
            cols = ['Package Name','Number of words','lines','base64','IP','sospicious token','strings','identifiers']
            return pd.DataFrame(columns=cols)

        p_database_code = database.groupby(['Package Name'], as_index=False)['code_py'].agg('\n'.join)
        p_database_code['Number of words'] = p_database_code["code_py"].apply(lambda n: len(n.split()))
        
        l_database = database.groupby(['Package Name'], as_index=False)['lines_py'].sum()
        base = database.groupby(['Package Name'], as_index=False)['base64_py'].sum()
        ip = database.groupby(['Package Name'], as_index=False)['IP_py'].sum()
        sospicious = database.groupby(['Package Name'], as_index=False)['suspicious_tokens_py'].sum()
        string = database.groupby(['Package Name'], as_index=False)['strings_py'].agg(' '.join)
        identifier = database.groupby(['Package Name'], as_index=False)['identifiers_py'].agg(' '.join)

        data_to_merge = [p_database_code[['Package Name', 'Number of words']], l_database, base, ip, sospicious, 
                        string[['Package Name', 'strings_py']], identifier[['Package Name', 'identifiers_py']]]
        final_database = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), data_to_merge)
        
        # Rename columns
        final_database = final_database.rename(columns={
            'lines_py': 'lines',
            'base64_py': 'base64',
            'IP_py': 'IP',
            'suspicious_tokens_py': 'sospicious token',
            'strings_py': 'strings',
            'identifiers_py': 'identifiers'
        })
        
        final_database.columns = ['Package Name','Number of words','lines','base64','IP','sospicious token','strings','identifiers']
        return final_database



    def count_package_files_extension(self) -> pd.DataFrame:
        """
        Đếm số file theo mỗi extension trong self.classes cho từng package.
        """
        package_ext_counts = {}

        for ext_name in self.classes:
            ext_with_dot = '.' + ext_name
            files_path = find_files_of_ext(self.path_to_scan, ext_with_dot)
            for fpath in files_path:
                p = Path(fpath)
                # Kiểm tra nếu file này nằm trực tiếp trong thư mục quét gốc (self.path_to_scan)
                # Và nó có phần mở rộng là một trong các phần mở rộng file nén
                if p.parent == Path(self.path_to_scan) and p.name.lower().endswith(self.ARCHIVE_EXTENSIONS):
                    continue # Bỏ qua file nén gốc

                try:
                    package_name = str(p.relative_to(self.path_to_scan).parts[0])
                except ValueError:
                    package_name = p.parent.name
                
                if package_name not in package_ext_counts:
                    package_ext_counts[package_name] = {}
                package_ext_counts[package_name][ext_with_dot] = package_ext_counts[package_name].get(ext_with_dot, 0) + 1
        
        if not package_ext_counts:
            empty_cols = ['Package Name'] + ['.' + e for e in self.classes]
            return pd.DataFrame(columns=empty_cols)
        
        db = pd.DataFrame.from_dict(package_ext_counts, orient='index').reset_index()
        db = db.rename(columns={'index': 'Package Name'})

        extensions_cols = ['.' + c for c in self.classes]
        for col in extensions_cols:
            if col not in db.columns:
                db[col] = 0
        
        db = db[['Package Name'] + extensions_cols]
        return db

    def extraction(self, database: pd.DataFrame, alphabetic_string_func, base_string_val, alphabetic_id_func, base_id_val) -> pd.DataFrame:
        """
        Tính các feature nâng cao theo format của cross-language dataset.
        """
        # Thêm cột repository cho PyPI
        database['repository'] = pd.Series([2 for x in range(len(database.index))])
        
        # Fill NA cho các cột numeric
        f = [c for c in database.columns if c not in ['strings_x','identifiers_x','strings_y','identifiers_y']]
        database[f] = database[f].fillna(0)

        database.index = range(0, len(database))

        # Xác định tên cột strings và identifiers (có thể có suffix _x/_y hoặc không)
        if 'strings_x' in database.columns:
            source_code_strings = database['strings_x']
        elif 'strings' in database.columns:
            source_code_strings = database['strings']
        else:
            source_code_strings = pd.Series(['' for _ in range(len(database))])
        
        if 'identifiers_x' in database.columns:
            source_code_identifiers = database['identifiers_x']
        elif 'identifiers' in database.columns:
            source_code_identifiers = database['identifiers']
        else:
            source_code_identifiers = pd.Series(['' for _ in range(len(database))])
        
        if 'strings_y' in database.columns:
            metadata_strings = database['strings_y']
        else:
            metadata_strings = pd.Series(['' for _ in range(len(database))])
        
        if 'identifiers_y' in database.columns:
            metadata_identifiers = database['identifiers_y']
        else:
            metadata_identifiers = pd.Series(['' for _ in range(len(database))])
        
        repository = database['repository']
        check_metadata_strings = metadata_strings.isna()
        check_metadata_identifiers = metadata_identifiers.isna()
        check_source_code_strings = source_code_strings.isna()
        check_source_code_identifiers = source_code_identifiers.isna()

        shannon_q3_id_sc = []; shannon_q3_str_sc = []; shannon_mean_id_sc = []; shannon_mean_str_sc = []
        shannon_std_id_sc = []; shannon_std_str_sc = []; shannon_max_id_sc = []; shannon_max_str_sc = []
        homogeneous_id_sc = []; homogeneous_str_sc = []; url_count_sc = []; obf_id_sc = []; obf_str_sc = []
        heterogeneous_id_sc = []; heterogeneous_str_sc = []

        shannon_q3_id_md = []; shannon_q3_str_md = []; shannon_mean_id_md = []; shannon_mean_str_md = []
        shannon_std_id_md = []; shannon_std_str_md = []; shannon_max_id_md = []; shannon_max_str_md = []
        homogeneous_id_md = []; homogeneous_str_md = []; url_count_md = []; obf_id_md = []; obf_str_md = []
        heterogeneous_id_md = []; heterogeneous_str_md = []
        
        presence_of_installation_script = []

        for i in range(len(database)):
            package_name = database['Package Name'].iloc[i]
            print(f"[+] Calculating advanced features for package: '{package_name}'")
            
            # Xác định entry points dựa trên repository
            if repository[i] == 3:
                install = ['extensions']
            elif repository[i] == 2:
                install = ['install']
            else:
                install = ['postinstall', 'preinstall', 'install']
            
            # Source code processing
            if check_source_code_strings[i] == False:
                strings_sc = source_code_strings.iloc[i].split()
            else:
                strings_sc = []
            
            if check_source_code_identifiers[i] == False:
                identifiers_sc = source_code_identifiers.iloc[i].split()
            else:
                identifiers_sc = []

            generalized_id_sc = [alphabetic_id_func(x) for x in identifiers_sc]
            generalized_str_sc = [alphabetic_string_func(s) for s in strings_sc]

            obf_score_id_sc = obfuscation(generalized_id_sc, symbols=['u','d','l','s'])
            obf_score_str_sc = obfuscation(generalized_str_sc, symbols=['u','d','l','s'])

            shannon_vals_id_sc = [shannon_entropy(x, base_id_val) for x in generalized_id_sc] if generalized_id_sc else [0.0]
            shannon_vals_str_sc = [shannon_entropy(x, base_string_val) for x in generalized_str_sc] if generalized_str_sc else [0.0]

            shannon_mean_id_sc.append(np.mean(shannon_vals_id_sc))
            shannon_std_id_sc.append(np.std(shannon_vals_id_sc))
            shannon_max_id_sc.append(np.max(shannon_vals_id_sc))
            shannon_q3_id_sc.append(np.quantile(shannon_vals_id_sc, 0.75))

            shannon_mean_str_sc.append(np.mean(shannon_vals_str_sc))
            shannon_std_str_sc.append(np.std(shannon_vals_str_sc))
            shannon_max_str_sc.append(np.max(shannon_vals_str_sc))
            shannon_q3_str_sc.append(np.quantile(shannon_vals_str_sc, 0.75))
            
            homogeneous_id_sc.append(sum(1 for v in shannon_vals_id_sc if abs(v) < 1e-9))
            homogeneous_str_sc.append(sum(1 for v in shannon_vals_str_sc if abs(v) < 1e-9))
            heterogeneous_id_sc.append(obf_score_id_sc)
            heterogeneous_str_sc.append(obf_score_str_sc)

            current_url_count_sc = 0
            for s in strings_sc:
                try:
                    current_url_count_sc += len(self.url_extractor.find_urls(s))
                except Exception:
                    current_url_count_sc += len(contains_URL(s))
            url_count_sc.append(current_url_count_sc)
            
            # Metadata processing
            if check_metadata_strings[i] == False:
                strings_md = metadata_strings.iloc[i].split()
            else:
                strings_md = []
            
            if check_metadata_identifiers[i] == False:
                identifiers_md = metadata_identifiers.iloc[i].split()
                if any(f in identifiers_md for f in install) == True:
                    install_script = 1
                else:
                    install_script = 0
            else:
                identifiers_md = []
                install_script = 0

            generalized_id_md = [alphabetic_id_func(x) for x in identifiers_md]
            generalized_str_md = [alphabetic_string_func(s) for s in strings_md]

            obf_score_id_md = obfuscation(generalized_id_md, symbols=['u','d','l','s'])
            obf_score_str_md = obfuscation(generalized_str_md, symbols=['u','d','l','s'])

            shannon_vals_id_md = [shannon_entropy(x, base_id_val) for x in generalized_id_md] if generalized_id_md else [0.0]
            shannon_vals_str_md = [shannon_entropy(x, base_string_val) for x in generalized_str_md] if generalized_str_md else [0.0]

            shannon_mean_id_md.append(np.mean(shannon_vals_id_md))
            shannon_std_id_md.append(np.std(shannon_vals_id_md))
            shannon_max_id_md.append(np.max(shannon_vals_id_md))
            shannon_q3_id_md.append(np.quantile(shannon_vals_id_md, 0.75))

            shannon_mean_str_md.append(np.mean(shannon_vals_str_md))
            shannon_std_str_md.append(np.std(shannon_vals_str_md))
            shannon_max_str_md.append(np.max(shannon_vals_str_md))
            shannon_q3_str_md.append(np.quantile(shannon_vals_str_md, 0.75))

            homogeneous_id_md.append(sum(1 for v in shannon_vals_id_md if abs(v) < 1e-9))
            homogeneous_str_md.append(sum(1 for v in shannon_vals_str_md if abs(v) < 1e-9))
            heterogeneous_id_md.append(obf_score_id_md)
            heterogeneous_str_md.append(obf_score_str_md)

            current_url_count_md = 0
            for s in strings_md:
                try:
                    current_url_count_md += len(self.url_extractor.find_urls(s))
                except Exception:
                    current_url_count_md += len(contains_URL(s))
            url_count_md.append(current_url_count_md)

            presence_of_installation_script.append(install_script)
        
        pd.options.mode.chained_assignment = None
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        database['presence of installation script'] = presence_of_installation_script
        database['shannon mean ID source code'] = shannon_mean_id_sc
        database['shannon std ID source code'] = shannon_std_id_sc
        database['shannon max ID source code'] = shannon_max_id_sc
        database['shannon q3 ID source code'] = shannon_q3_id_sc
        database['shannon mean string source code'] = shannon_mean_str_sc
        database['shannon std string source code'] = shannon_std_str_sc
        database['shannon max string source code'] = shannon_max_str_sc
        database['shannon q3 string source code'] = shannon_q3_str_sc
        database['homogeneous identifiers in source code'] = homogeneous_id_sc
        database['homogeneous strings in source code'] = homogeneous_str_sc
        database['heteregeneous identifiers in source code'] = heterogeneous_id_sc
        database['heterogeneous strings in source code'] = heterogeneous_str_sc
        database['URLs in source code'] = url_count_sc

        database['shannon mean ID metadata'] = shannon_mean_id_md
        database['shannon std ID metadata'] = shannon_std_id_md
        database['shannon max ID metadata'] = shannon_max_id_md
        database['shannon q3 ID metadata'] = shannon_q3_id_md
        database['shannon mean string metadata'] = shannon_mean_str_md
        database['shannon std string metadata'] = shannon_std_str_md
        database['shannon max string metadata'] = shannon_max_str_md
        database['shannon q3 string metadata'] = shannon_q3_str_md
        database['homogeneous identifiers in metadata'] = homogeneous_id_md
        database['homogeneous strings in metadata'] = homogeneous_str_md
        database['heterogeneous strings in metadata'] = heterogeneous_str_md
        database['URLs in metadata'] = url_count_md
        database['heteregeneous identifiers in metadata'] = heterogeneous_id_md

        # Drop các cột không cần thiết (chỉ drop nếu tồn tại)
        cols_to_drop = ['strings_x', 'strings_y','identifiers_x','identifiers_y',
                       'has_install_requires', 'has_entry_points', 'risky_install_code', 'metadata_text',
                       'Number of Words_py', 'lines_py', 'base64_py', 'IP_py', 'suspicious_tokens_py',
                       'URLs_in_strings_py', 'emails_in_file_py', 'strings_py', 'identifiers_py']
        existing_cols_to_drop = [c for c in cols_to_drop if c in database.columns]
        if existing_cols_to_drop:
            database.drop(existing_cols_to_drop, axis=1, inplace=True)
        
        # Remove duplicates (không dùng subset vì có thể không tồn tại tất cả các cột)
        # Chỉ drop duplicate nếu tất cả các giá trị của package giống hệt nhau
        database = database.drop_duplicates(keep='first')
        
        # Rename columns để match với format của labelled dataset (chỉ rename nếu tồn tại)
        rename_map = {
            'Number of words_x':'Number of Words in source code',
            'Number of words_y':'Number of Words in metadata',
            'Number of words':'Number of Words in source code',  # Fallback nếu không có merge
            'lines_x':'Number of lines in source code',
            'lines_y':'Number of lines in metadata',
            'lines':'Number of lines in source code',  # Fallback
            'IP_x':'Number of IP adress in source code',
            'IP_y':'Number of IP adress in metadata',
            'IP':'Number of IP adress in source code',  # Fallback
            'base64_x':'Number of base64 chunks in source code',
            'base64_y':'Number of base64 chunks in metadata',
            'base64':'Number of base64 chunks in source code',  # Fallback
            'sospicious token_x':'Number of sospicious token in source code',
            'sospicious token_y':'Number of sospicious token in metadata',
            'sospicious token':'Number of sospicious token in source code'  # Fallback
        }
        existing_renames = {k: v for k, v in rename_map.items() if k in database.columns}
        if existing_renames:
            database.rename(columns=existing_renames, inplace=True)
        
        # Đảm bảo có cột metadata (nếu không có thì tạo với giá trị 0)
        metadata_cols = [
            'Number of Words in metadata',
            'Number of lines in metadata',
            'Number of IP adress in metadata',
            'Number of base64 chunks in metadata',
            'Number of sospicious token in metadata'
        ]
        for col in metadata_cols:
            if col not in database.columns:
                database[col] = 0
        
        return database