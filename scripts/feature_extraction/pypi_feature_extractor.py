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
        # Danh sách extension cho PyPI
        # Bao gồm các loại file mã nguồn Python, metadata, nén và các file nhị phân/tài liệu phổ biến
        # Các phần mở rộng file nén (.zip, .tar, .gz) vẫn có trong self.classes
        # để đếm các file nén bên trong một gói đã giải nén (nếu có),
        # nhưng sẽ được lọc riêng nếu đó là file nén gốc ở cấp root.
        self.classes = [
            'py','pyc','pyd','so','whl','egg','txt','md','rst','cfg','ini','toml','yml','yaml','json',
            'c','cpp','h','data','dat','bin','exe','dll','zip','tar','gz','bz2'
        ]

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
        # Bắt đầu với py_files_df, sau đó merge lần lượt
        final_df = py_files_df
        if not metadata_df.empty:
            final_df = pd.merge(final_df, metadata_df, on=['Package Name'], how='outer', suffixes=('_py', '_meta'))
        else:
            for col in ['has_install_requires', 'has_entry_points', 'risky_install_code', 'metadata_text']:
                final_df[col] = np.nan
            
        if not extensions_files_df.empty:
            final_df = pd.merge(final_df, extensions_files_df, on=['Package Name'], how='outer')
        else:
             for ext_col in ['.' + c for c in self.classes]:
                 final_df[ext_col] = 0

        required_for_extraction_py = ['Number of Words_py', 'lines_py', 'base64_py', 'IP_py', 'suspicious_tokens_py', 'URLs_in_strings_py', 'emails_in_file_py', 'strings_py', 'identifiers_py']
        for col in required_for_extraction_py:
            if col not in final_df.columns:
                final_df[col] = np.nan
        
        required_for_extraction_meta = ['has_install_requires', 'has_entry_points', 'risky_install_code', 'metadata_text']
        for col in required_for_extraction_meta:
            if col not in final_df.columns:
                final_df[col] = np.nan
        
        final_df = final_df.fillna(0)

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
        Trích xuất feature từ tất cả file .py.
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

                for tok_type, tok_val in token_source:
                    if tok_type in Token.Name:
                        if tok_val and not tok_val.isspace():
                            ids.append(tok_val)
                    elif tok_type in (Token.Literal.String.Single, Token.Literal.String.Double,
                                      Token.Literal.String.Heredoc, Token.Literal.String):
                        if tok_val and not tok_val.isspace():
                            cleaned_str = tok_val.strip()
                            if cleaned_str.startswith(('"', "'")):
                                cleaned_str = cleaned_str[1:]
                            if cleaned_str.endswith(('"', "'")):
                                cleaned_str = cleaned_str[:-1]
                            strs.append(cleaned_str)

                num_lines = data.count("\n") + 1
                size_chars = len(data) if len(data) > 0 else 1

                count_base64_in_strs = sum(len(contains_base64(s)) for s in strs)
                count_ip_in_strs = sum(len(contains_IPAddress(s)) for s in strs)
                count_url_in_strs = sum(len(contains_URL(s)) for s in strs)
                count_emails = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', data))
                count_dangerous = len(contains_dangerous_token(data, self.dangerous_token))

                ids_join = ' '.join([s.replace("'", "").replace('"', "") for s in ids])
                string_join = ' '.join(list(set([s.replace("'", "").replace('"', "") for s in strs])))

                Package.append(package_name)
                pyfile.append(py_name)
                suspicious_tokens.append(count_dangerous)
                lines.append(num_lines)
                num_chars.append(size_chars)
                base64_counts.append(count_base64_in_strs)
                ip_counts.append(count_ip_in_strs)
                url_counts_in_strings.append(count_url_in_strs)
                email_counts_in_file.append(count_emails)
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
            'identifiers_py': identifiers_list,
            'strings_py': strings_list,
            'code_py': code_list
        })

        return self.merge_py_of_same_package(db)

    def merge_py_of_same_package(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp nhiều file .py của cùng 1 package thành 1 record package-level.
        """
        if database.empty:
            cols = ['Package Name','Number of Words_py','lines_py','base64_py','IP_py',
                    'suspicious_tokens_py','URLs_in_strings_py','emails_in_file_py',
                    'strings_py','identifiers_py']
            return pd.DataFrame(columns=cols)

        p_database_code = database.groupby(['Package Name'], as_index=False)['code_py'].agg('\n'.join)
        p_database_code['Number of Words_py'] = p_database_code["code_py"].apply(lambda n: len(str(n).split()))

        l_database = database.groupby(['Package Name'], as_index=False)['lines_py'].sum()
        base = database.groupby(['Package Name'], as_index=False)['base64_py'].sum()
        ip = database.groupby(['Package Name'], as_index=False)['IP_py'].sum()
        sospicious = database.groupby(['Package Name'], as_index=False)['suspicious_tokens_py'].sum()
        urls = database.groupby(['Package Name'], as_index=False)['URLs_in_strings_py'].sum()
        emails = database.groupby(['Package Name'], as_index=False)['emails_in_file_py'].sum()
        
        string = database.groupby(['Package Name'], as_index=False)['strings_py'].agg(' '.join)
        identifier = database.groupby(['Package Name'], as_index=False)['identifiers_py'].agg(' '.join)

        data_to_merge = [p_database_code, l_database, base, ip, sospicious, urls, emails, string, identifier]
        final_database = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), data_to_merge)

        if 'code_py' in final_database.columns:
            final_database.drop('code_py', axis=1, inplace=True)
        
        final_database.columns = ['Package Name','Number of Words_py','lines_py','base64_py','IP_py',
                                  'suspicious_tokens_py','URLs_in_strings_py','emails_in_file_py',
                                  'strings_py','identifiers_py']
        return final_database

    def extract_feature_from_setup(self) -> pd.DataFrame:
        """
        Dò tất cả file metadata quan trọng của PyPI: setup.py, pyproject.toml, setup.cfg.
        """
        setup_files_paths = []
        setup_files_paths.extend(find_files_of_ext(self.path_to_scan, "setup.py"))
        setup_files_paths.extend(find_files_of_ext(self.path_to_scan, "pyproject.toml"))
        setup_files_paths.extend(find_files_of_ext(self.path_to_scan, "setup.cfg"))

        Package = []
        meta_file_name = []
        has_install_requires = []
        has_entry_points = []
        risky_calls = []
        metadata_text = []

        print(f"[*] Found {len(setup_files_paths)} metadata files.")

        for fp in setup_files_paths:
            p = Path(fp)
            # BỎ QUA CÁC FILE NÉN GỐC Ở ĐÂY
            if p.parent == Path(self.path_to_scan) and p.name.lower().endswith(self.ARCHIVE_EXTENSIONS):
                continue
            try:
                try:
                    package_name = str(p.relative_to(self.path_to_scan).parts[0])
                except ValueError:
                    package_name = p.parent.name
                
                Package.append(package_name)
                meta_file_name.append(p.name)

                with open(fp, "r", encoding="utf-8", errors='ignore') as f:
                    text = f.read()
                metadata_text.append(text)

                has_req = 0
                has_entry = 0
                risky = 0

                if p.name.lower() == "pyproject.toml" or p.name.lower() == "setup.cfg":
                    if re.search(r"install_requires|dependencies", text, re.IGNORECASE):
                        has_req = 1
                    if re.search(r"entry_points|scripts|console_scripts", text, re.IGNORECASE):
                        has_entry = 1
                    if re.search(r"(os\.system|subprocess|exec|eval|base64\.b64decode|urllib|requests\.get)", text, re.IGNORECASE):
                        risky = 1
                elif p.name.lower() == "setup.py":
                    try:
                        tree = ast.parse(text)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call) and \
                                (isinstance(node.func, ast.Name) and node.func.id == "setup") or \
                                (isinstance(node.func, ast.Attribute) and node.func.attr == "setup"):
                                for kw in node.keywords:
                                    if kw.arg in ("install_requires", "requires", "setup_requires", "extras_require"):
                                        has_req = 1
                                    if kw.arg in ("entry_points", "scripts", "console_scripts"):
                                        has_entry = 1
                                    if kw.arg == "cmdclass":
                                        risky = 1
                            
                            if isinstance(node, ast.Call):
                                func_name = None
                                if isinstance(node.func, ast.Name):
                                    func_name = node.func.id
                                elif isinstance(node.func, ast.Attribute):
                                    func_name = node.func.attr
                                    if isinstance(node.func.value, ast.Name):
                                        if node.func.value.id == 'os' and func_name == 'system': risky = 1
                                        if node.func.value.id == 'subprocess' and func_name in ['run', 'Popen', 'call']: risky = 1
                                        if node.func.value.id == 'base64' and func_name == 'b64decode': risky = 1
                                if func_name in ("exec", "eval", "compile"): risky = 1

                    except SyntaxError as se:
                        print(f"[WARN] SyntaxError in {fp}: {se}. Falling back to regex scan.")
                        if re.search(r"(os\.system|subprocess|exec|eval|base64\.b64decode|urllib|requests\.get)", text, re.IGNORECASE):
                            risky = 1
                    except Exception as ex:
                        print(f"[WARN] Unexpected error during AST parsing for {fp}: {ex}. Falling back to regex scan.")
                        if re.search(r"(os\.system|subprocess|exec|eval|base64\.b64decode|urllib|requests\.get)", text, re.IGNORECASE):
                            risky = 1

                has_install_requires.append(has_req)
                has_entry_points.append(has_entry)
                risky_calls.append(risky)

            except Exception as e:
                print(f"[WARN] Error processing metadata file '{fp}': {e}")
                continue

        db = pd.DataFrame({
            'Package Name': Package,
            'meta_file_name': meta_file_name,
            'has_install_requires': has_install_requires,
            'has_entry_points': has_entry_points,
            'risky_install_code': risky_calls,
            'metadata_text': metadata_text
        })
        
        return self.p_db_benign_md(db)

    def p_db_benign_md(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp các hàng metadata của cùng 1 package thành 1 record package-level.
        """
        if database.empty:
            cols = ['Package Name','has_install_requires','has_entry_points','risky_install_code','metadata_text']
            return pd.DataFrame(columns=cols)

        p_database = database.groupby(['Package Name'], as_index=False).agg({
            'metadata_text': '\n'.join,
            'has_install_requires': 'max',
            'has_entry_points': 'max',
            'risky_install_code': 'max'
        })
        return p_database

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
        Tính các feature nâng cao.
        """
        for col_name in ['strings_py', 'identifiers_py', 'metadata_text']:
            if col_name in database.columns:
                database[col_name] = database[col_name].apply(lambda x: '' if pd.isna(x) else str(x))
            else:
                database[col_name] = ''

        database.index = range(0, len(database))

        source_code_strings = database['strings_py']
        source_code_identifiers = database['identifiers_py']
        metadata_content = database['metadata_text']

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

            strings_sc = source_code_strings.iloc[i].split() if source_code_strings.iloc[i] else []
            identifiers_sc = source_code_identifiers.iloc[i].split() if source_code_identifiers.iloc[i] else []

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

            meta_text_content = metadata_content.iloc[i]
            meta_words = meta_text_content.split() if meta_text_content else []

            generalized_id_md = [alphabetic_id_func(x) for x in meta_words]
            generalized_str_md = [alphabetic_string_func(s) for s in meta_words]

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
            try:
                current_url_count_md += len(self.url_extractor.find_urls(meta_text_content))
            except Exception:
                current_url_count_md += len(contains_URL(meta_text_content))
            url_count_md.append(current_url_count_md)

            install_flag = max(database['has_install_requires'].iloc[i], 
                               database['has_entry_points'].iloc[i], 
                               database['risky_install_code'].iloc[i])
            presence_of_installation_script.append(install_flag)
        
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

        for c in ['strings_py', 'identifiers_py', 'metadata_text', 'meta_file_name',
                  'has_install_requires', 'has_entry_points', 'risky_install_code', 'py_file_name', 'code_py',
                  'num_chars_py']:
            if c in database.columns:
                database.drop(c, axis=1, inplace=True)
        
        rename_map = {
            'Number of Words_py': 'Number of Words in source code',
            'lines_py': 'Number of lines in source code',
            'IP_py': 'Number of IP address in source code',
            'base64_py': 'Number of base64 chunks in source code',
            'suspicious_tokens_py': 'Number of suspicious token in source code',
            'URLs_in_strings_py': 'Number of URLs in source code strings',
            'emails_in_file_py': 'Number of emails in source code'
        }
        database.rename(columns={k: v for k, v in rename_map.items() if k in database.columns}, inplace=True)
        
        database = database.loc[:,~database.columns.duplicated()]

        return database