from typing import Optional
import nltk
import json
import os
import tarfile
import warnings
from pathlib import Path
from functools import reduce

import pandas as pd
import numpy as np
import statistics

from pygments.lexers import JavascriptLexer, JsonLexer
from pygments.token import Token
from urlextract import URLExtract

import utilities_functions


class NPM_Feature_Extractor:
    """
    Lớp chính thực hiện pipeline trích xuất features cho tập NPM packages.
    Các phương thức:
    - __init__: khởi tạo config, tải stopwords, đọc danh sách dangerous tokens
    - extract_features(path): entrypoint dùng để chạy toàn bộ pipeline
    - unzip_packages(): giải nén các file .tgz (nếu cần)
    - extract_feature_from_js(): trích xuất feature ở mức file .js, rồi gộp theo package
    - merge_js_of_same_package(): gộp nhiều .js trong cùng package thành summary
    - extract_feature_from_package_json(): tương tự cho package.json
    - p_db_benign_md(): gộp metadata (package.json) tương tự merge_js...
    - count_package_files_extension(): đếm số file theo extension cho mỗi package
    - extraction(...): tính các feature nâng cao (entropy, obfuscation, URLs, installation script)
    """

    def __init__(self) -> None:
        # Danh sách extension quan tâm
        self.classes = [
            'bat','bz2','c','cert','conf','cpp','crt','css','csv','deb','erb','gemspec','gif','gz','h','html','ico','ini','jar',
            'java','jpg','js','json','key','m4v','markdown','md','pdf','pem','png','ps','py','rb','rpm','rst','sh','svg','toml',
            'ttf','txt','xml','yaml','yml','eot','exe','jpeg','properties','sql','swf','tar','woff','woff2','aac','bmp','cfg',
            'dcm','dll','doc','flac','flv','ipynb','m4a','mid','mkv','mp3','mp4','mpg','ogg','otf','pickle','pkl','psd','pxd',
            'pxi','pyc','pyx','r','rtf','so','sqlite','tif','tp','wav','webp','whl','xcf','xz','zip','mov','wasm','webm'
        ]

        # Tải stopwords của NLTK (nếu chưa có thì download)
        try:
            nltk.data.find('corpora/stopwords')
        except Exception:
            nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        # Nạp danh sách dangerous token (nếu file không tồn tại thì dùng danh sách rỗng)
        try:
            here = Path(__file__).parent
            with open(here / 'resources' / 'dangerous_tokens.json', 'r', encoding='utf-8') as fh:
                self.dangerous_token = json.load(fh)
        except Exception:
            # Giữ mặc định rỗng để tránh crash nếu file không tìm thấy
            self.dangerous_token = []

        # Đường dẫn gốc để quét (được gán khi gọi extract_features)
        self.path_to_scan: Optional[str] = None

    def extract_features(self, path: str) -> pd.DataFrame:
        """
        Entrypoint chính:
        - path: thư mục chứa các file .tgz hoặc đã giải nén packages
        - Trả về pandas.DataFrame với tất cả các feature đã tính và cũng lưu file CSV
        """
        self.path_to_scan = path
        # Giải nén .tgz (không ghi đè nếu đã giải nén)
        self.unzip_packages()

        # Trích xuất feature từ mã nguồn và metadata
        javascript_files_df = self.extract_feature_from_js()
        packagejson_files_df = self.extract_feature_from_package_json()
        extensions_files_df = self.count_package_files_extension()

        # Gộp các DataFrame theo 'Package Name' (outer join để giữ tất cả package)
        dfs = [javascript_files_df, packagejson_files_df, extensions_files_df]
        final_df = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), dfs)

        # Tính các feature nâng cao (entropy, obfuscation, URL, installation script...)
        final_df = self.extraction(final_df, utilities_functions.gen_language_4, 4, utilities_functions.gen_language_4, 4)

        # Lưu kết quả và trả về
        final_df.to_csv("npm_feature_extracted.csv", encoding='utf-8', index=False)
        return final_df

    def unzip_packages(self) -> None:
        """
        Duyệt cây thư mục self.path_to_scan, tìm file .tgz và giải nén từng file vào thư mục con có tên package.
        - Chỉ giải nén khi file thực sự tồn tại và có kích thước > 0.
        - Sử dụng full path để kiểm tra kích thước (fix bug so với original).
        """
        if not self.path_to_scan:
            return

        for root, dirs, files in os.walk(self.path_to_scan):
            for fname in files:
                if fname.lower().endswith(".tgz"):
                    fullpath = os.path.join(root, fname)
                    try:
                        # kiểm tra kích thước file (dùng fullpath)
                        if os.path.getsize(fullpath) > 0:
                            # tạo thư mục đích cùng tên file bỏ .tgz, nằm cùng cấp với file
                            dest_dir = os.path.splitext(fullpath)[0]
                            # nếu đã giải nén thì bỏ qua
                            if not os.path.exists(dest_dir):
                                os.makedirs(dest_dir, exist_ok=True)
                                with tarfile.open(fullpath, 'r:gz') as pkg:
                                    print(f"[*] Extracting {fullpath}")
                                    pkg.extractall(path=dest_dir)
                    except Exception:
                        # im lặng nếu có lỗi giải nén (giữ hành vi tương tự bản gốc)
                        continue

    def extract_feature_from_js(self) -> pd.DataFrame:
        """
        Trích xuất feature từ tất cả file .js:
        - Dùng utilities_functions.find_files_of_ext để lấy danh sách path .js
        - Tokenize bằng Pygments JavascriptLexer, thu thập operators, punctuation, identifiers, strings
        - Tính số dòng, tỷ lệ operator '+', '=' và dấu '[', đếm base64/IP/dangerous token trong strings
        - Trả về DataFrame tạm với 1 hàng tương ứng 1 file .js, sau đó gọi merge_js_of_same_package để tóm tắt theo package
        """
        files_path = utilities_functions.find_files_of_ext(self.path_to_scan, ".js")

        # Các list để chứa kết quả từng file
        Package = []
        jsfile = []
        strings_list = []
        identifiers_list = []
        suspicious_tokens = []
        lines = []
        plus_ratio = []
        equal_ratio = []
        square_ratio = []
        base64_counts = []
        ip_counts = []
        code_list = []

        for fp in files_path:
            try:
                p = Path(fp)
                # Lấy package name theo relative path so với path_to_scan nếu có thể, fallback về phần tử parts[2]
                try:
                    package_name = str(Path(fp).relative_to(self.path_to_scan).parts[0])
                except Exception:
                    package_name = p.parts[2] if len(p.parts) > 2 else p.parent.name

                js_name = p.name

                # Đọc toàn bộ file (dùng với encoding ignore để tránh crash)
                with open(fp, "r", encoding="utf8", errors='ignore') as f:
                    data = f.read()

                # Tokenize bằng JavascriptLexer
                lexer = JavascriptLexer(stripnl=False, ensurenl=False)
                token_source = lexer.get_tokens(data)

                operators = []
                punctuation = []
                ids = []
                strs = []

                for tok_type, tok_val in token_source:
                    if tok_type in Token.Operator:
                        operators.append(tok_val)
                    elif tok_type in Token.Punctuation:
                        punctuation.append(tok_val)
                    elif tok_type in Token.Name:
                        if tok_val and not tok_val.isspace():
                            ids.append(tok_val)
                    elif tok_type in (Token.Literal.String.Single, Token.Literal.String.Double,
                                      Token.Literal.String.Affix, Token.Literal.String.Backtick,
                                      Token.Literal.String.Char, Token.Literal.String.Delimiter,
                                      Token.Literal.String.Doc, Token.Literal.String.Escape,
                                      Token.Literal.String.Heredoc, Token.Literal.String.Interpol,
                                      Token.Literal.String.Other):
                        if tok_val and not tok_val.isspace():
                            strs.append(tok_val)
                    else:
                        # bỏ qua các token khác
                        continue

                # Thống kê cơ bản: số dòng & kích thước (số ký tự) để tính tỷ lệ (giữ ý nghĩa tương tự bản gốc)
                num_lines = data.count("\n") + 1
                size_chars = len(data) if len(data) > 0 else 1  # tránh chia cho 0

                equalities = sum(v.count('=') for v in operators) / size_chars
                pluses = sum(v.count('+') for v in operators) / size_chars
                Lbrackets = punctuation.count('[') / size_chars

                # Tính base64, IP, dangerous token trong các string literal
                count_base64 = 0
                count_IP = 0
                byte = 0
                for s in strs:
                    count_base64 += len(utilities_functions.contains_base64(s))
                    count_IP += len(utilities_functions.contains_IPAddress(s))
                    byte += len(utilities_functions.contains_dangerous_token(s, self.dangerous_token))

                # Chuẩn hóa identifiers & strings (bỏ dấu nháy) và loại stopwords
                ids = [s.replace("'", "").replace('"', "") for s in ids]
                ids_join = ' '.join(ids)

                strs_clean = [s.replace("'", "").replace('"', "") for s in strs]
                string_tokens = list(set(strs_clean) - self.stopwords)
                string_join = ' '.join(string_tokens)

                # Lưu kết quả
                Package.append(package_name)
                jsfile.append(js_name)
                suspicious_tokens.append(byte)
                lines.append(num_lines)
                plus_ratio.append(pluses)
                equal_ratio.append(equalities)
                square_ratio.append(Lbrackets)
                identifiers_list.append(ids_join)
                base64_counts.append(count_base64)
                ip_counts.append(count_IP)
                strings_list.append(string_join)
                code_list.append(data)

            except Exception:
                # Nếu 1 file lỗi thì bỏ qua (giữ hành vi bản gốc)
                continue

        # Xây DataFrame và gọi merge_js_of_same_package để tóm tắt theo package
        db = pd.DataFrame({
            'Package Name': Package,
            '.js': jsfile,
            'sospicious token': suspicious_tokens,
            'lines': lines,
            'equal ratio': equal_ratio,
            'plus ratio': plus_ratio,
            'bracket ratio': square_ratio,
            'identifiers': identifiers_list,
            'base64': base64_counts,
            'IP': ip_counts,
            'strings': strings_list,
            'code': code_list
        })

        return self.merge_js_of_same_package(db)

    def merge_js_of_same_package(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp nhiều file .js của cùng 1 package thành 1 record package-level:
        - nối code, tính Number of words từ toàn bộ code,
        - tổng lines, tổng base64/IP/suspicious, và thống kê cho tỷ lệ operator/punctuation
        """
        if database.empty:
            # Trả về DataFrame rỗng với các cột mong đợi
            cols = ['Package Name','Number of words','lines','plus ratio mean','plus ratio max','plus ratio std','plus ratio q3',
                    'eq ratio mean','eq ratio max','eq ratio std','eq ratio q3',
                    'bracket ratio mean','bracket ratio max','bracket ratio std','bracket ratio q3',
                    'base64','IP','sospicious token','strings','identifiers']
            return pd.DataFrame(columns=cols)

        p_database = database.groupby(['Package Name'], as_index=False)['code'].agg('\n'.join)
        p_database['Number of words'] = p_database["code"].apply(lambda n: len(n.split()))

        l_database = database.groupby(['Package Name'], as_index=False)['lines'].sum()

        plus_mean = database.groupby(['Package Name'], as_index=False)['plus ratio'].mean().rename(columns={"plus ratio": "plus ratio mean"})
        plus_max = database.groupby(['Package Name'], as_index=False)['plus ratio'].max().rename(columns={"plus ratio": "plus ratio max"})
        plus_std = database.groupby(['Package Name'], as_index=False)['plus ratio'].std().rename(columns={"plus ratio": "plus ratio std"})
        plus_q3 = database.groupby(['Package Name'], as_index=False)['plus ratio'].quantile(0.75).rename(columns={"plus ratio": "plus ratio q3"})

        eq_mean = database.groupby(['Package Name'], as_index=False)['equal ratio'].mean().rename(columns={"equal ratio": "equal ratio mean"})
        eq_max = database.groupby(['Package Name'], as_index=False)['equal ratio'].max().rename(columns={"equal ratio": "equal ratio max"})
        eq_std = database.groupby(['Package Name'], as_index=False)['equal ratio'].std().rename(columns={"equal ratio": "equal ratio std"})
        eq_q3 = database.groupby(['Package Name'], as_index=False)['equal ratio'].quantile(0.75).rename(columns={"equal ratio": "equal ratio q3"})

        bracket_mean = database.groupby(['Package Name'], as_index=False)['bracket ratio'].mean().rename(columns={"bracket ratio": "bracket ratio mean"})
        bracket_max = database.groupby(['Package Name'], as_index=False)['bracket ratio'].max().rename(columns={"bracket ratio": "bracket ratio max"})
        bracket_std = database.groupby(['Package Name'], as_index=False)['bracket ratio'].std().rename(columns={"bracket ratio": "bracket ratio std"})
        bracket_q3 = database.groupby(['Package Name'], as_index=False)['bracket ratio'].quantile(0.75).rename(columns={"bracket ratio": "bracket ratio q3"})

        base = database.groupby(['Package Name'], as_index=False)['base64'].sum()
        ip = database.groupby(['Package Name'], as_index=False)['IP'].sum()
        sospicious = database.groupby(['Package Name'], as_index=False)['sospicious token'].sum()
        string = database.groupby(['Package Name'], as_index=False)['strings'].agg(' '.join)
        identifier = database.groupby(['Package Name'], as_index=False)['identifiers'].agg(' '.join)

        data = [p_database, l_database, plus_mean, plus_max, plus_std, plus_q3, eq_mean, eq_max, eq_std, eq_q3,
                bracket_mean, bracket_max, bracket_std, bracket_q3, base, ip, sospicious, string, identifier]

        final_database = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), data)

        # Loại cột code (raw source) không cần thiết ở cấp summary
        if 'code' in final_database.columns:
            final_database.drop('code', axis=1, inplace=True)

        final_database.columns = ['Package Name','Number of words','lines','plus ratio mean','plus ratio max','plus ratio std','plus ratio q3',
                                  'eq ratio mean','eq ratio max','eq ratio std','eq ratio q3',
                                  'bracket ratio mean','bracket ratio max','bracket ratio std','bracket ratio q3',
                                  'base64','IP','sospicious token','strings','identifiers']
        return final_database

    def extract_feature_from_package_json(self) -> pd.DataFrame:
        """
        Dò tất cả file .json, chỉ xử lý file có tên 'package.json':
        - Tokenize bằng JsonLexer, thu thập strings/identifiers/operators tương tự JS
        - Trả về DataFrame tạm (1 hàng = 1 package.json file) và gọi p_db_benign_md để gộp theo package
        """
        files_path = utilities_functions.find_files_of_ext(self.path_to_scan, '.json')

        Package = []
        jsfile = []
        strings_list = []
        identifiers_list = []
        suspicious_tokens = []
        lines = []
        plus_ratio = []
        equal_ratio = []
        square_ratio = []
        base64_counts = []
        ip_counts = []
        code_list = []

        for fp in files_path:
            # Chỉ xử lý package.json
            if Path(fp).name != "package.json":
                continue
            try:
                p = Path(fp)
                try:
                    package_name = str(Path(fp).relative_to(self.path_to_scan).parts[0])
                except Exception:
                    package_name = p.parts[2] if len(p.parts) > 2 else p.parent.name

                js_name = p.name

                with open(fp, "r", encoding="utf8", errors='ignore') as f:
                    data = f.read()

                lexer = JsonLexer(stripnl=False, ensurenl=False)
                token_source = lexer.get_tokens(data)

                operators = []
                punctuation = []
                ids = []
                strs = []

                for tok_type, tok_val in token_source:
                    if tok_type in Token.Operator:
                        operators.append(tok_val)
                    elif tok_type in Token.Punctuation:
                        punctuation.append(tok_val)
                    elif tok_type in Token.Name:
                        if tok_val and not tok_val.isspace():
                            ids.append(tok_val)
                    elif tok_type in (Token.Literal.String.Single, Token.Literal.String.Double,
                                      Token.Literal.String.Affix, Token.Literal.String.Backtick,
                                      Token.Literal.String.Char, Token.Literal.String.Delimiter,
                                      Token.Literal.String.Doc, Token.Literal.String.Escape,
                                      Token.Literal.String.Heredoc, Token.Literal.String.Interpol,
                                      Token.Literal.String.Other):
                        if tok_val and not tok_val.isspace():
                            strs.append(tok_val)
                    else:
                        continue

                num_lines = data.count("\n") + 1
                size_chars = len(data) if len(data) > 0 else 1

                equalities = sum(v.count('=') for v in operators) / size_chars
                pluses = sum(v.count('+') for v in operators) / size_chars
                Lbrackets = punctuation.count('[') / size_chars

                count_base64 = 0
                count_IP = 0
                byte = 0
                for s in strs:
                    count_base64 += len(utilities_functions.contains_base64(s))
                    count_IP += len(utilities_functions.contains_IPAddress(s))
                    byte += len(utilities_functions.contains_dangerous_token(s, self.dangerous_token))

                ids = [s.replace("'", "").replace('"', "") for s in ids]
                ids_join = ' '.join(ids)

                strs_clean = [s.replace("'", "").replace('"', "") for s in strs]
                string_tokens = list(set(strs_clean) - self.stopwords)
                string_join = ' '.join(string_tokens)

                Package.append(package_name)
                jsfile.append(js_name)
                suspicious_tokens.append(byte)
                lines.append(num_lines)
                plus_ratio.append(pluses)
                equal_ratio.append(equalities)
                square_ratio.append(Lbrackets)
                identifiers_list.append(ids_join)
                base64_counts.append(count_base64)
                ip_counts.append(count_IP)
                strings_list.append(string_join)
                code_list.append(data)
            except Exception:
                continue

        db = pd.DataFrame({
            'Package Name': Package,
            '.json': jsfile,
            'sospicious token': suspicious_tokens,
            'lines': lines,
            'equal ratio': equal_ratio,
            'plus ratio': plus_ratio,
            'bracket ratio': square_ratio,
            'identifiers': identifiers_list,
            'base64': base64_counts,
            'IP': ip_counts,
            'strings': strings_list,
            'code': code_list
        })

        return self.p_db_benign_md(db)

    def p_db_benign_md(self, database: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp các hàng package.json theo package:
        - Nối code, tổng lines/base64/IP/suspicious, gộp strings và identifiers
        - Trả về DataFrame tóm tắt per-package (metadata)
        """
        if database.empty:
            cols = ['Package Name','Number of words','lines','base64','IP','sospicious token','strings','identifiers']
            return pd.DataFrame(columns=cols)

        p_database = database.groupby(['Package Name'], as_index=False)['code'].agg('\n'.join)
        p_database['Number of words'] = p_database["code"].apply(lambda n: len(n.split()))
        l_database = database.groupby(['Package Name'], as_index=False)['lines'].sum()
        base = database.groupby(['Package Name'], as_index=False)['base64'].sum()
        ip = database.groupby(['Package Name'], as_index=False)['IP'].sum()
        sospicious = database.groupby(['Package Name'], as_index=False)['sospicious token'].sum()
        string = database.groupby(['Package Name'], as_index=False)['strings'].agg(' '.join)
        identifier = database.groupby(['Package Name'], as_index=False)['identifiers'].agg(' '.join)

        data = [p_database, l_database, base, ip, sospicious, string, identifier]
        final_database = reduce(lambda left, right: pd.merge(left, right, on=['Package Name'], how='outer'), data)

        if 'code' in final_database.columns:
            final_database.drop('code', axis=1, inplace=True)

        final_database.columns = ['Package Name','Number of words','lines','base64','IP','sospicious token','strings','identifiers']
        return final_database

    def count_package_files_extension(self) -> pd.DataFrame:
        """
        Đếm số file theo mỗi extension trong self.classes cho từng package.
        - Sử dụng utilities_functions.find_files_of_ext để lấy file theo extension.
        - Trả về DataFrame (Package Name + từng cột extension).
        """
        Package = []
        extension = []

        for ext_name in self.classes:
            ext = '.' + ext_name
            files_path = utilities_functions.find_files_of_ext(self.path_to_scan, ext)
            for fpath in files_path:
                p = Path(fpath)
                try:
                    package_name = str(Path(fpath).relative_to(self.path_to_scan).parts[0])
                except Exception:
                    package_name = p.parts[2] if len(p.parts) > 2 else p.parent.name
                Package.append(package_name)
                extension.append(ext)

        db = pd.DataFrame({'Package Name': Package, 'extension': extension})
        if db.empty:
            # tạo khung rỗng có cột extension đầy đủ
            empty_cols = ['Package Name'] + ['.' + e for e in self.classes]
            return pd.DataFrame(columns=empty_cols)

        db = db.groupby(['Package Name', 'extension']).size().unstack(fill_value=0)
        db = db.reset_index()
        db = db.groupby('Package Name').last()

        extensions = ['.' + c for c in self.classes]
        # Thêm cột thiếu nếu cần (điền 0)
        for col in extensions:
            if col not in db.columns:
                db[col] = 0

        # Sắp xếp cột theo danh sách extensions đã định
        db = db[extensions]
        db.reset_index(inplace=True)
        return db

    def extraction(self, database: pd.DataFrame, alphabetic_string, base_string, alphabetic_id, base_id) -> pd.DataFrame:
        """
        Tính các feature nâng cao (áp dụng sau khi đã merge source + metadata + extension counts):
        - entropy (Shannon) cho identifiers và strings (cả source code và metadata)
        - obfuscation (utilities_functions.obfuscation)
        - số URL trong strings (URLExtract)
        - presence of installation script trong metadata
        - Chuẩn hóa cột, đổi tên cột và trả về DataFrame hoàn chỉnh
        """
        extractor = URLExtract()

        # Tạo các cột strings_x / identifiers_x / strings_y / identifiers_y nếu thiếu (để tránh KeyError)
        for col in ['strings_x', 'identifiers_x', 'strings_y', 'identifiers_y']:
            if col not in database.columns:
                database[col] = np.nan

        # Thêm cột repository mặc định 1 (giữ cách làm như bản gốc)
        database['repository'] = 1

        # Điền NaN cho các cột khác (tránh lỗi khi tính toán)
        keep_cols = ['strings_x', 'identifiers_x', 'strings_y', 'identifiers_y']
        f = [c for c in database.columns if c not in keep_cols]
        database[f] = database[f].fillna(0)

        # Reset index để iterate an toàn
        database.index = range(0, len(database))

        # Chuẩn bị series nguồn/metadata để tiện truy xuất
        source_code_strings = database['strings_x']
        source_code_identifiers = database['identifiers_x']
        metadata_strings = database['strings_y']
        metadata_identifiers = database['identifiers_y']
        name = database['Package Name']
        repository = database['repository']

        check_metadata_strings = metadata_strings.isna()
        check_metadata_identifiers = metadata_identifiers.isna()
        check_source_code_strings = source_code_strings.isna()
        check_source_code_identifiers = source_code_identifiers.isna()

        # Các list lưu kết quả từng package (source & metadata)
        q3_id_sc = []; q3_str_sc = []; m_id_sc = []; m_str_sc = []; dev_id_sc = []; dev_str_sc = []
        maximum_id_sc = []; maximum_str_sc = []; flat_id_sc = []; flat_string_sc = []; count_url_sc = []; obf_id_sc = []; obf_string_sc = []
        q3_id_md = []; q3_str_md = []; m_id_md = []; m_str_md = []; dev_id_md = []; dev_str_md = []
        maximum_id_md = []; maximum_str_md = []; flat_id_md = []; flat_string_md = []; count_url_md = []; obf_id_md = []; obf_string_md = []
        installation = []

        # Lặp theo mỗi package trong database
        for i in range(len(database)):
            # In tên package (giữ như bản gốc có in để debug/monitor)
            try:
                print(name[i])
            except Exception:
                print(f"[?] Unknown package index {i}")

            # Chọn entrypoint install dựa trên repository (giữ logic như bản gốc)
            if repository[i] == 3:
                install = ['extensions']
            elif repository[i] == 2:
                install = ['install']
            else:
                install = ['postinstall', 'preinstall', 'install']

            # --- Source code processing ---
            if not check_source_code_strings[i]:
                string_sourcecode = source_code_strings[i]
                strings = string_sourcecode.split()
            else:
                strings = []

            if not check_source_code_identifiers[i]:
                identifiers = source_code_identifiers[i].split()
            else:
                identifiers = []

            # Generalize identifiers & strings bằng hàm được truyền vào
            generalization_id = [alphabetic_id(x) for x in identifiers]
            obf_sc = utilities_functions.obfuscation(generalization_id, symbols=['u','d','l','s'])

            generalization_str = []
            url_sc = 0
            for s in strings:
                try:
                    url_sc += len(extractor.find_urls(s))
                except Exception:
                    # fallback dùng utilities_functions.contains_URL nếu URLExtract lỗi
                    url_sc += len(utilities_functions.contains_URL(s)) if hasattr(utilities_functions, 'contains_URL') else 0
                generalization_str.append(alphabetic_string(s))
            obf_sc_str = utilities_functions.obfuscation(generalization_str, symbols=['u','d','l','s'])

            # Tính Shannon entropy cho mỗi token generalization
            shannon_id = [utilities_functions.shannon_entropy(x, base_id) for x in generalization_id] if generalization_id else []
            shannon_str = [utilities_functions.shannon_entropy(x, base_string) for x in generalization_str] if generalization_str else []

            null_string_sc = len([v for v in shannon_str if abs(v) == 0])
            null_id_sc = len([v for v in shannon_id if abs(v) == 0])

            if len(shannon_str) >= 1:
                mean_str = statistics.mean(shannon_str)
                max_str = max(shannon_str)
                quart_str = np.quantile(shannon_str, 0.75)
            else:
                mean_str = max_str = quart_str = 0.0
            std_str = np.std(shannon_str) if len(shannon_str) > 1 else 0.0

            if len(shannon_id) >= 1:
                mean_id = statistics.mean(shannon_id)
                max_id = max(shannon_id)
                quart_id = np.quantile(shannon_id, 0.75)
            else:
                mean_id = max_id = quart_id = 0.0
            std_id = np.std(shannon_id) if len(shannon_id) > 1 else 0.0

            m_str_sc.append(mean_str); dev_str_sc.append(std_str); maximum_str_sc.append(max_str); q3_str_sc.append(quart_str)
            m_id_sc.append(mean_id); dev_id_sc.append(std_id); maximum_id_sc.append(max_id); q3_id_sc.append(quart_id)
            flat_id_sc.append(null_id_sc); flat_string_sc.append(null_string_sc); count_url_sc.append(url_sc)
            obf_id_sc.append(obf_sc); obf_string_sc.append(obf_sc_str)

            # --- Metadata processing (package.json) ---
            if not check_metadata_strings[i]:
                string_md = metadata_strings[i].split()
            else:
                string_md = []

            if not check_metadata_identifiers[i]:
                identifiers_md = metadata_identifiers[i].split()
                install_script = 1 if any(f in identifiers_md for f in install) else 0
            else:
                identifiers_md = []
                install_script = 0

            generalization_id_md = [alphabetic_id(x) for x in identifiers_md]
            obf_md = utilities_functions.obfuscation(generalization_id_md, symbols=['u','d','l','s'])

            generalization_str_md = []
            url_md = 0
            for s in string_md:
                try:
                    url_md += len(extractor.find_urls(s))
                except Exception:
                    url_md += len(utilities_functions.contains_URL(s)) if hasattr(utilities_functions, 'contains_URL') else 0
                generalization_str_md.append(alphabetic_string(s))
            obf_md_str = utilities_functions.obfuscation(generalization_str_md, symbols=['u','d','l','s'])

            shannon_id_md = [utilities_functions.shannon_entropy(x, base_id) for x in generalization_id_md] if generalization_id_md else []
            shannon_str_md = [utilities_functions.shannon_entropy(x, base_string) for x in generalization_str_md] if generalization_str_md else []

            null_id_md = len([v for v in shannon_id_md if abs(v) == 0])
            null_string_md = len([v for v in shannon_str_md if abs(v) == 0])

            if len(shannon_str_md) >= 1:
                mean_str_md = statistics.mean(shannon_str_md)
                max_str_md = max(shannon_str_md)
                quart_str_md = np.quantile(shannon_str_md, 0.75)
            else:
                mean_str_md = max_str_md = quart_str_md = 0.0
            std_str_md = np.std(shannon_str_md) if len(shannon_str_md) > 1 else 0.0

            if len(shannon_id_md) >= 1:
                mean_id_md = statistics.mean(shannon_id_md)
                max_id_md = max(shannon_id_md)
                quart_id_md = np.quantile(shannon_id_md, 0.75)
            else:
                mean_id_md = max_id_md = quart_id_md = 0.0
            std_id_md = np.std(shannon_id_md) if len(shannon_id_md) > 1 else 0.0

            installation.append(install_script)
            m_str_md.append(mean_str_md); dev_str_md.append(std_str_md); maximum_str_md.append(max_str_md); q3_str_md.append(quart_str_md)
            m_id_md.append(mean_id_md); dev_id_md.append(std_id_md); maximum_id_md.append(max_id_md); q3_id_md.append(quart_id_md)
            flat_id_md.append(null_id_md); flat_string_md.append(null_string_md); count_url_md.append(url_md)
            obf_id_md.append(obf_md); obf_string_md.append(obf_md_str)

        # Gán các cột mới vào DataFrame gốc (cảnh báo performance suppressed như bản gốc)
        pd.options.mode.chained_assignment = None
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        database['presence of installation script'] = installation
        database['shannon mean ID source code'] = m_id_sc
        database['shannon std ID source code'] = dev_id_sc
        database['shannon max ID source code'] = maximum_id_sc
        database['shannon q3 ID source code'] = q3_id_sc
        database['shannon mean string source code'] = m_str_sc
        database['shannon std string source code'] = dev_str_sc
        database['shannon max string source code'] = maximum_str_sc
        database['shannon q3 string source code'] = q3_str_sc
        database['homogeneous identifiers in source code'] = flat_id_sc
        database['homogeneous strings in source code'] = flat_string_sc
        database['heteregeneous identifiers in source code'] = obf_id_sc
        database['heterogeneous strings in source code'] = obf_string_sc
        database['URLs in source code'] = count_url_sc

        database['shannon mean ID metadata'] = m_id_md
        database['shannon std ID metadata'] = dev_id_md
        database['shannon max ID metadata'] = maximum_id_md
        database['shannon q3 ID metadata'] = q3_id_md
        database['shannon mean string metadata'] = m_str_md
        database['shannon std string metadata'] = dev_str_md
        database['shannon max string metadata'] = maximum_str_md
        database['shannon q3 string metadata'] = q3_str_md
        database['homogeneous identifiers in metadata'] = flat_id_md
        database['homogeneous strings in metadata'] = flat_string_md
        database['heterogeneous strings in metadata'] = obf_string_md
        database['URLs in metadata'] = count_url_md
        database['heteregeneous identifiers in metadata'] = obf_id_md

        # Xóa các cột thô chứa strings/identifiers (đã chuyển thành features)
        for c in ['strings_x', 'strings_y', 'identifiers_x', 'identifiers_y']:
            if c in database.columns:
                database.drop(c, axis=1, inplace=True)

        # Loại duplicates dựa trên một số cột numeric giống bản gốc (nếu các cột này tồn tại)
        subset_cols = ['Number of words_x', 'Number of words_y', 'lines_x', 'lines_y', 'repository', 'presence of installation script']
        subset_present = [c for c in subset_cols if c in database.columns]
        if subset_present:
            database.drop_duplicates(subset=subset_present, keep='first', inplace=True)

        # Đổi tên một vài cột cho rõ nghĩa (chỉ đổi nếu tồn tại)
        rename_map = {
            'Number of words_x': 'Number of Words in source code',
            'Number of words_y': 'Number of Words in metadata',
            'lines_x': 'Number of lines in source code',
            'lines_y': 'Number of lines in metadata',
            'IP_x': 'Number of IP adress in source code',
            'base64_x': 'Number of base64 chunks in source code',
            'sospicious token_x': 'Number of sospicious token in source code',
            'IP_y': 'Number of IP adress in metadata',
            'base64_y': 'Number of base64 chunks in metadata',
            'sospicious token_y': 'Number of sospicious token in metadata'
        }
        database.rename(columns={k: v for k, v in rename_map.items() if k in database.columns}, inplace=True)

        return database
