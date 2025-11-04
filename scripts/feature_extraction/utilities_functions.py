import base64, os, re, socket, json
import chardet
from pathlib import Path
from collections import Counter
import math

# Biểu thức chính quy để bắt các chuỗi có thể là base64 (dùng sau khi lọc bằng is_base64)
b64regex = re.compile(r'[a-zA-Z0-9=/\+]*')

# Biểu thức chính quy để tìm các địa chỉ IPv4 (dạng số)
ipaddr_regex = re.compile(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}')

# Biểu thức chính quy phức tạp để dò URL (dùng trong contains_URL)
urls_regex = re.compile(r"""((?:(?:http|http|ssh|ftp|sftp|ws|wss|dns|file|git|jni|imap|ldap|ldaps|nfs|smb|smbs|telnet|udp|vnc)?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|uk|ac)\b/?(?!@)))""")

def detect_encoding(data: bytes) -> str:
    """
    Phát hiện mã hóa của dữ liệu nhị phân sử dụng chardet.
    Nếu không thể phát hiện, mặc định là 'utf-8'.
    """
    result = chardet.detect(data)
    return result['encoding'] if result['encoding'] else 'utf-8'

def is_base64(sb):
    """
    Kiểm tra một chuỗi có phải base64 hợp lệ hay không.
    - Nhận chuỗi hoặc bytes, giải mã bằng base64 và kiểm tra tính in được.
    - Trả về True nếu chuỗi là base64 chuẩn (sau decode-encode vẫn giữ nguyên).
    """
    try:
        if isinstance(sb, str):
            # Nếu có unicode, bytes(sb, 'ascii') sẽ ném ngoại lệ -> trả False
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        decoded_string = base64.b64decode(sb_bytes).decode("utf-8")
        # loại bỏ whitespace dư thừa để kiểm tra printable
        decoded_string = ' '.join(decoded_string.split())
        if (decoded_string.isprintable()):
            return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        else:
            return False
    except Exception:
        return False

def is_IPAddress(s):
    """
    Kiểm tra một token có thể là địa chỉ IP (IPv4).
    - Chia theo ':' để loại bỏ cổng nếu có (ví dụ '1.2.3.4:8080').
    - Dùng socket.inet_aton để kiểm tra tính hợp lệ.
    """
    try:
        socket.inet_aton(s.split(":")[0])
        return True
    except socket.error:
        return False


def contains_base64(string):
    """
    Tìm các chuỗi có khả năng base64 trong một chuỗi lớn.
    - Dùng b64regex để tách các từ khả dĩ, sau đó dùng is_base64 để xác thực.
    - Trả về danh sách các chuỗi base64 (unique, theo thứ tự xuất hiện).
    """
    list_of_words = list(dict.fromkeys(b64regex.findall(string)))

    base64_strings = []
    for w in list_of_words:
        if len(w) > 1:
            if is_base64(w):
                base64_strings.append(w)
    return base64_strings


def contains_IPAddress(string):
    """
    Tìm các chuỗi có dạng IP trong một chuỗi.
    - Dùng ipaddr_regex để tách candidates, sau đó dùng is_IPAddress để xác thực.
    - Trả về danh sách địa chỉ IP hợp lệ (unique).
    """
    list_of_words = list(dict.fromkeys(ipaddr_regex.findall(string)))
    IPAddress_strings = []
    for w in list_of_words:
        if (len(w) > 6):
            if is_IPAddress(w):
                IPAddress_strings.append(w)
    return IPAddress_strings

def contains_URL(string):
    """
    Tìm các URL trong một chuỗi bằng biểu thức chính quy urls_regex.
    - urls_regex.findall trả về tuples/nhóm, ta chọn phần dài nhất trong mỗi match group.
    - Trả về danh sách các URL tìm được.
    """
    list_of_matches = urls_regex.findall(string)
    list_of_candidates = []
    for m in list_of_matches:
        list_of_candidates.append(max(list(m),key=len))
    return list_of_candidates

def contains_dangerous_token(string,dangerous_tok):
    """
    Tìm các token nguy hiểm trong một chuỗi metadata hoặc code.
    - dangerous_tok là một iterable (ví dụ từ file JSON).
    - Trả về danh sách token tìm được (có thể trùng nhau nếu xuất hiện nhiều lần trong chuỗi).
    """
    findings_list = [] 
    for susp in dangerous_tok:          
        if susp in string:
                findings_list.append(susp)
    return findings_list


def find_files_of_ext(root, ext):
    """
    Duyệt thư mục root (đệ quy) và trả về danh sách đường dẫn đầy đủ
    của các file có phần mở rộng ext (ví dụ '.js', '.json').
    - Trả về list[str].
    """
    return [str(Path(dir, file_)) for dir, subdir, files in os.walk(root) for file_ in files if Path(file_).suffix == ext] 

def shannon_entropy(data, base=2):
    """
    Tính Shannon entropy của một sequence (string).
    - base mặc định là 2 (log2), có thể chuyển thành base khác (ví dụ 4).
    - Trả về entropy dưới dạng số thực.
    """
    entropy = 0.0
    if len(data) > 0:
        cnt = Counter(data)
        length = len(data)
        for count in cnt.values():
                entropy += (count / length) * math.log(count / length, base)
        entropy = entropy * -1.0
    return (entropy)

def obfuscation(list_id,symbols=['u','d','l','s']):
    """
    Đo độ 'obfuscation' dựa trên sự xuất hiện các loại ký tự
    trong các identifier đã được generalize (ví dụ 'udlss').
    - Tính số identifier mà bộ ký tự duy nhất của nó chứa các kết hợp
      ký tự nằm trong các mẫu nhất định (giống logic gốc).
    - Trả về một số nguyên (count).
    """
    unique_symbols_id=[]
    # get unique symbols from each identifiers
    for i in range(0,len(list_id)):
        unique_symbols_id.append("".join(set(list_id[i])))
    # initialize the count for obfuscation:
    obs=0
    for i in range(0,len(unique_symbols_id)):
        # Upper case, digit, lower case, symbol
        if (check(unique_symbols_id[i],symbols))==['True', 'True', 'True', 'True']:
            obs+=1
        # upper case, digit, symbol
        if (check(unique_symbols_id[i],symbols))==['True', 'True', 'False', 'True']:
            obs+=1
        # digit, lower case, symbol
        if (check(unique_symbols_id[i],symbols))==['False', 'True', 'True', 'True']:
            obs+=1
        # digit, symbol
        if (check(unique_symbols_id[i],symbols))==['False', 'True', 'False', 'True']:
            obs+=1
    
        
    return(obs)

def check(s, arr):
    """
    Kiểm tra sự hiện diện của mỗi symbol trong arr trong chuỗi s.
    - Trả về list các chuỗi 'True' hoặc 'False' giống logic gốc.
    """
    result = []
    for i in arr:
        if i in s:
            result.append("True")
        else:
            result.append("False")
    return result

def gen_language_4(value):
    """
    Generalization language với 4 ký tự:
    - d: chữ số
    - u: chữ hoa
    - l: chữ thường
    - s: ký tự đặc biệt khác
    Trả về chuỗi pattern tương ứng (ví dụ 'uldds').
    """
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        else:
            pattern += 's'
    
    return pattern

# generalization languages
def gen_language_3(value):
    """
    Generalization language với 3 ký tự:
    - d: chữ số
    - l: chữ thường hoặc chữ hoa (tất cả chữ cái => 'l')
    - s: ký tự đặc biệt
    """
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isalpha():
            pattern += 'l'
        else:
            pattern += 's'
    
    return (pattern)


def gen_language_8(value):
    """
    Generalization language với 8 ký tự:
    - d: digit
    - u: uppercase
    - l: lowercase
    - p: dot '.'
    - h: slash '/'
    - a: hyphen '-'
    - i: one of '|', '%', '$', '~', '?'
    - s: other symbols
    """
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        elif c=='.':
            pattern +='p'
        elif c=='/':
            pattern +='h'
        elif c=='-':
            pattern +='a' 
        elif c=='|' or c=='%' or c=='$'or c=='~'or c=='?':
            pattern +='i'
        else:
            pattern += 's'
    
    return (pattern)

def gen_language_16(value):
    """
    Generalization language với 16 ký tự (mở rộng nhiều ký tự đặc biệt hơn).
    Quy ước tương tự gen_language_8 nhưng thêm mapping cho %, |, =, :, $, >, <, ~, ?.
    """
    pattern = ''
    value = list(str(value))
    for c in value:
        if c.isnumeric():
            pattern += 'd'
        elif c.isupper():
            pattern += 'u'
        elif c.islower():
            pattern +='l'
        elif c=='.':
            pattern +='p'
        elif c=='/':
            pattern +='h'
        elif c=='-':
            pattern +='a'
        elif c=='%':
            pattern +='p'
        elif c=='|':
            pattern +='i'
        elif c=='=':
            pattern +='e'
        elif c==':':
            pattern +='c'
        elif c=='$':
            pattern +='m'
        elif c=='>':
            pattern +='g'
        elif c=='<':
            pattern +='o'
        elif c=='~':
            pattern +='t'
        elif c=='?':
            pattern +='q'
        else:
            pattern += 's'
    
    return (pattern)

# Tính Shannon entropy của chuỗi (dùng trong pypi_feature_extractor.py)
def compute_entropy(text):
    """Tính Shannon entropy của chuỗi."""
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum(p * math.log2(p) for p in prob)
    return round(entropy, 4)

# Phát hiện URL, IP, email trong chuỗi (dùng trong pypi_feature_extractor.py)
def detect_urls(text):
    """Phát hiện URL trong chuỗi."""
    pattern = r'https?://[^\s\'"]+'
    return re.findall(pattern, text)

# Phát hiện địa chỉ IP trong chuỗi (dùng trong pypi_feature_extractor.py)
def detect_ips(text):
    """Phát hiện địa chỉ IPv4."""
    pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    return re.findall(pattern, text)

# Phát hiện địa chỉ email trong chuỗi (dùng trong pypi_feature_extractor.py)
def detect_emails(text):
    """Phát hiện địa chỉ email."""
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

# Đếm số token nguy hiểm trong chuỗi (dùng trong pypi_feature_extractor.py)
def count_dangerous_tokens(code, dangerous_tokens):
    """Đếm số token nguy hiểm xuất hiện trong mã nguồn."""
    count = 0
    for token in dangerous_tokens:
        # đếm không phân biệt hoa thường
        count += len(re.findall(rf'\b{re.escape(token)}\b', code, flags=re.IGNORECASE))
    return count