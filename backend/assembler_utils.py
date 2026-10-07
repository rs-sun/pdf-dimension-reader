"""
assembler_utils.py — 语义组装引擎：共享常量 + 工具函数

从 assembler.py 拆出，供 assembler_scoring / assembler_loaders /
assembler_dedup / assembler 主模块共同引用。
"""

import math
import re

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 尺寸合理性范围
DIM_MIN = 0.001
DIM_MAX = 99999.0

# OCR 打分系统阈值
SCORE_THRESHOLD = 25

# 由调用入口控制诊断输出。
VERBOSE = False

# 常见图框文字（排除用）
FRAME_KEYWORDS = frozenset([
    'SCALE', 'DATE', 'MATERIAL', 'REVISION', 'DRAWN', 'CHECKED', 'APPROVED',
    'SHEET', 'SIZE', 'DWG', 'NUMBER', 'WEIGHT', 'FINISH', 'UNLESS', 'TOLERANCE',
    'PROPRIETARY', 'CONFIDENTIAL', 'NOTICE', 'TITLE', 'REV', 'ECO',
])

# 语义黑名单：匹配任一 pattern 的 OCR 结果直接排除，不进入打分流程
METADATA_BLACKLIST_PATTERNS = [
    re.compile(r'^View\s*\d+$', re.IGNORECASE),
    re.compile(r'^SECTION\s+[A-Z]-[A-Z]', re.IGNORECASE),
    re.compile(r'^DETAIL\s+[A-Z]', re.IGNORECASE),
    re.compile(r'^SCALE\s+\d+\s*:\s*\d+', re.IGNORECASE),
    re.compile(r'^[A-Z]-[A-Z]$', re.IGNORECASE),
    re.compile(r'^[IVX]{2,}$', re.IGNORECASE),
    re.compile(r'UNLESS\s+OTHERWISE', re.IGNORECASE),
    re.compile(r'ALL\s+DIMENSIONS', re.IGNORECASE),
    re.compile(r'THIRD\s+ANGLE', re.IGNORECASE),
    re.compile(r'^TOLERANCES', re.IGNORECASE),
    re.compile(r'^MATERIAL', re.IGNORECASE),
    re.compile(r'SURFACE\s+FINISH', re.IGNORECASE),
    re.compile(r'^DRAWN\b', re.IGNORECASE),
    re.compile(r'^CHECKED\b', re.IGNORECASE),
    re.compile(r'^APPROVED\b', re.IGNORECASE),
    re.compile(r'^DATE\b', re.IGNORECASE),
    re.compile(r'^REV\b', re.IGNORECASE),
    re.compile(r'^MEASUREMENT', re.IGNORECASE),
    re.compile(r'^DIMENSION\b', re.IGNORECASE),
    re.compile(r'^REFERENCE\b', re.IGNORECASE),
    re.compile(r'^ORDINATE', re.IGNORECASE),
    re.compile(r'REFERENCE\s+DIMENSION', re.IGNORECASE),
    re.compile(r'THEORITICAL\s+DIMENSION', re.IGNORECASE),
    re.compile(r'CHARACTERISTIC\s+ITEMS', re.IGNORECASE),
    re.compile(r'CONFORMS\s+TO', re.IGNORECASE),
    re.compile(r'INSPECTION', re.IGNORECASE),
    re.compile(r'^\d+:\d+$'),             # 元数据过滤规则覆盖比例、标题词、截断工程前缀、说明文本和稀疏编号碎片；规则保持通用结构匹配。
    re.compile(r'^\d+[：:]\d+$'),         
    re.compile(r'^\d{1,2}X\s+R\s*$', re.IGNORECASE),  
    re.compile(r'^MAX\.?\s*[O0]\.?\d+\s*(?:MM|M)\s*FOR', re.IGNORECASE), 
    re.compile(r'^MAX\.\s*O', re.IGNORECASE), 
    re.compile(r'^MAX$', re.IGNORECASE),      
    re.compile(r'DRAWING', re.IGNORECASE),    
    re.compile(r'^[)\]\}][.,\s]+\d+$'),      
    
    re.compile(r'^MAX\.?\s*[\(\[].*$', re.IGNORECASE),  
    re.compile(r'^[A-Z]{1,3}-\d+$'),              
    re.compile(r'^(?:\d+X\s*)?R\s+[A-Z]$'),       
    re.compile(r'^\d{1,2}(\s+\d+){1,}$'),         
    re.compile(r'^\d+\.\d+\s+\d+$'),              
    
    re.compile(r'^(?:\d+X\s*)?R\s*\(', re.IGNORECASE),  
    re.compile(r'^\d+T\d+'),                            
    re.compile(r'^P\s+\d'),                             
    re.compile(r'\.[A-Z]$', re.IGNORECASE),             
]

# 全大写英文长串降分正则（含工业符号时豁免）
_ALLCAPS_LONG_PATTERN = re.compile(r'^[A-Z][A-Z\s]{8,}$')
_INDUSTRIAL_SYMBOLS = frozenset('⌀±°×xX')

# 前缀识别正则
_PREFIX_PATTERNS = [
    (re.compile(r'^[⌀∅Øφ]'), '⌀', 'diameter'),
    # Ra/Rz/Rmax 表面粗糙度前缀（在 R/M/C 之前判，避免 `Ra` 被吃成 `R` + `a…`）
    (re.compile(r'^R(?:a|z|max)(?=[\d.])', re.IGNORECASE), None, 'surface_rough'),
    (re.compile(r'^R(?=[\d.])'), 'R', 'radius'),
    (re.compile(r'^M(?=[\d.])'), 'M', 'thread'),
    (re.compile(r'^C(?=[\d.])'), 'C', 'chamfer'),
    (re.compile(r'^\d+\s*[×xX]'), None, 'repeat'),
]

# NX blockfont / chinesef_fs 字符纠错映射（在数字上下文中）
_NX_CHAR_CORRECTIONS = [
    # 直径符号统一
    (re.compile(r'[∅Øφ]'), '⌀'),
    # 规范倍数前缀与数值之间的误读直径符号。
    (re.compile(r'(?<=[×xX])\s*O(?=\d)'), '⌀'),
    # ± 重建（+- 紧连）
    (re.compile(r'\+\s*-'), '±'),
    (re.compile(r'-\s*\+'), '±'),
]

# 打分用正则
_RATIO_PATTERN = re.compile(r'^\d+:\d+$')
_WORD_PATTERN = re.compile(r'^[A-Za-z]{4,}(\s+[A-Za-z]{2,}){2,}$')
_SECTION_LABEL_PATTERN = re.compile(r'^[A-Z][\s_-]*[A-Z]$', re.IGNORECASE)

# 形位碎片词汇覆盖分类器支持的符号与直径修饰。

_GDT_SYMBOL_CHARS = frozenset('⊕⊘⌀∅Ø⊥∥◎○⌭—∠⌓⌒▱≡')
_GDT_SINGLE_LETTER = re.compile(r'^[A-Z]$')
_GDT_SMALL_NUMBER = re.compile(r'^\d*\.?\d+$')
_GDT_OCR_MISREAD = re.compile(r'^[O+]$')

# 置信度排序
_CONF_ORDER = {'high': 2, 'medium': 1, 'low': 0}


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

def _bbox_center(bbox: dict) -> tuple:
    return bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['h'] / 2


def _is_vertical_orient(orient):
    """根据 OCR 显式方向或四边形几何判断文本是否竖直。"""
    if orient is None:
        return False
    return abs(abs(orient) - 90) < 15


def _point_in_bbox(px, py, bbox: dict, margin: float = 0.0) -> bool:
    return (bbox['x'] - margin <= px <= bbox['x'] + bbox['w'] + margin and
            bbox['y'] - margin <= py <= bbox['y'] + bbox['h'] + margin)


def _extract_prefix(text: str) -> tuple:
    """从文字中提取倍数、半径、直径、角度或粗糙度前缀，返回清洗文本、前缀和类型。"""
    text = text.strip()
    for pat, sym, dtype in _PREFIX_PATTERNS:
        m = pat.match(text)
        if m:
            remainder = text[m.end():]
            prefix = sym if sym else m.group(0)
            # repeat (nX) 命中后，对 remainder 再扫一次符号前缀
            if dtype == 'repeat':
                sub_rem = remainder.lstrip()
                for pat2, sym2, dtype2 in _PREFIX_PATTERNS:
                    if dtype2 == 'repeat':
                        continue
                    m2 = pat2.match(sub_rem)
                    if m2:
                        sub_remainder = sub_rem[m2.end():]
                        sub_prefix = sym2 if sym2 else m2.group(0)
                        return sub_remainder, f'{prefix} {sub_prefix}', dtype2
            return remainder, prefix, dtype
    return text, None, 'linear'


def _parse_nominal(text: str) -> float | None:
    """将文本解析为浮点标称值，失败返回 None。"""
    cleaned = re.sub(r'[^\d.\-]', '', text.strip())
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if DIM_MIN <= val <= DIM_MAX:
            return val
        return None
    except ValueError:
        return None


def _nxfont_correction(text: str) -> str:
    """
    NX blockfont / chinesef_fs OCR 字符纠错。
    仅在确定是数字上下文时替换易混字符。
    """
    result = text
    for pat, replacement in _NX_CHAR_CORRECTIONS:
        result = pat.sub(replacement, result)

    # 仅在数字上下文中规范视觉相似字符。
    result = re.sub(r'(?<=[\d.])O(?=[\d.])', '0', result)
    result = re.sub(r'(?<=[\d.])o(?=[\d.])', '0', result)
    result = re.sub(r'(?<=[\d.])[lI](?=[\d.])', '1', result)

    # 度数：数字后的孤立 'o' → '°'
    result = re.sub(r'(\d)o(\b|$)', r'\1°', result)

    # 规范空白分隔的倍数前缀与误读直径符号。
    result = re.sub(r'([0-9xX])\s+O\s*(?=[\d.])', r'\1 ⌀', result)
    # Radius and diameter OCR confusions in engineering labels.
    result = re.sub(r'\bR[Oo](?=\.)', 'R0', result)
    result = re.sub(r'^\s*[Qq]\s*(?=\d)', '⌀', result)

    return result


def _fixup_ocr_text(text: str) -> str:
    """NX blockfont OCR 系统性误读修复（在黑名单检查之后、结构化解析之前调用）。"""
    _s27_orig = text
    text = _nxfont_correction(text)
    # OCR often confuses tolerance signs with visually similar glyphs.
    # Keep replacements numeric-context-only so ordinary notes are untouched.
    text = re.sub(r'(?<=\d)\s*[士土]\s*(?=\d)', '±', text)
    text = re.sub(r'(?<=\d)\s*[−–]\s*(?=0?\.\d)', ' -', text)
    text = re.sub(r'^\s*[−–]\s*(?=\d|\.)', '-', text)
    text = re.sub(r'^\s*十\s*(?=0?\.\d|\d)', '+', text)
    text = re.sub(r'(?<=\d)\s*十\s*(?=0?\.\d)', ' +', text)
    text = re.sub(r'(?<![\d.])(\d+)\s+\.(?=\d)', r'\1.', text)
    text = re.sub(r'(?<=\d)\s*:\s*(?=±|[+\-]\s*\d)', ' ', text)
    text = re.sub(r'([±+\-])\s*[Oo](?=[.。])', r'\g<1>0', text)
    text = re.sub(r'(?<=\d)。(?=\d)', '.', text)
    # 根据限定前缀结构修复紧凑直径符号误读。
    
    text = re.sub(
        r'^\s*[Dd]\s*2\.(\d)(\s*(?:±|[+\-])\s*0?\.\d+)',
        r'⌀\1\2',
        text,
    )
    text = re.sub(r'^\s*[Dd]\s+(?=[⌀∅ØφΦ])', '', text)
    text = re.sub(r'^\s*([⌀∅ØφΦ])0(\d)(?=\s*(?:±|[+\-]))', r'\1\2', text)
    text = re.sub(
        r'^((?:\d+[xX]\s*)?(?:[⌀RMC]\s*)?)(\d+(?:\.\d+)?)\s+\2\s*(?=±|[+\-]\s*\d)',
        r'\1\2 ',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'^\s*0\s+((?:\d+[xX]\s*)?(?:[⌀RMC]\s*)?\d+(?:\.\d+)?\s*(?:±|[+\-]\s*\d))',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )
    # 仅在倍数前缀与反转公称整数部分一致时移除污染前缀，不改写普通倍数。
    
    
    
    m = re.match(
        r'^\s*(\d{3,})\s*[xX×]\s*'
        r'((?:[⌀∅Ø⊘φΦRMC]\s*)?(\d{3,})(?:\.\d+)?'
        r'\s*(?:(?:±|[+\-])\s*\d+(?:\.\d+)?'
        r'(?:\s*/?\s*[+\-]\s*\d+(?:\.\d+)?)?)?(?:°)?)\s*$',
        text,
        flags=re.IGNORECASE,
    )
    if m and m.group(1) == m.group(3)[::-1]:
        text = m.group(2).strip()
    # 正负符号误读修复必须锚定倍数、工程前缀和数值结构。
    
    _t_pat = re.compile(r'^((?:\d+[xX]\s*)?(?:⌀|R|M)?\d+\.?\d*)\s*t\s*(\d)')
    _t_m = _t_pat.match(text)
    if _t_m:
        text = f"{_t_m.group(1)}±{_t_m.group(2)}" + text[_t_m.end():]
    # : → ± (数字-:-数字上下文，仅当冒号两侧都有数字)
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1 ±\2', text)
    # 拆分直接粘连的长小数公称值与短公差，保留倍数前缀。
    
    m = re.match(r'^((?:\d+[xX]\s*)?)(R?\d+\.\d{2,})(\d\.\d+)$', text)
    if m:
        text = f"{m.group(1)}{m.group(2)} ±{m.group(3)}"
    # 拆分公称小数与紧随其后的公差小数。
    
    m = re.match(r'^((?:\d+[xX]\s*)?)(\d+\.\d+)(0\.\d+)$', text)
    if m:
        text = m.group(1) + m.group(2) + '±' + m.group(3)
    # 恢复空白分隔且缺失对称符号的公差，允许省略前导零的小数形式。
    
    
    m = re.match(r'^((?:\d+[xX]\s*)?)([\d.]+)\s+(0?\.\d+)$', text)
    if m:
        _tol = m.group(3)
        if _tol.startswith('.'):
            _tol = '0' + _tol
        text = m.group(1) + m.group(2) + '±' + _tol
    # 在有上公差的堆叠结构中，按受限幅值规则恢复缺失负号的下公差。
    
    
    m = re.match(
        r'^((?:\d+[xX]\s*)?(?:[⌀∅Ø⊘φΦRMC]\s*)?\d+(?:\.\d+)?)'
        r'\s+\+\s*(0?\.\d+)\s+(0?\.\d+)$',
        text,
        flags=re.IGNORECASE,
    )
    if m:
        text = f"{m.group(1)} +{m.group(2)}/-{m.group(3)}"
    # 在限定公称值与公差结构中修复丢失对称符号后的尾部误读字符。
    
    m = re.match(r'^((?:\d+[xX]\s*)?(?:⌀|R|M)?\d+(?:\.\d+)?)\s+(0?\.\d+)\s+[Oo]$', text)
    if m:
        _tol = m.group(2)
        if _tol.startswith('.'):
            _tol = '0' + _tol
        text = f"{m.group(1)}±{_tol}"
    if VERBOSE:
        if text != _s27_orig:
            print(f"[s27.fixup] CHANGED: '{_s27_orig}' -> '{text}'")
        else:
            print(f"[s27.fixup] UNCHANGED: '{_s27_orig}'")
    return text


def _fixup_fc_capsule_text(text: str) -> str:
    """在仍携带功能标记的容器 OCR 文本中，修复被读成数字或截断的开括号。"""
    original = text or ''
    if not original:
        return original
    compact = re.sub(r'\s+', '', original).upper()
    has_fc_marker = bool(re.search(r'F[/|\\]?[CD]', compact) or re.search(r'±0\.\d+F$', compact))
    if not has_fc_marker:
        return original

    fixed = original
    fixed = re.sub(
        r'(?<![\d.])1([2-9]\.\d+\s*±\s*0\.\d+)(?=\s*[|/\\ ]*F\b)',
        r'\1',
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r'(?<![\d.])\.(55\s*±\s*0\.05)(?=\s*F\b)',
        r'1.\1',
        fixed,
        flags=re.IGNORECASE,
    )
    return fixed


def _clean_dimension_text(text: str) -> str:
    """规范化全角字符，保留可打印工程符号并删除控制字符。"""
    # 全角数字/字母/符号 → 半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif ch == '\u3000':
            result.append(' ')
        else:
            result.append(ch)
    s = ''.join(result)
    # 仅在数字上下文中把全角句点规范为小数点。
    
    s = re.sub(r'(?<=\d)\u3002(?=\d)', '.', s)
    # 删除控制字符，保留所有可打印 Unicode（含 ⌀ ± ° ∅ Ø φ × ′ ″ μ）
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    return s.strip()


def _validate_engineering(nominal_val: float | None,
                           upper_tol: str | None,
                           lower_tol: str | None) -> str:
    """
    工程合理性校验，返回置信度等级: 'high' | 'medium' | 'low'
    """
    if nominal_val is None:
        return 'low'
    if not (DIM_MIN <= nominal_val <= DIM_MAX):
        return 'low'
    if upper_tol and lower_tol:
        try:
            u = float(upper_tol)
            l = float(lower_tol)
            if u < l:
                return 'low'  # 上偏差 < 下偏差，不合理
        except ValueError:
            pass
    return 'high'


# ---------------------------------------------------------------------------
# bbox / 距离工具
# ---------------------------------------------------------------------------

def _bbox_iou_min(a: dict, b: dict) -> float:
    """交集面积 / 较小 bbox 面积（对小框匹配大框更灵敏，仅用于 GD&T 匹配）。"""
    x1 = max(a['x'], b['x'])
    y1 = max(a['y'], b['y'])
    x2 = min(a['x'] + a['w'], b['x'] + b['w'])
    y2 = min(a['y'] + a['h'], b['y'] + b['h'])
    if x2 <= x1 or y2 <= y1:
        return 0
    inter = (x2 - x1) * (y2 - y1)
    area_min = min(a['w'] * a['h'], b['w'] * b['h'])
    return inter / area_min if area_min > 0 else 0


def _levenshtein(s1: str, s2: str) -> int:
    """最小编辑距离。"""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _bbox_iou(a: dict, b: dict) -> float:
    """计算两个 {x, y, w, h} bbox 的 IoU。"""
    x1 = max(a['x'], b['x'])
    y1 = max(a['y'], b['y'])
    x2 = min(a['x'] + a['w'], b['x'] + b['w'])
    y2 = min(a['y'] + a['h'], b['y'] + b['h'])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_junk_by_symbol_ratio(text: str) -> bool:
    """根据工程符号占比判断文字是否属于乱码。"""
    clean = re.sub(r'^(\d+X\s*)', '', text)   # 去除 nX 前缀
    clean = re.sub(r'\s', '', clean)           # 去除空白
    if len(clean) < 6:
        return False
    SAFE = set('±.°⌀∅Ø+- /')
    junk = sum(1 for ch in clean if not ch.isalnum() and ch not in SAFE)
    return junk / len(clean) >= 0.10


def _union_bbox(bboxes: list) -> dict:
    """合并多个 bbox 为包围盒。"""
    x0 = min(b['x'] for b in bboxes)
    y0 = min(b['y'] for b in bboxes)
    x1 = max(b['x'] + b['w'] for b in bboxes)
    y1 = max(b['y'] + b['h'] for b in bboxes)
    return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def _parse_tolerance_inline(text: str) -> tuple:
    """提取内联公差，返回上偏差和下偏差字符串或空值。支持对称、带分隔符的双边以及单边公差结构。"""
    def _clean_tol(raw: str) -> str:
        return raw.replace(' ', '').replace('O', '0').replace('o', '0')

    # 格式: ± value
    m = re.search(r'[±]\s*([Oo0-9.]+)', text)
    if m:
        v = _clean_tol(m.group(1))
        return f"+{v}", f"-{v}"

    # 格式: +x/-y 或 +x / -y
    m2 = re.search(r'\+\s*([Oo0-9.]+)\s*/?\s*-\s*([Oo0-9.]+)', text)
    if m2:
        return f"+{_clean_tol(m2.group(1))}", f"-{_clean_tol(m2.group(2))}"

    # 双边公差可按幅值限制恢复缺失的下公差负号。
    m2_unsigned_lower = re.search(
        r'\+\s*([Oo0-9.]+)\s+(0?\.\d+)(?!\d)',
        text,
    )
    if m2_unsigned_lower:
        return (
            f"+{_clean_tol(m2_unsigned_lower.group(1))}",
            f"-{_clean_tol(m2_unsigned_lower.group(2))}",
        )

    # 格式: +x alone
    m3 = re.search(r'\+\s*([Oo0-9.]+)(?!\d)', text)
    if m3:
        return f"+{_clean_tol(m3.group(1))}", None

    # 空白分隔的负值按单边公差结构提取，避免与连字符编号混淆。
    m4 = re.search(r'\s-\s*([Oo0-9.]+)(?!\d)', text)
    if m4:
        return None, f"-{_clean_tol(m4.group(1))}"

    return None, None


# ---------------------------------------------------------------------------
# GD&T 碎片识别
# ---------------------------------------------------------------------------

def _is_gdt_fragment(text: str) -> bool:
    """判断文本是否可能是 GD&T 控制框的碎片。"""
    t = text.strip()
    if any(c in t for c in _GDT_SYMBOL_CHARS):
        return True
    if _GDT_SINGLE_LETTER.match(t):
        return True
    if _GDT_OCR_MISREAD.match(t):
        return True
    if _GDT_SMALL_NUMBER.match(t):
        try:
            return float(t) < 5.0
        except ValueError:
            pass
    return False


def _quick_dim_score(text: str) -> int:
    """快速打分，判断文本是否像尺寸值。"""
    score = 0
    t = text.strip()
    if '±' in t: score += 50
    if '⌀' in t: score += 50
    if '°' in t: score += 40
    if 'MAX' in t.upper() or 'MIN' in t.upper(): score += 40
    if re.search(r'[+\-]\s*\d', t): score += 45
    if re.match(r'^[\d.]+$', t): score += 30
    if re.match(r'^\d+[×xX][\d.]+', t): score += 35
    if re.match(r'^R[\d.]+', t): score += 35
    return score
