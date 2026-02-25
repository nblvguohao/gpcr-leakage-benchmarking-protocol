#!/usr/bin/env python3
"""
GPCRdb数据获取脚本 - 从GPCRdb REST API获取GPCR蛋白序列 + GtoPdb偶联注释
用于构建GPCR-Gq偶联预测的大规模训练数据集

数据来源:
  - 蛋白序列: GPCRdb REST API (https://gpcrdb.org/services/)
  - 偶联标签: Guide to Pharmacology (GtoPdb) / GPCRdb curated data
  - 参考: Inoue et al. 2019, Okashah et al. 2019

输出: data/gpcrdb_coupling_dataset.csv
"""
import os
import sys
import json
import time
import requests
import csv
from datetime import datetime

# ================================================================
# 配置
# ================================================================
BASE_URL = "https://gpcrdb.org/services"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpcrdb_coupling_dataset.csv")

# 只获取人类受体 (用于训练)
TARGET_SPECIES = "Homo sapiens"

# 要获取的GPCR家族 (排除嗅觉受体和孤儿受体)
TARGET_FAMILIES = [
    "001_001",  # Aminergic receptors
    "001_002",  # Peptide receptors
    "001_003",  # Protein receptors
    "001_004",  # Lipid receptors
    "001_005",  # Melatonin receptors
    "001_006",  # Nucleotide receptors
    "001_007",  # Steroid receptors
    "001_008",  # Alicarboxylic acid receptors
    "001_009",  # Sensory receptors (opsins)
    "002_001",  # Class B1 Peptide receptors
    "004_001",  # Class C Ion receptors
    "004_002",  # Class C Amino acid receptors
]

# ================================================================
# Gq/11偶联注释 (来自GtoPdb + GPCRdb + Inoue et al. 2019)
# 格式: GPCRdb entry_name前缀 -> (couples_gq, coupling_type)
# coupling_type: 'primary' = 主要偶联, 'secondary' = 次要偶联, 'none' = 不偶联
# ================================================================
GQ_COUPLING_MAP = {
    # === Aminergic receptors ===
    # 5-HT receptors
    "5ht1a": ("none", "Gi/o primary"),
    "5ht1b": ("none", "Gi/o primary"),
    "5ht1d": ("none", "Gi/o primary"),
    "5ht1e": ("none", "Gi/o primary"),
    "5ht1f": ("none", "Gi/o primary"),
    "5ht2a": ("primary", "Gq/11 primary"),
    "5ht2b": ("primary", "Gq/11 primary"),
    "5ht2c": ("primary", "Gq/11 primary"),
    "5ht4":  ("none", "Gs primary"),
    "5ht5a": ("none", "Gi/o primary"),
    "5ht6":  ("none", "Gs primary"),
    "5ht7":  ("none", "Gs primary"),
    # Muscarinic receptors
    "acm1":  ("primary", "Gq/11 primary"),
    "acm2":  ("none", "Gi/o primary"),
    "acm3":  ("primary", "Gq/11 primary"),
    "acm4":  ("none", "Gi/o primary"),
    "acm5":  ("primary", "Gq/11 primary"),
    # Adrenoceptors
    "ada1a": ("primary", "Gq/11 primary"),
    "ada1b": ("primary", "Gq/11 primary"),
    "ada1d": ("primary", "Gq/11 primary"),
    "ada2a": ("none", "Gi/o primary"),
    "ada2b": ("none", "Gi/o primary"),
    "ada2c": ("none", "Gi/o primary"),
    "adrb1": ("none", "Gs primary"),
    "adrb2": ("none", "Gs primary"),
    "adrb3": ("none", "Gs primary"),
    # Dopamine receptors
    "drd1":  ("none", "Gs primary"),
    "drd2":  ("none", "Gi/o primary"),
    "drd3":  ("none", "Gi/o primary"),
    "drd4":  ("none", "Gi/o primary"),
    "drd5":  ("none", "Gs primary"),
    # Histamine receptors
    "hrh1":  ("primary", "Gq/11 primary"),
    "hrh2":  ("none", "Gs primary"),
    "hrh3":  ("none", "Gi/o primary"),
    "hrh4":  ("none", "Gi/o primary"),
    # Trace amine receptors
    "ta1r":  ("none", "Gs primary"),

    # === Peptide receptors ===
    # Angiotensin
    "agtr1": ("primary", "Gq/11 primary"),
    "agtr2": ("none", "Gi/o primary"),
    # Apelin
    "aplnr": ("none", "Gi/o primary"),
    # Bombesin
    "brs3":  ("primary", "Gq/11 primary"),
    "grpr":  ("primary", "Gq/11 primary"),
    "nmbr":  ("primary", "Gq/11 primary"),
    # Bradykinin
    "bkrb1": ("primary", "Gq/11 primary"),
    "bkrb2": ("primary", "Gq/11 primary"),
    # Cholecystokinin
    "cckar": ("primary", "Gq/11 primary"),
    "cckbr": ("primary", "Gq/11 primary"),
    # Complement peptide
    "c3ar":  ("secondary", "Gi/o primary, Gq secondary"),
    "c5ar1": ("secondary", "Gi/o primary, Gq secondary"),
    "c5ar2": ("none", "Gi/o primary"),
    # Endothelin
    "ednra": ("primary", "Gq/11 primary"),
    "ednrb": ("primary", "Gq/11 primary"),
    # Formylpeptide
    "fpr1":  ("secondary", "Gi/o primary, Gq secondary"),
    "fpr2":  ("secondary", "Gi/o primary, Gq secondary"),
    "fpr3":  ("secondary", "Gi/o primary, Gq secondary"),
    # Galanin
    "galr1": ("none", "Gi/o primary"),
    "galr2": ("primary", "Gq/11 primary"),
    "galr3": ("none", "Gi/o primary"),
    # Ghrelin
    "ghsr":  ("primary", "Gq/11 primary"),
    # GnRH
    "gnrhr": ("primary", "Gq/11 primary"),
    # Kisspeptin
    "kissr": ("primary", "Gq/11 primary"),
    # MCH
    "mchr1": ("secondary", "Gi/o primary, Gq secondary"),
    "mchr2": ("primary", "Gq/11 primary"),
    # Melanocortin
    "mc1r":  ("none", "Gs primary"),
    "mc2r":  ("none", "Gs primary"),
    "mc3r":  ("none", "Gs primary"),
    "mc4r":  ("none", "Gs primary"),
    "mc5r":  ("none", "Gs primary"),
    # Motilin
    "mtlr":  ("primary", "Gq/11 primary"),
    # Neuromedin U
    "nmur1": ("primary", "Gq/11 primary"),
    "nmur2": ("primary", "Gq/11 primary"),
    # Neuropeptide FF
    "npff1": ("none", "Gi/o primary"),
    "npff2": ("none", "Gi/o primary"),
    # NPS
    "npsr":  ("secondary", "Gs primary, Gq secondary"),
    # Neuropeptide W/B
    "npbw1": ("secondary", "Gi/o primary, Gq secondary"),
    "npbw2": ("secondary", "Gi/o primary, Gq secondary"),
    # NPY
    "npy1r": ("none", "Gi/o primary"),
    "npy2r": ("none", "Gi/o primary"),
    "npy4r": ("none", "Gi/o primary"),
    "npy5r": ("none", "Gi/o primary"),
    # Neurotensin
    "nts1r": ("primary", "Gq/11 primary"),
    "nts2r": ("primary", "Gq/11 primary"),
    # Opioid
    "oprm":  ("none", "Gi/o primary"),
    "oprd":  ("none", "Gi/o primary"),
    "oprk":  ("none", "Gi/o primary"),
    "oprx":  ("none", "Gi/o primary"),
    # Orexin
    "ox1r":  ("primary", "Gq/11 primary"),
    "ox2r":  ("primary", "Gq/11 primary"),
    # QRFP
    "qrfpr": ("secondary", "Gi/o primary, Gq secondary"),
    # PrRP
    "prlhr": ("primary", "Gq/11 primary"),
    # PAR
    "par1":  ("primary", "Gq/11 primary"),
    "par2":  ("primary", "Gq/11 primary"),
    "par3":  ("secondary", "Gq secondary"),
    "par4":  ("primary", "Gq/11 primary"),
    # Relaxin
    "rxfp1": ("none", "Gs primary"),
    "rxfp2": ("none", "Gs primary"),
    "rxfp3": ("none", "Gi/o primary"),
    "rxfp4": ("none", "Gi/o primary"),
    # Somatostatin
    "sst1r": ("none", "Gi/o primary"),
    "sst2r": ("none", "Gi/o primary"),
    "sst3r": ("none", "Gi/o primary"),
    "sst4r": ("none", "Gi/o primary"),
    "sst5r": ("none", "Gi/o primary"),
    # Tachykinin
    "nk1r":  ("primary", "Gq/11 primary"),
    "nk2r":  ("primary", "Gq/11 primary"),
    "nk3r":  ("primary", "Gq/11 primary"),
    # TRH
    "trhr":  ("primary", "Gq/11 primary"),
    # Urotensin
    "uts2r": ("primary", "Gq/11 primary"),
    # Vasopressin/Oxytocin
    "v1ar":  ("primary", "Gq/11 primary"),
    "v1br":  ("primary", "Gq/11 primary"),
    "v2r":   ("none", "Gs primary"),
    "oxtr":  ("primary", "Gq/11 primary"),

    # === Protein receptors ===
    # Chemerin
    "cmklr": ("none", "Gi/o primary"),
    # Chemokine receptors (mostly Gi/o)
    "ccr1":  ("none", "Gi/o primary"),
    "ccr2":  ("none", "Gi/o primary"),
    "ccr3":  ("none", "Gi/o primary"),
    "ccr4":  ("none", "Gi/o primary"),
    "ccr5":  ("none", "Gi/o primary"),
    "ccr6":  ("none", "Gi/o primary"),
    "ccr7":  ("none", "Gi/o primary"),
    "ccr8":  ("none", "Gi/o primary"),
    "ccr9":  ("none", "Gi/o primary"),
    "ccr10": ("none", "Gi/o primary"),
    "cxcr1": ("none", "Gi/o primary"),
    "cxcr2": ("none", "Gi/o primary"),
    "cxcr3": ("none", "Gi/o primary"),
    "cxcr4": ("none", "Gi/o primary"),
    "cxcr5": ("none", "Gi/o primary"),
    "cxcr6": ("none", "Gi/o primary"),
    "cx3c1": ("none", "Gi/o primary"),
    "xcr1":  ("none", "Gi/o primary"),
    # Glycoprotein hormone
    "fshr":  ("none", "Gs primary"),
    "lshr":  ("none", "Gs primary"),
    "tshr":  ("secondary", "Gs primary, Gq secondary"),
    # Prokineticin
    "prokr1": ("primary", "Gq/11 primary"),
    "prokr2": ("primary", "Gq/11 primary"),

    # === Lipid receptors ===
    # Free fatty acid
    "ffar1": ("primary", "Gq/11 primary"),
    "ffar2": ("secondary", "Gi/o primary, Gq secondary"),
    "ffar3": ("none", "Gi/o primary"),
    "ffar4": ("primary", "Gq/11 primary"),
    # Leukotriene
    "blt1":  ("secondary", "Gi/o primary, Gq secondary"),
    "blt2":  ("secondary", "Gi/o primary, Gq secondary"),
    "cltr1": ("primary", "Gq/11 primary"),
    "cltr2": ("primary", "Gq/11 primary"),
    "oxer":  ("none", "Gi/o primary"),
    # LPA
    "lpar1": ("secondary", "Gi/o primary, Gq secondary"),
    "lpar2": ("primary", "Gq/11 primary"),
    "lpar3": ("primary", "Gq/11 primary"),
    "lpar4": ("none", "Gs primary"),
    "lpar5": ("secondary", "G12/13 primary, Gq secondary"),
    "lpar6": ("none", "G12/13 primary"),
    # S1P
    "s1pr1": ("none", "Gi/o primary"),
    "s1pr2": ("secondary", "G12/13 primary, Gq secondary"),
    "s1pr3": ("secondary", "Gi/o primary, Gq secondary"),
    "s1pr4": ("none", "Gi/o primary"),
    "s1pr5": ("none", "Gi/o primary"),
    # Cannabinoid
    "cnr1":  ("none", "Gi/o primary"),
    "cnr2":  ("none", "Gi/o primary"),
    # GPR18/55/119
    "gpr18": ("none", "Gi/o primary"),
    "gpr55": ("secondary", "G12/13 primary, Gq secondary"),
    "gp119": ("none", "Gs primary"),
    # PAF
    "pafr":  ("primary", "Gq/11 primary"),
    # Prostanoid
    "ptger1": ("primary", "Gq/11 primary"),
    "ptger2": ("none", "Gs primary"),
    "ptger3": ("secondary", "Gi/o primary, Gq secondary"),
    "ptger4": ("none", "Gs primary"),
    "ptgfr":  ("primary", "Gq/11 primary"),
    "ptgir":  ("none", "Gs primary"),
    "ta2r":   ("primary", "Gq/11 primary"),
    "ptgdr":  ("none", "Gs primary"),
    "ptgdr2": ("secondary", "Gi/o primary, Gq secondary"),

    # === Melatonin ===
    "mtr1a": ("none", "Gi/o primary"),
    "mtr1b": ("none", "Gi/o primary"),

    # === Nucleotide receptors ===
    # Adenosine
    "aa1r":  ("none", "Gi/o primary"),
    "aa2ar": ("none", "Gs primary"),
    "aa2br": ("none", "Gs primary"),
    "aa3r":  ("none", "Gi/o primary"),
    # P2Y
    "p2ry1": ("primary", "Gq/11 primary"),
    "p2ry2": ("primary", "Gq/11 primary"),
    "p2ry4": ("primary", "Gq/11 primary"),
    "p2ry6": ("primary", "Gq/11 primary"),
    "p2y11": ("secondary", "Gs primary, Gq secondary"),
    "p2y12": ("none", "Gi/o primary"),
    "p2y13": ("none", "Gi/o primary"),
    "p2y14": ("none", "Gi/o primary"),

    # === Steroid receptors ===
    "gpbar": ("none", "Gs primary"),
    "gper1": ("none", "Gs primary"),

    # === Alicarboxylic acid ===
    "hcar1": ("none", "Gi/o primary"),
    "hcar2": ("none", "Gi/o primary"),
    "hcar3": ("none", "Gi/o primary"),
    "oxgr1": ("secondary", "Gi/o primary, Gq secondary"),
    "sucnr": ("none", "Gi/o primary"),

    # === Sensory receptors (Opsins) ===
    "opsd":  ("none", "Gt primary"),
    "opn3":  ("none", "Gi/o primary"),
    "opn4":  ("primary", "Gq/11 primary"),
    "opn5":  ("none", "Gi/o primary"),
    "opsb":  ("none", "Gt primary"),
    "opsg":  ("none", "Gt primary"),
    "opsr":  ("none", "Gt primary"),
    "opsw":  ("none", "Gt primary"),
    "rrh":   ("none", "Gi/o primary"),

    # === Class B1 (Secretin) ===
    "calcr": ("secondary", "Gs primary, Gq secondary"),
    "calcrl": ("none", "Gs primary"),
    "crfr1": ("none", "Gs primary"),
    "crfr2": ("none", "Gs primary"),
    "ghrhr": ("none", "Gs primary"),
    "gipr":  ("none", "Gs primary"),
    "glp1r": ("none", "Gs primary"),
    "glp2r": ("none", "Gs primary"),
    "glcr":  ("none", "Gs primary"),
    "sctr":  ("none", "Gs primary"),
    "pthr1": ("secondary", "Gs primary, Gq secondary"),
    "pthr2": ("none", "Gs primary"),
    "pacr":  ("secondary", "Gs primary, Gq secondary"),
    "vipr1": ("none", "Gs primary"),
    "vipr2": ("none", "Gs primary"),

    # === Class C (Glutamate) ===
    "casr":  ("primary", "Gq/11 primary"),
    "gabr1": ("none", "Gi/o primary"),
    "gabr2": ("none", "Gi/o primary"),
    "grm1":  ("primary", "Gq/11 primary"),
    "grm2":  ("none", "Gi/o primary"),
    "grm3":  ("none", "Gi/o primary"),
    "grm4":  ("none", "Gi/o primary"),
    "grm5":  ("primary", "Gq/11 primary"),
    "grm6":  ("none", "Gi/o primary"),
    "grm7":  ("none", "Gi/o primary"),
    "grm8":  ("none", "Gi/o primary"),
    "tas1r1": ("primary", "Gq/11 primary"),
    "tas1r2": ("primary", "Gq/11 primary"),
    "tas1r3": ("primary", "Gq/11 primary"),

    # === GPCRdb命名别名 (补充未匹配的受体) ===
    # Atypical chemokine receptors (no G protein signaling)
    "ackr1": ("none", "No G protein coupling"),
    "ackr2": ("none", "No G protein coupling"),
    "ackr3": ("none", "Gi/o primary"),  # CXCR7, couples Gi
    "ackr4": ("none", "No G protein coupling"),
    "ccrl2": ("none", "No G protein coupling"),
    # Melanocortin aliases
    "acthr": ("none", "Gs primary"),      # ACTH receptor = MC2R
    "mshr":  ("none", "Gs primary"),      # MSH receptor = MC1R
    # Apelin alias
    "apj":   ("none", "Gi/o primary"),    # Apelin receptor
    # Calcitonin receptor-like alias
    "calrl": ("none", "Gs primary"),      # CALCRL
    # Chemerin-like
    "cml1":  ("none", "Gi/o primary"),    # Chemerin receptor 1
    "cml2":  ("none", "Unknown"),         # Chemerin receptor 2
    # Gastrin receptor alias
    "gasr":  ("primary", "Gq/11 primary"),  # Gastrin = CCK2R
    # Glucagon receptor alias
    "glr":   ("none", "Gs primary"),      # Glucagon receptor
    # GPR42 (pseudogene/FFA related)
    "gpr42": ("none", "Gi/o primary"),    # FFA3-like
    # Leukotriene B4 receptors aliases
    "lt4r1": ("secondary", "Gi/o primary, Gq secondary"),  # BLT1
    "lt4r2": ("secondary", "Gi/o primary, Gq secondary"),  # BLT2
    # NPY6 receptor
    "npy6r": ("none", "Unknown"),         # Pseudogene in human
    # Neurotensin aliases
    "ntr1":  ("primary", "Gq/11 primary"),   # NTS1R
    "ntr2":  ("primary", "Gq/11 primary"),   # NTS2R
    # Opsin X
    "opsx":  ("none", "Unknown"),         # Opsin X (OPN3-like)
    # Glycoprotein hormone aliases
    "lhr":   ("none", "Gs primary"),      # LH receptor
    "tshr":  ("secondary", "Gs primary, Gq secondary"),  # TSH receptor

    # Additional Class A aliases commonly used by GPCRdb
    "5ht2":  ("primary", "Gq/11 primary"),  # Generic 5-HT2
    "agtr":  ("primary", "Gq/11 primary"),  # Generic AT1
    "nk1":   ("primary", "Gq/11 primary"),
    "nk2":   ("primary", "Gq/11 primary"),
    "nk3":   ("primary", "Gq/11 primary"),
    "ednr":  ("primary", "Gq/11 primary"),  # Generic endothelin
    "bkrb":  ("primary", "Gq/11 primary"),  # Generic bradykinin
    "ox1":   ("primary", "Gq/11 primary"),
    "ox2":   ("primary", "Gq/11 primary"),

    # Additional Class B1 aliases
    "crfr":  ("none", "Gs primary"),      # Generic CRF
    "pthr":  ("secondary", "Gs primary, Gq secondary"),

    # Additional missing receptors
    "gp183": ("none", "Gi/o primary"),    # GPR183/EBI2
    "gp119": ("none", "Gs primary"),      # GPR119
    "gp120": ("none", "Gs primary"),      # GPR120 = FFAR4
    "gpr34": ("none", "Gi/o primary"),
    "gpr35": ("secondary", "Gi/o primary, Gq secondary"),
    "gpr37": ("none", "Unknown"),
    "gpr39": ("secondary", "Gq secondary"),
    "gpr56": ("none", "G12/13 primary"),
    "gpr68": ("secondary", "Gs primary, Gq secondary"),  # OGR1
    "gpr84": ("none", "Gi/o primary"),
    "gp101": ("none", "Unknown"),
    "gp132": ("none", "Unknown"),
    "gp141": ("none", "Unknown"),
    "gp146": ("none", "Unknown"),
    "gp148": ("none", "Unknown"),
    "gp149": ("none", "Unknown"),
    "gp150": ("none", "Unknown"),
    "gp151": ("none", "Unknown"),
    "gp152": ("none", "Unknown"),
    "gp153": ("none", "Unknown"),
    "gp160": ("none", "Unknown"),
    "gp161": ("none", "Unknown"),
    "gp162": ("none", "Unknown"),
    "gp171": ("none", "Unknown"),
    "gp173": ("none", "Unknown"),
    "gp174": ("none", "Unknown"),
    "gp176": ("none", "Unknown"),
    "gp182": ("none", "Unknown"),

    # Mas-related GPCRs
    "mrgrd": ("none", "Gi/o primary"),
    "mrgre": ("none", "Gq secondary"),
    "mrgrf": ("none", "Gq secondary"),
    "mrgrx1": ("primary", "Gq/11 primary"),
    "mrgrx2": ("primary", "Gq/11 primary"),
    "mrgrx3": ("none", "Unknown"),
    "mrgrx4": ("none", "Unknown"),
    "mas":    ("none", "Gi/o primary"),
    "masl1":  ("none", "Unknown"),

    # Bile acid receptor
    "gpbar1": ("none", "Gs primary"),
    "gper":   ("none", "Gs primary"),       # GPER1

    # Taste receptors (Class C)
    "tas1r":  ("primary", "Gq/11 primary"),

    # === Final round: remaining GPCRdb aliases ===
    # Oxytocin alias
    "oxyr":  ("primary", "Gq/11 primary"),     # Oxytocin receptor
    # Prostanoid aliases (GPCRdb naming)
    "pd2r":  ("none", "Gs primary"),           # DP1 = PTGDR
    "pd2r2": ("secondary", "Gi/o primary, Gq secondary"),  # DP2 = PTGDR2
    "pe2r1": ("primary", "Gq/11 primary"),     # EP1 = PTGER1
    "pe2r2": ("none", "Gs primary"),           # EP2 = PTGER2
    "pe2r3": ("secondary", "Gi/o primary, Gq secondary"),  # EP3 = PTGER3
    "pe2r4": ("none", "Gs primary"),           # EP4 = PTGER4
    "pf2r":  ("primary", "Gq/11 primary"),     # FP = PTGFR
    "pi2r":  ("none", "Gs primary"),           # IP = PTGIR
    "ptafr": ("primary", "Gq/11 primary"),     # PAF receptor
    # PTH receptor aliases
    "pth1r": ("secondary", "Gs primary, Gq secondary"),  # PTH1R
    "pth2r": ("none", "Gs primary"),           # PTH2R
    # Prokineticin aliases
    "pkr1":  ("primary", "Gq/11 primary"),     # PROKR1
    "pkr2":  ("primary", "Gq/11 primary"),     # PROKR2
    # Relaxin aliases
    "rl3r1": ("none", "Gi/o primary"),         # RXFP3
    "rl3r2": ("none", "Gi/o primary"),         # RXFP4
    # Somatostatin aliases
    "ssr1":  ("none", "Gi/o primary"),         # SSTR1
    "ssr2":  ("none", "Gi/o primary"),         # SSTR2
    "ssr3":  ("none", "Gi/o primary"),         # SSTR3
    "ssr4":  ("none", "Gi/o primary"),         # SSTR4
    "ssr5":  ("none", "Gi/o primary"),         # SSTR5
    # Thromboxane alias
    "ta2r":  ("primary", "Gq/11 primary"),     # TP = TA2R
    # VIP/PACAP aliases
    "vipr":  ("none", "Gs primary"),           # Generic VIP receptor
    "pacr1": ("secondary", "Gs primary, Gq secondary"),  # PAC1

    # === Last 4 unmatched ===
    "sucr1": ("none", "Gi/o primary"),         # Succinate receptor 1 = SUCNR1
    "taar1": ("none", "Gs primary"),           # Trace amine receptor 1
    "trfr":  ("primary", "Gq/11 primary"),     # TRH receptor = TRHR
    "ur2r":  ("primary", "Gq/11 primary"),     # Urotensin-II receptor = UTS2R
}


def fetch_json(url, retries=3, delay=1.0):
    """带重试的API请求"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                print(f"  HTTP {resp.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"  请求失败 (尝试 {attempt+1}/{retries}): {e}")
        time.sleep(delay)
    return None


def get_subfamily_slugs(parent_slug):
    """递归获取所有叶子节点家族slug"""
    url = f"{BASE_URL}/proteinfamily/children/{parent_slug}/"
    children = fetch_json(url)
    if not children:
        return [parent_slug]

    leaf_slugs = []
    for child in children:
        slug = child["slug"]
        # 检查是否还有子节点
        sub_children = fetch_json(f"{BASE_URL}/proteinfamily/children/{slug}/")
        if sub_children:
            # 还有子节点, 继续递归
            for sc in sub_children:
                leaf_slugs.append(sc["slug"])
        else:
            leaf_slugs.append(slug)
    return leaf_slugs


def match_gq_coupling(entry_name):
    """根据entry_name匹配Gq偶联状态"""
    # entry_name格式: 受体名_物种, 如 "5ht2a_human"
    prefix = entry_name.split("_")[0]

    # 直接匹配
    if prefix in GQ_COUPLING_MAP:
        return GQ_COUPLING_MAP[prefix]

    # 尝试更长的前缀匹配 (如 "cckar" vs "ccka")
    for key, value in GQ_COUPLING_MAP.items():
        if prefix.startswith(key) or key.startswith(prefix):
            return value

    return None


def main():
    print("=" * 70)
    print("GPCRdb数据获取脚本 - GPCR-Gq偶联预测训练集构建")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_proteins = []
    unmatched = []

    for family_slug in TARGET_FAMILIES:
        print(f"\n[获取家族] {family_slug}")

        # 获取叶子节点
        leaf_slugs = get_subfamily_slugs(family_slug)
        print(f"  叶子节点: {len(leaf_slugs)} 个受体类型")

        for slug in leaf_slugs:
            url = f"{BASE_URL}/proteinfamily/proteins/{slug}/"
            proteins = fetch_json(url)
            if not proteins:
                continue

            # 只取人类受体
            human_proteins = [p for p in proteins if p.get("species") == TARGET_SPECIES]

            for prot in human_proteins:
                entry_name = prot["entry_name"]
                coupling = match_gq_coupling(entry_name)

                if coupling is None:
                    unmatched.append(entry_name)
                    continue

                gq_status, coupling_desc = coupling
                gq_label = 1 if gq_status in ("primary", "secondary") else 0

                all_proteins.append({
                    "entry_name": entry_name,
                    "name": prot.get("name", "").replace("<sub>", "").replace("</sub>", ""),
                    "accession": prot.get("accession", ""),
                    "family": prot.get("family", ""),
                    "species": prot.get("species", ""),
                    "sequence": prot.get("sequence", ""),
                    "seq_length": len(prot.get("sequence", "")),
                    "gq_coupling": gq_status,
                    "gq_label": gq_label,
                    "coupling_description": coupling_desc,
                })

            time.sleep(0.2)  # 限速, 对API友好

    # 统计
    print(f"\n{'=' * 70}")
    print("数据获取完成")
    print(f"{'=' * 70}")
    print(f"总蛋白数: {len(all_proteins)}")
    gq_count = sum(1 for p in all_proteins if p["gq_label"] == 1)
    non_gq_count = len(all_proteins) - gq_count
    print(f"  Gq偶联 (primary+secondary): {gq_count}")
    print(f"  非Gq偶联: {non_gq_count}")

    if unmatched:
        print(f"\n未匹配偶联数据的受体 ({len(unmatched)}):")
        for name in sorted(set(unmatched))[:20]:
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... 等{len(unmatched)-20}个")

    # 按偶联类型细分
    primary_count = sum(1 for p in all_proteins if p["gq_coupling"] == "primary")
    secondary_count = sum(1 for p in all_proteins if p["gq_coupling"] == "secondary")
    print(f"\nGq偶联细分:")
    print(f"  primary (主要): {primary_count}")
    print(f"  secondary (次要): {secondary_count}")
    print(f"  none (不偶联): {non_gq_count}")

    # 保存CSV
    if all_proteins:
        fieldnames = ["entry_name", "name", "accession", "family", "species",
                       "sequence", "seq_length", "gq_coupling", "gq_label",
                       "coupling_description"]
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_proteins)
        print(f"\n✓ 数据已保存: {OUTPUT_FILE}")
        print(f"  文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # 同时保存JSON格式 (方便后续使用)
    json_file = OUTPUT_FILE.replace(".csv", ".json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_proteins, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON格式: {json_file}")

    return all_proteins


if __name__ == "__main__":
    data = main()
