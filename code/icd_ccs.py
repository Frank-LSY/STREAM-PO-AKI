import os
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# HCUP single-level CCS crosswalks (ICD-9-CM and ICD-10-CM), downloaded from the AHRQ HCUP website.
CCS_DIR = Path(os.environ.get("CCS_MAP_DIR", Path(__file__).resolve().parents[1] / "data" / "reference"))
ICD_9CM_REF = pd.read_csv(CCS_DIR / "ccs_icd9cm.csv")
ICD_10CM_REF = pd.read_csv(CCS_DIR / "ccs_icd10cm.csv")

def icd_to_ccs(icd_code,icd_version):
    ccs_readable_icd_code = icd_code.replace('.', '')
    if icd_version == 'ICD9CM':
        if ccs_readable_icd_code in ICD_9CM_REF['ICD.9.CM.CODE'].values:
            return ICD_9CM_REF[ICD_9CM_REF['ICD.9.CM.CODE'] == ccs_readable_icd_code]['CCS.CATEGORY'].values[0]
        else:
            return None
    elif icd_version == 'ICD10CM':
        ccs_readable_icd_code = "'"+ccs_readable_icd_code+"'"
        if ccs_readable_icd_code in ICD_10CM_REF['\'ICD-10-CM CODE\''].values:
            return ICD_10CM_REF[ICD_10CM_REF['\'ICD-10-CM CODE\''] == ccs_readable_icd_code]['\'CCS CATEGORY\''].values[0].strip("'")
        else:
            return None
    else:
        return None