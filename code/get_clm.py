import os
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import urllib
from dateutil.relativedelta import relativedelta
import requests
import json
from icd_ccs import icd_to_ccs

# Connection settings for an OMOP CDM v5.4 database on SQL Server; supply them via environment variables.
server = os.environ["OMOP_SERVER"]
database = os.environ["OMOP_DATABASE"]
username = os.environ["OMOP_USER"]
password = os.environ["OMOP_PASSWORD"]
# Fully qualified schema prefix used in the queries, e.g. "[MyOmopDb].[v5_4]".
OMOP_SCHEMA = os.environ.get("OMOP_SCHEMA", f"[{database}].[dbo]")

odbc_string = (
    f"DRIVER=ODBC Driver 18 for SQL Server;"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "TrustServerCertificate=yes;"
    "Encrypt=yes"
)

conn_string = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_string)}"
engine = create_engine(conn_string)


poi_dict = {
    # 'Delirium':['ICD10CM|F05%','ICD9CM|290.4%','ICD9CM|293.0%','ICD9CM|293.1%'],
    # 'Stroke':['ICD10CM|G46%','ICD9CM|435%','ICD9CM|437.8%'],
    # 'AMI':['ICD10CM|I21%','ICD10CM|I22%','ICD9CM|410%'],
    # 'ARDS':['ICD10CM|J80%','ICD9CM|518.82%'],
    # 'Pulmonary Embolism':['ICD10CM|I26%','ICD9CM|415%'],
    # 'Liver Injury':['ICD10CM|K71.2%','ICD10CM|K72.0%','ICD9CM|573.3%','ICD9CM|572.2%','ICD9CM|570%'],
    'AKI':['ICD10CM|N17.0','ICD10CM|N17.1','ICD10CM|N17.2','ICD10CM|N17.8','ICD10CM|N17.9',
    'ICD9CM|584.5','ICD9CM|584.6','ICD9CM|584.7','ICD9CM|584.8','ICD9CM|584.9'],
    # 'AKI':['ICD10CM|N17%','ICD9CM|584%']
}

def get_ingredient_rxcui(ndc):
    url_ndc = f"https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={ndc}"
    response_ndc = requests.get(url_ndc)
    if response_ndc.status_code == 200:
        data_ndc = response_ndc.json()
        rxcui = data_ndc.get('ndcStatus', {}).get('rxcui', [])
        return get_rxcuis(rxcui)
    else:
        return None

def get_rxcuis(rxcui):
    if rxcui:
        url_ingredient = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
        response_ingredient = requests.get(url_ingredient)
        # print(response)
        if response_ingredient.status_code == 200:
            data_ingredient = response_ingredient.json()
            # print(rxcui, data_ingredient)
            rxcuis = [item['rxcui'] for item in data_ingredient.get('relatedGroup', {}).get('conceptGroup', [])[0].get('conceptProperties', [])]
            if rxcuis:
                return rxcuis
            else:
                return [rxcui]
    else:
        return None

def get_condition_occurrence_window(row, engine):
    """
    性能优化版 get_condition_occurrence_window：
    1. 避免多次SQL round-trip，合并SQL语句为尽量少的批量查询。
    2. 用LIST扩展拼接SQL，减少字符串拼接与切割。
    3. pandas直接转dict。
    4. 使用loc等减少遍历。
    5. 尝试减少调用外部API（get_ingredient_rxcui）次数。
    """

    procedure_datetime = pd.to_datetime(row.procedure_datetime)
    procedure_source_value = row.procedure_source_value
    condition_list = row.condition_list
    person_id = row.person_id
    age = row.age
    gender_concept_id = row.gender_concept_id
    race_concept_id = row.race_concept_id
    start_window = procedure_datetime - relativedelta(days=7)
    end_window = procedure_datetime + relativedelta(days=30)

    # # 性能优化1：避免字符串拼接，直接构建列表后JOIN，并且用IN过滤condition_code
    # all_condition_codes = []
    # for condition in condition_list:
    #     codes = poi_dict[condition.strip(" ' ")]
    #     all_condition_codes.extend(codes)
    # # 批量SQL过滤，避免多次OR
    # codes_in_sql = ','.join([f"'{c}'" for c in all_condition_codes]) or "''"
    all_condition_codes = poi_dict['AKI']
    # 优化2：一次查出表中所有需要用到的condition_occurrence，避免两次read_sql_query
    cond_sql = f"""
        SELECT condition_source_value, condition_start_datetime
        FROM {OMOP_SCHEMA}.[condition_occurrence]
        WHERE person_id = {person_id}
        AND condition_source_value IS NOT NULL
        AND condition_start_datetime >= '{start_window}'
        AND condition_start_datetime <= '{end_window}'
        ORDER BY condition_start_datetime ASC
    """
    cond_df = pd.read_sql_query(cond_sql, engine)

    # 提取cut_time: 首次出现我们关注的codes的时间点（>= procedure_datetime）
    cond_mask = (
        cond_df["condition_source_value"].isin(all_condition_codes)
        & (cond_df["condition_start_datetime"] >= procedure_datetime)
    )
    # 直接通过loc高效查找
    if cond_mask.any():
        cut_time = cond_df.loc[cond_mask, "condition_start_datetime"].min()
    else:
        cut_time = end_window
    cut_time = pd.to_datetime(cut_time)

    # 提取不属于关注codes且 <= cut_time
    condition_mask_not = ~cond_df["condition_source_value"].isin(all_condition_codes) & \
                         (cond_df["condition_start_datetime"] <= cut_time) & \
                         (cond_df["condition_start_datetime"] >= start_window)
    if condition_mask_not.any():
        conditions = cond_df.loc[condition_mask_not, ["condition_source_value", "condition_start_datetime"]]
        conditions = [
            {
                "condition_source_value": row["condition_source_value"],
                "ccs_category": icd_to_ccs(row["condition_source_value"].split('|')[1], row["condition_source_value"].split('|')[0]),
                "time": str(row["condition_start_datetime"])
            }
            for _, row in conditions.iterrows()
        ]
    else:
        conditions = []

    # 优化3：实验室检查查询和处理
    lab_query = f"""
        SELECT c.concept_name,m.measurement_concept_id, m.measurement_datetime, m.value_as_number, m.unit_source_value,m.unit_concept_id
        FROM {OMOP_SCHEMA}.[measurement] m
        INNER JOIN {OMOP_SCHEMA}.[concept] c
        ON m.measurement_concept_id = c.concept_id
        WHERE person_id = {person_id}
          AND m.value_as_number IS NOT NULL
          AND c.concept_name != 'No matching concept'
          AND m.measurement_datetime >= '{start_window}'
          AND m.measurement_datetime <= '{cut_time}'
    """
    related_labs = pd.read_sql_query(lab_query, engine)
    if not related_labs.empty:
        # Ensure measurement_datetime is string type
        related_labs["measurement_datetime"] = related_labs["measurement_datetime"].astype(str)
        labs = related_labs.rename(
            columns={
                "concept_name": "lab_name",
                "measurement_concept_id": "lab_code",
                "measurement_datetime": "time",
                "value_as_number": "value",
                "unit_source_value": "unit",
                "unit_concept_id": "unit_code"
            }
        ).to_dict(orient='records')
    else:
        labs = []

    # 药物暴露：确保查询的时间窗口为 start_window ~ cut_time
    med_query = f"""
        SELECT drug_source_value, drug_exposure_start_datetime, drug_exposure_end_datetime
        FROM {OMOP_SCHEMA}.[drug_exposure]
        WHERE person_id = {person_id}
        AND drug_exposure_end_datetime >= '{start_window}'
        AND drug_exposure_start_datetime <= '{cut_time}'
    """
    related_meds = pd.read_sql_query(med_query, engine)
    meds = []
    # 性能优化4：批量NDC转rxcui可进一步向量化和缓存，不过这里保持原有逻辑
    for _, med_row in related_meds.iterrows():
        code_type, code_value = med_row['drug_source_value'].split('|', 1)
        delta_days = (med_row['drug_exposure_end_datetime'] - med_row['drug_exposure_start_datetime']).days
        # When drug list is long, calling get_ingredient_rxcui/get_rxcuis inside the loop can be slow.
        # So we cache results for each unique code_value (NDC/RXNORM) using an in-memory dict.
        if not hasattr(get_condition_occurrence_window, "_rxcui_cache"):
            get_condition_occurrence_window._rxcui_cache = {}
        rxcui_cache = get_condition_occurrence_window._rxcui_cache

        if code_type == 'NDC':
            if code_value in rxcui_cache:
                rxcuis = rxcui_cache[code_value]
            else:
                rxcuis = get_ingredient_rxcui(code_value)
                rxcui_cache[code_value] = rxcuis
            for rxcui in (rxcuis or []):
                meds.append({
                    "drug_code": med_row['drug_source_value'],
                    "start_time": str(med_row['drug_exposure_start_datetime']),
                    "end_time": str(med_row['drug_exposure_end_datetime']),
                    "rxcui": rxcui,
                    "delta_days": delta_days
                })
        elif code_type == 'RXNORM':
            if code_value in rxcui_cache:
                rxcui_list = rxcui_cache[code_value]
            else:
                rxcui_list = get_rxcuis(code_value)
                rxcui_cache[code_value] = rxcui_list
            for rxcui in (rxcui_list or []):
                meds.append({
                    "drug_code": med_row['drug_source_value'],
                    "start_time": str(med_row['drug_exposure_start_datetime']),
                    "end_time": str(med_row['drug_exposure_end_datetime']),
                    "rxcui": rxcui,
                    "delta_days": delta_days
                })
        else:
            meds.append({
                "drug_code": med_row['drug_source_value'],
                "start_time": str(med_row['drug_exposure_start_datetime']),
                "end_time": str(med_row['drug_exposure_end_datetime']),
                "rxcui": None,
                "delta_days": delta_days
            })
    result_json = {
        "person_id": str(person_id),
        "age": age,
        "gender_concept_id": gender_concept_id,
        "race_concept_id": race_concept_id,
        "procedure_source_value": procedure_source_value,
        "procedure_datetime": str(procedure_datetime),
        "outcome": condition_list,
        "condition": conditions,
        "lab": labs,
        "med": meds
    }

    return result_json

def get_condition_occurrence_window_no_cut_time(row, engine):
    """
    性能优化版 get_condition_occurrence_window（不做cut time）：
    1. 避免多次SQL round-trip，合并SQL语句为尽量少的批量查询。
    2. 用LIST扩展拼接SQL，减少字符串拼接与切割。
    3. pandas直接转dict。
    4. 使用loc等减少遍历。
    5. 尝试减少调用外部API（get_ingredient_rxcui）次数。
    6. 不做cut time，直接用start_window到end_window窗口。
    """

    procedure_datetime = pd.to_datetime(row.procedure_datetime)
    procedure_source_value = row.procedure_source_value
    person_id = row.person_id
    age = row.age
    gender_concept_id = row.gender_concept_id
    race_concept_id = row.race_concept_id
    start_window = procedure_datetime - relativedelta(days=7)
    end_window = procedure_datetime + relativedelta(days=30)

    all_condition_codes = poi_dict['AKI']  # 如果业务逻辑需要更换，这里保留原注释
    # 优化2：一次查出表中所有需要用到的condition_occurrence
    cond_sql = f"""
        SELECT condition_source_value, condition_start_datetime
        FROM {OMOP_SCHEMA}.[condition_occurrence]
        WHERE person_id = {person_id}
        AND condition_source_value IS NOT NULL
        AND condition_start_datetime >= '{start_window}'
        AND condition_start_datetime <= '{end_window}'
        ORDER BY condition_start_datetime ASC
    """
    cond_df = pd.read_sql_query(cond_sql, engine)

    # 不做cut_time：全部区间数据
    # 只要不是all_condition_codes的，且在窗口内的都保留
    condition_mask_not = ~cond_df["condition_source_value"].isin(all_condition_codes) & \
                         (cond_df["condition_start_datetime"] >= start_window) & \
                         (cond_df["condition_start_datetime"] <= end_window)
    if condition_mask_not.any():
        conditions = cond_df.loc[condition_mask_not, ["condition_source_value", "condition_start_datetime"]]
        conditions = [
            {
                "condition_source_value": row["condition_source_value"],
                "ccs_category": icd_to_ccs(row["condition_source_value"].split('|')[1], row["condition_source_value"].split('|')[0]),
                "time": str(row["condition_start_datetime"])
            }
            for _, row in conditions.iterrows()
        ]
    else:
        conditions = []

    # 实验室检查，不用cut_time
    lab_query = f"""
        SELECT c.concept_name,m.measurement_concept_id, m.measurement_datetime, m.value_as_number, m.unit_source_value,m.unit_concept_id
        FROM {OMOP_SCHEMA}.[measurement] m
        INNER JOIN {OMOP_SCHEMA}.[concept] c
        ON m.measurement_concept_id = c.concept_id
        WHERE person_id = {person_id}
          AND m.value_as_number IS NOT NULL
          AND c.concept_name != 'No matching concept'
          AND m.measurement_datetime >= '{start_window}'
          AND m.measurement_datetime <= '{end_window}'
    """
    related_labs = pd.read_sql_query(lab_query, engine)
    if not related_labs.empty:
        # Ensure measurement_datetime is string type
        related_labs["measurement_datetime"] = related_labs["measurement_datetime"].astype(str)
        labs = related_labs.rename(
            columns={
                "concept_name": "lab_name",
                "measurement_concept_id": "lab_code",
                "measurement_datetime": "time",
                "value_as_number": "value",
                "unit_source_value": "unit",
                "unit_concept_id": "unit_code"
            }
        ).to_dict(orient='records')
    else:
        labs = []

    # 药物暴露：确保查询的时间窗口为 start_window ~ end_window
    med_query = f"""
        SELECT drug_source_value, drug_exposure_start_datetime, drug_exposure_end_datetime
        FROM {OMOP_SCHEMA}.[drug_exposure]
        WHERE person_id = {person_id}
        AND drug_exposure_end_datetime >= '{start_window}'
        AND drug_exposure_start_datetime <= '{end_window}'
    """
    related_meds = pd.read_sql_query(med_query, engine)
    meds = []
    # 性能优化4：批量NDC转rxcui可进一步向量化和缓存，不过这里保持原有逻辑
    for _, med_row in related_meds.iterrows():
        code_type, code_value = med_row['drug_source_value'].split('|', 1)
        delta_days = (med_row['drug_exposure_end_datetime'] - med_row['drug_exposure_start_datetime']).days
        if not hasattr(get_condition_occurrence_window_no_cut_time, "_rxcui_cache"):
            get_condition_occurrence_window_no_cut_time._rxcui_cache = {}
        rxcui_cache = get_condition_occurrence_window_no_cut_time._rxcui_cache

        if code_type == 'NDC':
            if code_value in rxcui_cache:
                rxcuis = rxcui_cache[code_value]
            else:
                rxcuis = get_ingredient_rxcui(code_value)
                rxcui_cache[code_value] = rxcuis
            for rxcui in (rxcuis or []):
                meds.append({
                    "drug_code": med_row['drug_source_value'],
                    "start_time": str(med_row['drug_exposure_start_datetime']),
                    "end_time": str(med_row['drug_exposure_end_datetime']),
                    "rxcui": rxcui,
                    "delta_days": delta_days
                })
        elif code_type == 'RXNORM':
            if code_value in rxcui_cache:
                rxcui_list = rxcui_cache[code_value]
            else:
                rxcui_list = get_rxcuis(code_value)
                rxcui_cache[code_value] = rxcui_list
            for rxcui in (rxcui_list or []):
                meds.append({
                    "drug_code": med_row['drug_source_value'],
                    "start_time": str(med_row['drug_exposure_start_datetime']),
                    "end_time": str(med_row['drug_exposure_end_datetime']),
                    "rxcui": rxcui,
                    "delta_days": delta_days
                })
        else:
            meds.append({
                "drug_code": med_row['drug_source_value'],
                "start_time": str(med_row['drug_exposure_start_datetime']),
                "end_time": str(med_row['drug_exposure_end_datetime']),
                "rxcui": None,
                "delta_days": delta_days
            })
    result_json = {
        "person_id": str(person_id),
        "age": age,
        "gender_concept_id": gender_concept_id,
        "race_concept_id": race_concept_id,
        "procedure_source_value": procedure_source_value,
        "procedure_datetime": str(procedure_datetime),
        "outcome": [],
        "condition": conditions,
        "lab": labs,
        "med": meds
    }

    return result_json


if __name__ == "__main__":
    condition_df = pd.read_csv('../data/all_conditions_clean.csv')
    condition_df = condition_df[~condition_df['procedure_source_value'].isin(['SNOMED|232717009','SNOMED|119564002'])]
    condition_df['condition_list'] = condition_df['condition_list'].apply(lambda x: x[1:-1].split(','))

    output_file = "../data/7d_window_clm.jsonl"
    # output_file = "../data/test2_clm.jsonl"

    with open(output_file, "w") as f:
        for index, row in tqdm(condition_df.iterrows(), total=condition_df.shape[0], desc="Processing persons"):
            result_json = get_condition_occurrence_window(row, engine)
            f.write(json.dumps(result_json, ensure_ascii=False) + "\n")

    # no_poi_df = pd.read_csv('../data/condition_list_all.csv')
    # no_poi_df = no_poi_df[~no_poi_df['procedure_source_value'].isin(['SNOMED|232717009','SNOMED|119564002'])]
    # no_poi_df = no_poi_df[no_poi_df['condition_list']=='[]']
    # no_poi_df['condition_list'] = no_poi_df['condition_list'].apply(lambda x: x[1:-1].split(','))
    # no_poi_df = no_poi_df[['birth_datetime','age','gender_concept_id','race_concept_id','person_id','procedure_source_value','procedure_datetime']]
    # output_file = "../data/7d_window_clm_no_poi.jsonl"

    # with open(output_file, "w") as f:
    #     for index, row in tqdm(no_poi_df.iterrows(), total=no_poi_df.shape[0], desc="Processing persons"):
    #         result_json = get_condition_occurrence_window_no_cut_time(row, engine)
    #         f.write(json.dumps(result_json, ensure_ascii=False) + "\n")
