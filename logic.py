"""
Gas Ratio Analysis - Rule-Based Logic Implementation

DGA 가스 비율 분석 (IEEE C57.104 / IEC 60599):
1. Rogers Ratio Method (IEEE C57.104 Table 5)
2. IEC Ratio Method (IEC 60599 Table 1)
3. Solid Insulation Monitoring (CO2/CO ratio)
"""

from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

import os
import math
import pandas as pd
import mlflow
import mlflow.pyfunc

from coreflow.exceptions import MLflowError
from coreflow.utils.logging_helpers import setup_model_logger
from coreflow.utils.mlflow_helpers import init_mlflow, log_deploy_bundle

MODEL_NAME = "gas_ratio_analysis"
logger = setup_model_logger(MODEL_NAME)


def _get_experiment_name() -> str:
    """config.yml의 mlflow.experiment_name을 읽어 반환"""
    import yaml
    config_path = Path(__file__).parent / "config.yml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("mlflow", {}).get("experiment_name", MODEL_NAME)
    except Exception:
        return MODEL_NAME


# =============================================================================
# 핵심 가스 비율 분석 로직
# =============================================================================

def _safe_ratio(numerator: float, denominator: float) -> float:
    """분모가 0일 경우 infinity 반환"""
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def compute_ratios(
    h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float
) -> Dict[str, float]:
    """
    기본 가스 비율 계산.

    R1 = C2H2 / C2H4
    R2 = CH4  / H2
    R3 = C2H4 / C2H6
    """
    return {
        "R1": _safe_ratio(c2h2, c2h4),
        "R2": _safe_ratio(ch4,  h2),
        "R3": _safe_ratio(c2h4, c2h6),
    }


def rogers_ratio_method(r1: float, r2: float, r3: float) -> Dict[str, str]:
    """
    Rogers Ratio Method (IEEE C57.104 Table 5).

    Returns:
        case: 고장 케이스 번호
        diagnosis: 진단 결과 문자열
    """
    if r1 < 0.1 and r2 > 1.0 and r3 > 3.0:
        return {"case": "Case 5", "diagnosis": "고온 열적 고장 (Thermal Fault > 700°C)"}
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 < 3.0:
        return {"case": "Case 4", "diagnosis": "열적 고장 (Thermal Fault < 700°C)"}
    if r1 < 0.1 and 0.1 <= r2 < 1.0 and 1.0 <= r3 < 3.0:
        return {"case": "Case 3", "diagnosis": "저온 열적 고장 (Low Temperature Thermal)"}
    if 0.1 <= r1 < 3.0 and 0.1 <= r2 < 1.0 and r3 > 3.0:
        return {"case": "Case 2", "diagnosis": "고에너지 아크 방전 (High Energy Arcing)"}
    if r1 < 0.1 and r2 < 0.1 and r3 < 1.0:
        return {"case": "Case 1", "diagnosis": "저에너지 부분 방전 (Low Energy PD)"}
    if r1 < 0.1 and 0.1 <= r2 < 1.0 and r3 < 1.0:
        return {"case": "Case 0", "diagnosis": "정상 (Normal)"}

    return {"case": "Unknown", "diagnosis": "판정 불가 (Unknown - 비율이 정의된 케이스에 해당하지 않음)"}


def iec_ratio_method(r1: float, r2: float, r3: float) -> Dict[str, str]:
    """
    IEC Ratio Method (IEC 60599 Table 1).

    Returns:
        fault_type: 고장 유형 코드
        diagnosis: 진단 결과 문자열
    """
    if 0.6 <= r1 < 2.5 and 0.1 <= r2 < 1.0 and r3 > 2.0:
        return {"fault_type": "D2", "diagnosis": "고에너지 방전 (High Energy Discharge)"}
    if r1 > 1.0 and 0.1 <= r2 < 0.5 and r3 > 1.0:
        return {"fault_type": "D1", "diagnosis": "저에너지 방전 (Low Energy Discharge)"}
    if r1 < 0.2 and r2 > 1.0 and r3 > 4.0:
        return {"fault_type": "T3", "diagnosis": "고온 열적 고장 > 700°C (Thermal Fault > 700°C)"}
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 < 4.0:
        return {"fault_type": "T2", "diagnosis": "열적 고장 300~700°C (Thermal Fault 300~700°C)"}
    if r2 > 1.0 and r3 < 1.0:
        return {"fault_type": "T1", "diagnosis": "저온 열적 고장 < 300°C (Thermal Fault < 300°C)"}
    if r2 < 0.1 and r3 < 0.2:
        return {"fault_type": "PD", "diagnosis": "부분 방전 (Partial Discharge)"}

    return {"fault_type": "Unknown", "diagnosis": "판정 불가 (Unknown - 비율이 정의된 케이스에 해당하지 않음)"}


def solid_insulation_monitoring(co: float, co2: float) -> Dict[str, str]:
    """
    CO2/CO 비율로 고체 절연 상태 모니터링.

    Returns:
        ratio: CO2/CO 비율 문자열
        status: 정상/경고/위험
        diagnosis: 상세 진단
    """
    if co2 == 0.0 and co == 0.0:
        return {
            "ratio": "N/A",
            "status": "Invalid",
            "diagnosis": "CO와 CO2가 모두 0 — 측정 불가",
        }

    if co == 0.0:
        return {
            "ratio": "inf",
            "status": "정상 (Normal)",
            "diagnosis": "CO=0, CO2/CO=inf — 고체 절연 정상",
        }

    ratio = co2 / co
    ratio_str = f"{ratio:.2f}"
    co_high = co >= 1000.0

    if ratio < 3.0:
        status = "위험 (Danger)"
        diagnosis = f"CO2/CO={ratio_str} < 3.0 — 고체 절연 심각 열화 의심"
    elif ratio <= 10.0:
        status = "경고 (Warning)" if co_high else "주의 (Caution)"
        diagnosis = f"CO2/CO={ratio_str} (3.0~10.0) — 고체 절연 열화 주의"
    else:
        status = "경고 (Warning)" if co_high else "정상 (Normal)"
        diagnosis = f"CO2/CO={ratio_str} > 10.0 — 고체 절연 정상"

    if co_high:
        diagnosis += f" [CO={co:.1f}ppm >= 1000ppm — 위험 수준 상향]"

    return {
        "ratio": ratio_str,
        "status": status,
        "diagnosis": diagnosis,
    }


def analyze_gas_ratio(
    h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float, co: float, co2: float
) -> Dict[str, Any]:
    """
    전체 가스 비율 분석 수행.

    Returns:
        R1, R2, R3, rogers_case, rogers_diagnosis,
        iec_fault_type, iec_diagnosis,
        insulation_ratio, insulation_status, insulation_diagnosis
    """
    ratios = compute_ratios(h2, ch4, c2h2, c2h4, c2h6)
    r1, r2, r3 = ratios["R1"], ratios["R2"], ratios["R3"]

    rogers = rogers_ratio_method(r1, r2, r3)
    iec = iec_ratio_method(r1, r2, r3)
    insulation = solid_insulation_monitoring(co, co2)

    return {
        "R1": None if math.isinf(r1) else r1,
        "R2": None if math.isinf(r2) else r2,
        "R3": None if math.isinf(r3) else r3,
        "rogers_case": rogers["case"],
        "rogers_diagnosis": rogers["diagnosis"],
        "iec_fault_type": iec["fault_type"],
        "iec_diagnosis": iec["diagnosis"],
        "insulation_ratio": insulation["ratio"],
        "insulation_status": insulation["status"],
        "insulation_diagnosis": insulation["diagnosis"],
    }


# =============================================================================
# MLflow PythonModel 래퍼 (서빙용)
# =============================================================================

class RuleModel(mlflow.pyfunc.PythonModel):
    """가스 비율 분석 로직을 MLflow 모델로 서빙하기 위한 래퍼 클래스"""

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        elif not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        return rule_logic(model_input)


def rule_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 입력을 받아 가스 비율 분석 결과를 DataFrame으로 반환.

    입력 컬럼: h2, ch4, c2h2, c2h4, c2h6, co, co2 (ppm, float)
    출력 컬럼: R1, R2, R3, rogers_case, rogers_diagnosis,
               iec_fault_type, iec_diagnosis,
               insulation_ratio, insulation_status, insulation_diagnosis
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    records = []
    for _, row in df.iterrows():
        h2   = float(row.get("h2",   0.0))
        ch4  = float(row.get("ch4",  0.0))
        c2h2 = float(row.get("c2h2", 0.0))
        c2h4 = float(row.get("c2h4", 0.0))
        c2h6 = float(row.get("c2h6", 0.0))
        co   = float(row.get("co",   0.0))
        co2  = float(row.get("co2",  0.0))

        result = analyze_gas_ratio(h2, ch4, c2h2, c2h4, c2h6, co, co2)
        records.append(result)

    return pd.DataFrame(records)


# =============================================================================
# Airflow 파이프라인 스테이지
# =============================================================================

def prepare_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """데이터 준비"""
    logger.info("prepare_data: Gas Ratio Analysis logic model")
    return {}


def run_logic(context: Dict[str, Any], inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """비즈니스 로직 실행"""
    logger.info(f"run_logic inputs={inputs}")

    sample_inputs = inputs if inputs else {
        "h2": 80.0, "ch4": 100.0, "c2h2": 2.0,
        "c2h4": 50.0, "c2h6": 30.0, "co": 300.0, "co2": 2000.0,
    }

    df = pd.DataFrame([sample_inputs])
    results_df = rule_logic(df)
    results = results_df.to_dict(orient="records")

    logger.info(f"Processed {len(results)} records")
    return results


def export_result(context: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, str]:
    """결과 내보내기"""
    logger.info(f"export_result: {results}")
    return {"exported": f"{len(results)}_records"}


def register_logic_model(context: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, str]:
    """서빙을 위해 모델을 MLflow에 등록"""
    logger.info(f"register_logic_model results={results}")
    try:
        logger.info("Registering Gas Ratio Analysis rule model to MLflow")

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        experiment_name = _get_experiment_name()
        init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

        with mlflow.start_run() as run:
            from mlflow.models.signature import infer_signature

            example_input = pd.DataFrame([
                {"h2": 80.0,  "ch4": 100.0, "c2h2": 2.0,  "c2h4": 50.0,  "c2h6": 30.0, "co": 300.0, "co2": 2000.0},
                {"h2": 10.0,  "ch4": 5.0,   "c2h2": 15.0, "c2h4": 20.0,  "c2h6": 5.0,  "co": 100.0, "co2": 500.0},
                {"h2": 200.0, "ch4": 500.0, "c2h2": 0.5,  "c2h4": 300.0, "c2h6": 80.0, "co": 500.0, "co2": 8000.0},
            ])
            example_output = rule_logic(example_input)
            signature = infer_signature(example_input, example_output)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RuleModel(),
                signature=signature,
                registered_model_name=MODEL_NAME,
            )

            logger.info("Logging deploy bundle to MLflow")
            model_dir = Path(__file__).parent
            log_deploy_bundle(MODEL_NAME, model_dir)

            logger.info(f"Model registered. Run ID: {run.info.run_id}")
            return {
                "run_id": run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/model",
            }

    except Exception as e:
        logger.error(f"Failed to register model: {e}", exc_info=True)
        raise MLflowError(f"Model registration failed: {e}") from e


if __name__ == "__main__":
    test_cases = [
        {"h2": 50.0,  "ch4": 80.0,  "c2h2": 0.1, "c2h4": 10.0, "c2h6": 20.0, "co": 200.0, "co2": 3000.0},
        {"h2": 50.0,  "ch4": 200.0, "c2h2": 1.0, "c2h4": 500.0,"c2h6": 100.0,"co": 300.0, "co2": 4000.0},
        {"h2": 100.0, "ch4": 50.0,  "c2h2": 20.0,"c2h4": 30.0, "c2h6": 10.0, "co": 100.0, "co2": 500.0},
        {"h2": 30.0,  "ch4": 50.0,  "c2h2": 0.5, "c2h4": 20.0, "c2h6": 30.0, "co": 1200.0,"co2": 2400.0},
        {"h2": 30.0,  "ch4": 50.0,  "c2h2": 0.2, "c2h4": 15.0, "c2h6": 25.0, "co": 0.0,   "co2": 1500.0},
    ]

    df_in = pd.DataFrame(test_cases)
    df_out = rule_logic(df_in)

    print("=" * 80)
    print("Gas Ratio Analysis - 테스트 결과")
    print("=" * 80)
    for i, (inp, (_, out)) in enumerate(zip(test_cases, df_out.iterrows())):
        print(f"\n[케이스 {i+1}] H2={inp['h2']}, CH4={inp['ch4']}, C2H2={inp['c2h2']}, "
              f"C2H4={inp['c2h4']}, C2H6={inp['c2h6']}, CO={inp['co']}, CO2={inp['co2']}")
        print(f"  비율: R1={out['R1']}, R2={out['R2']}, R3={out['R3']}")
        print(f"  Rogers: [{out['rogers_case']}] {out['rogers_diagnosis']}")
        print(f"  IEC:    [{out['iec_fault_type']}] {out['iec_diagnosis']}")
        print(f"  절연:   [{out['insulation_status']}] {out['insulation_diagnosis']}")
