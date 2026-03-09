"""
Gas Ratio Analysis BentoML API 테스트 스크립트

Usage:
    python api-test.py
"""

import json
import os
import requests

BENTO_URL = os.getenv("BENTO_URL", "https://mlops.lab.atgdevs.com/gas_ratio_analysis/predict")
MODEL_NAME = "gas_ratio_analysis"


def prepare_request_payload(instances):
    return {"req": {"instances": instances}}


def send_prediction_request(payload):
    try:
        response = requests.post(
            BENTO_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"응답 코드: {e.response.status_code}")
            print(f"응답 내용: {e.response.text}")
        return None


def main():
    print("=" * 70)
    print(f"{MODEL_NAME} BentoML API 테스트 (가스 비율 분석)")
    print("=" * 70)

    test_cases = [
        {"h2": 50.0,  "ch4": 80.0,  "c2h2": 0.1, "c2h4": 10.0, "c2h6": 20.0, "co": 200.0, "co2": 3000.0},
        {"h2": 50.0,  "ch4": 200.0, "c2h2": 1.0, "c2h4": 500.0,"c2h6": 100.0,"co": 300.0, "co2": 4000.0},
        {"h2": 100.0, "ch4": 50.0,  "c2h2": 20.0,"c2h4": 30.0, "c2h6": 10.0, "co": 100.0, "co2": 500.0},
        {"h2": 30.0,  "ch4": 50.0,  "c2h2": 0.5, "c2h4": 20.0, "c2h6": 30.0, "co": 1200.0,"co2": 2400.0},
    ]

    print("\n[1] 테스트 데이터:")
    for i, tc in enumerate(test_cases, 1):
        print(f"   [{i}] H2={tc['h2']}, CH4={tc['ch4']}, C2H2={tc['c2h2']}, "
              f"C2H4={tc['c2h4']}, C2H6={tc['c2h6']}, CO={tc['co']}, CO2={tc['co2']}")

    payload = prepare_request_payload(test_cases)
    print(f"\n[2] BentoML API 호출: {BENTO_URL}")
    response = send_prediction_request(payload)

    if response is None:
        print("\nAPI 요청 실패")
        return

    print("\n[3] 예측 결과:")
    print("-" * 70)

    predictions = response.get('predictions', [])
    if not predictions:
        print("예측 결과가 비어있습니다.")
        print(f"응답 내용: {json.dumps(response, indent=2, ensure_ascii=False)}")
        return

    for i, pred in enumerate(predictions, 1):
        print(f"\n결과 {i}:")
        print(f"  비율: R1={pred.get('R1')}, R2={pred.get('R2')}, R3={pred.get('R3')}")
        print(f"  Rogers: [{pred.get('rogers_case', 'N/A')}] {pred.get('rogers_diagnosis', 'N/A')}")
        print(f"  IEC:    [{pred.get('iec_fault_type', 'N/A')}] {pred.get('iec_diagnosis', 'N/A')}")
        print(f"  절연:   [{pred.get('insulation_status', 'N/A')}] {pred.get('insulation_diagnosis', 'N/A')}")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
