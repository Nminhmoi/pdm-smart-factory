"""
Script suy luận (Inference / Prediction) cho bài toán Bảo trì dự đoán (Predictive Maintenance).
Được thiết kế để chạy độc lập từ dòng lệnh hoặc import làm module.
"""

import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dự đoán nguy cơ máy hỏng hóc (Machine Failure Prediction) bằng Pipeline học máy."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Đường dẫn tới file pipeline đã huấn luyện (.joblib).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="M",
        choices=["L", "M", "H"],
        help="Loại máy / cấp chất lượng sản phẩm (L, M, H).",
    )
    parser.add_argument(
        "--air_temp",
        type=float,
        default=300.0,
        help="Nhiệt độ không khí xung quanh [K] (ví dụ: 300.0).",
    )
    parser.add_argument(
        "--process_temp",
        type=float,
        default=310.0,
        help="Nhiệt độ quy trình [K] (ví dụ: 310.0).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1500.0,
        help="Tốc độ quay trục chính [rpm] (ví dụ: 1500.0).",
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=40.0,
        help="Mô-men xoắn [Nm] (ví dụ: 40.0).",
    )
    parser.add_argument(
        "--tool_wear",
        type=float,
        default=100.0,
        help="Thời gian mòn dụng cụ cắt [phút] (ví dụ: 100.0).",
    )
    return parser.parse_args()


def predict(input_data: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH):
    """
    Nạp pipeline đã đóng gói và thực hiện dự đoán.
    
    Parameters:
    -----------
    input_data : pd.DataFrame
        DataFrame chứa các đặc trưng đầu vào chuẩn tên cột.
    model_path : str
        Đường dẫn file .joblib.
        
    Returns:
    --------
    dict: Kết quả dự đoán gồm nhãn phân loại (0/1) và xác suất lỗi (%).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại '{model_path}'. "
            f"Vui lòng chạy notebook huấn luyện để xuất file 'best_model.joblib' trước."
        )

    pipeline = joblib.load(model_path)

    prediction = pipeline.predict(input_data)[0]
    
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(input_data)[0][1])

    return {
        "prediction": int(prediction),
        "status": "HỎNG HÓC (CẢNH BÁO)" if prediction == 1 else "HOẠT ĐỘNG BÌNH THƯỜNG",
        "failure_probability": probability,
    }


def main():
    args = parse_args()

    sample_df = pd.DataFrame([{
        "Type": args.type,
        "Air temperature [K]": args.air_temp,
        "Process temperature [K]": args.process_temp,
        "Rotational speed [rpm]": args.speed,
        "Torque [Nm]": args.torque,
        "Tool wear [min]": args.tool_wear,
    }])

    print("=" * 60)
    print(" HỆ THỐNG BẢO TRÌ DỰ ĐOÁN NHÀ MÁY THÔNG MINH - PREDICT CLI")
    print("=" * 60)
    print("Dữ liệu đầu vào:")
    for k, v in sample_df.iloc[0].items():
        print(f"  - {k}: {v}")
    print("-" * 60)

    try:
        result = predict(sample_df, args.model_path)
        print(f"Kết quả dự đoán : [{result['prediction']}] -> {result['status']}")
        if result['failure_probability'] is not None:
            print(f"Xác suất máy hỏng: {result['failure_probability'] * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"[THÔNG BÁO]: {e}")
        print("Gợi ý: Sau khi huấn luyện ở notebook 02, file models/best_model.joblib sẽ được tạo ra.")
    print("=" * 60)


if __name__ == "__main__":
    main()
