import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from train import load_data, predict_scores, score_binary_classifier


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_feature_names(model, feature_info):
    preprocessor = model.named_steps["preprocessor"]
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return feature_info.get("feature_cols", [])


def get_feature_importance(model, feature_names):
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        return pd.DataFrame()

    n = min(len(feature_names), len(values))
    result = pd.DataFrame({"feature": feature_names[:n], "importance": values[:n]})
    return result.sort_values("importance", ascending=False).head(20)


def save_bar_chart(df, x_col, y_col, path, title):
    plt.figure(figsize=(8, 4))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_evaluation(cfg, train_result=None):
    paths = cfg["paths"]
    target = cfg["data"].get("target_col", "fire_occurrence")
    threshold = float(cfg["train"].get("threshold", 0.5))

    output_dir = Path(paths["outputs_dir"])
    metric_dir = ensure_dir(output_dir / "metrics")
    figure_dir = ensure_dir(output_dir / "figures")

    train_result = train_result or {}
    dataset_path = train_result.get(
        "dataset_path",
        str(Path(paths["processed_dir"]) / "wildfire_dataset.csv"),
    )
    model_path = train_result.get(
        "best_model_path",
        str(output_dir / "models" / "best_model.joblib"),
    )

    print("테스트 데이터 준비")
    df = load_data(dataset_path)

    test_start = pd.to_datetime(cfg["train"]["test_start_date"])
    test_df = df[df["date"] >= test_start].copy()

    if test_df.empty:
        raise ValueError("테스트 데이터 없음")

    with open(metric_dir / "feature_info.json", "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    model = joblib.load(model_path)
    feature_cols = feature_info["feature_cols"]

    x_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)
    y_prob = predict_scores(model, x_test)

    test_metrics = score_binary_classifier(y_test, y_prob, threshold)
    pd.DataFrame([test_metrics]).to_csv(
        metric_dir / "test_metrics.csv", index=False, encoding="utf-8-sig"
    )

    prediction_df = test_df[["date", "grid_id", target]].copy()
    prediction_df["proba"] = y_prob
    prediction_df["pred"] = (y_prob >= threshold).astype(int)
    prediction_df.to_csv(
        metric_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    valid_metric_path = metric_dir / "validation_metrics.csv"
    if valid_metric_path.exists():
        valid_df = pd.read_csv(valid_metric_path)
        save_bar_chart(
            valid_df,
            "model_name",
            "pr_auc",
            figure_dir / "model_compare_pr_auc.png",
            "검증 PR-AUC 비교",
        )

    importance_df = get_feature_importance(model, get_feature_names(model, feature_info))
    importance_path = figure_dir / "feature_importance.png"

    if not importance_df.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
        plt.title("피처 중요도")
        plt.tight_layout()
        plt.savefig(importance_path, dpi=150)
        plt.close()
        # TODO: 날짜별로 예측 확률 지도 그려보기

    print(
        f"테스트 결과 — PR-AUC {test_metrics['pr_auc']:.4f} | "
        f"ROC-AUC {test_metrics['roc_auc']:.4f} | "
        f"F1 {test_metrics['f1']:.4f}"
    )

    return {
        "test_metrics_path": str(metric_dir / "test_metrics.csv"),
        "test_predictions_path": str(metric_dir / "test_predictions.csv"),
        "feature_importance_path": str(importance_path) if importance_path.exists() else None,
    }


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg)
