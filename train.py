import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).copy()


def split_data(df, valid_start, test_start):
    valid_start = pd.to_datetime(valid_start)
    test_start = pd.to_datetime(test_start)

    train = df[df["date"] < valid_start].copy()
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("train/valid/test 분할 결과가 비어 있음")

    return train, valid, test



def score_binary_classifier(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    has_both_classes = y_true.nunique() > 1

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def predict_scores(model, x):
    return model.predict_proba(x)[:, 1]


def build_models(preprocessor, seed):
    # 복잡한 튜닝보다 성격이 다른 모델 3개를 먼저 비교하는 데 집중했다.
    # xgboost도 써보려 했는데 클래스 불균형 대응이 번거로워서 일단 제외
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def run_training(cfg, dataset_path):
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    seed = int(cfg.get("project", {}).get("seed", 1266))
    target = cfg["data"].get("target_col", "fire_occurrence")
    threshold = float(train_cfg.get("threshold", 0.5))

    output_dir = ensure_dir(paths["outputs_dir"])
    model_dir = ensure_dir(output_dir / "models")
    metric_dir = ensure_dir(output_dir / "metrics")

    print("데이터 로드")
    df = load_data(dataset_path)
    train_df, valid_df, test_df = split_data(
        df,
        train_cfg["valid_start_date"],
        train_cfg["test_start_date"],
    )
    print(f"학습 {len(train_df):,}건 / 검증 {len(valid_df):,}건 / 테스트 {len(test_df):,}건")

    drop_cols = {"date", "grid_id", target, "hotspot_count"}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    x_train = train_df[feature_cols]
    y_train = train_df[target].astype(int)
    x_valid = valid_df[feature_cols]
    y_valid = valid_df[target].astype(int)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
    ])
    models = build_models(preprocessor, seed)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"{model_name} 학습 중...")
        model.fit(x_train, y_train)

        valid_prob = predict_scores(model, x_valid)
        valid_metrics = score_binary_classifier(y_valid, valid_prob, threshold)

        results.append({"model_name": model_name, **valid_metrics})
        trained_models[model_name] = model

        print(
            f"  PR-AUC {valid_metrics['pr_auc']:.4f} | "
            f"ROC-AUC {valid_metrics['roc_auc']:.4f} | "
            f"F1 {valid_metrics['f1']:.4f}"
        )

        joblib.dump(model, model_dir / f"{model_name}.joblib")

    score_df = (
        pd.DataFrame(results)
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = score_df.iloc[0]["model_name"]
    best_model_path = model_dir / "best_model.joblib"
    joblib.dump(trained_models[best_model_name], best_model_path)

    score_df.to_csv(metric_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")

    with open(metric_dir / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"검증 기준에서 가장 괜찮았던 모델은 {best_model_name}입니다.")

    return {
        "dataset_path": str(dataset_path),
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_size": int(len(test_df)),
        "best_model_name": str(best_model_name),
        "best_model_path": str(best_model_path),
        "validation_metrics_path": str(metric_dir / "validation_metrics.csv"),
    }


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_training(cfg, str(Path(cfg["paths"]["processed_dir"]) / "wildfire_dataset.csv"))
