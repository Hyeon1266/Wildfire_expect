import argparse
import random
from pathlib import Path

import numpy as np
import yaml

from preprocess import build_dataset
from train import run_training
from evaluate import run_evaluation

CFG_FILE = "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="산불 발생 위험 예측 파이프라인")
    parser.add_argument("--config", default=CFG_FILE, help="설정 파일 경로")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "preprocess", "train", "evaluate"],
        help="실행할 단계",
    )
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / path

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾지 못했음: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    paths = cfg["paths"]
    targets = [
        paths["firms_dir"],
        paths["era5_dir"],
        paths["worldcover_dir"],
        paths["dem_dir"],
        paths["processed_dir"],
        Path(paths["outputs_dir"]) / "models",
        Path(paths["outputs_dir"]) / "metrics",
        Path(paths["outputs_dir"]) / "figures",
    ]

    for target in targets:
        Path(target).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    ensure_dirs(cfg)
    set_seed(int(cfg.get("project", {}).get("seed", 1266)))

    dataset_path = str(Path(cfg["paths"]["processed_dir"]) / "wildfire_dataset.csv")
    train_result = None

    if args.step in {"all", "preprocess"}:
        print("전처리부터 시작합니다.")
        dataset_path = build_dataset(cfg)

    if args.step in {"all", "train"}:
        print("모델 학습을 진행합니다.")
        train_result = run_training(cfg, dataset_path)

    if args.step in {"all", "evaluate"}:
        print("테스트 평가를 진행합니다.")
        run_evaluation(cfg, train_result)

    print("끝났습니다. 결과는 outputs 폴더에 저장했습니다.")


if __name__ == "__main__":
    main()
