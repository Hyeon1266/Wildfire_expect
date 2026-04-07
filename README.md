# 산불 발생 위험 예측

위성·기상 데이터를 묶어서 한반도 격자별 산불 발생 가능성을 예측해보는 프로젝트입니다.  
처음부터 확산 예측까지 같이 보려 했지만, 데이터 기간과 해상도가 일정하지 않아서 이번에는 발생 여부 분류 파이프라인부터 먼저 정리했습니다.

## 사용 기술

- 주요: pandas, numpy, scikit-learn
- 보조: xarray, rasterio, matplotlib, PyYAML, joblib

## 실행 방법

```bash
pip install -r requirements.txt
```

원본 데이터는 아래 폴더에 넣어서 사용했습니다.

- data/raw/firms : FIRMS csv 파일
- data/raw/era5_land : ERA5-Land nc 파일
- data/raw/dem : SRTM hgt 또는 tif 파일
- data/raw/worldcover : ESA WorldCover tif 파일

설정값은 config.yaml에서 확인하면 됩니다.

- bbox
- grid_size
- start_date / end_date
- valid_start_date / test_start_date

```bash
python main.py --step all
```

실행 후에는 outputs/ 폴더 아래에 모델 파일, 검증 지표, 테스트 결과, 그래프가 저장됩니다.

## 데이터

- FIRMS 화재 감지 데이터
- ERA5-Land 기상 데이터
- SRTM DEM 지형 데이터
- ESA WorldCover 토지피복 데이터

샘플 데이터는 아래 링크에 따로 정리해두었습니다.  
https://drive.google.com/file/d/1DReZptR1HdvIDnC8vF6YMdeXmvUIf_IJ/view?usp=sharing

## 모델 / 평가

이번 버전에서는 Logistic Regression, Random Forest, Histogram Gradient Boosting 세 모델을 먼저 비교했습니다.  
복잡한 튜닝보다는 기본 파이프라인이 실제로 돌아가는지 확인하는 쪽에 더 집중했습니다.

모델 비교는 검증 단계에서 PR-AUC를 우선으로 보고, ROC-AUC와 F1도 같이 확인했습니다.

## 시행착오

- 처음에는 확산 예측까지 같이 해보려고 했는데, 데이터 기간과 해상도가 제각각이라 바로 확장하기는 무리였습니다. 그래서 이번에는 발생 위험 예측부터 먼저 정리했습니다.
- FIRMS에서는 화재 강도까지 세밀하게 반영하기보다, 먼저 그 날짜-격자에서 감지가 있었는지를 라벨로 잡는 쪽으로 단순화했습니다.
- 기상 변수도 처음부터 너무 많이 넣지 않고 온도, 이슬점, 풍속, 강수, 토양수분 정도만 먼저 사용했습니다.
- 데이터 기간이 짧아서 결과를 일반화해서 해석하기에는 한계가 있습니다. 그래서 이번 결과는 "가능성 확인"에 더 가깝습니다.

## 아쉬운 점

- 무료로 바로 구할 수 있는 데이터 위주로 작업하다 보니 기간이 짧았습니다.
- 지역별 식생 상태나 사람 활동 변수까지는 충분히 반영하지 못했습니다.
- 아직은 발생 위험 예측 단계라서, 실제 확산 방향이나 피해 범위까지 설명하긴 어렵습니다.

## 앞으로 보완하고 싶은 점

- 분석 기간 더 길게 확보하기
- 식생 지수 추가하고 불균형 대응 더 보기
- 확산 위험 해석으로 확장하기

Hyeon1266
