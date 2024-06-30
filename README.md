# finetune_whisper

## dataset
- data 폴더 생성해야함
- Our target data for fine tuning
    - https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123

### data/audio
- 아래와 같은 디렉토리가 있어야 합니다. 
- 각 디렉토리에는 음성 파일(.wav)이 저장되어 있어야 합니다.
    - KsponSpeech_01/
    - KsponSpeech_02/
    - KsponSpeech_03/
    - KsponSpeech_04/
    - KsponSpeech_05/
    - eval_clean/
    - eval_other/

### data/info
- 이 폴더에는 포함되어야 할 내용
    - 단, _test.csv 또는 _train.csv 으로 끝나는 파일은 $ python preprocess.py file -t data/info/train_KsponSpeech_05.csv -s 와 같이 생성 
    - 그냥 csv는 $ python preprocess.py file -t data/info/train_KsponSpeech_05.trn --csv 와 같이 생성
- `dev.trn`
- `eval_clean.csv`
- `eval_clean.trn`
- `eval_other.csv`
- `eval_other.trn`
- `train.csv`
- `train.trn`
- `train_KsponSpeech_01.csv`
- `train_KsponSpeech_01.trn`
- `train_KsponSpeech_01_test.csv`
- `train_KsponSpeech_01_train.csv`
- `train_KsponSpeech_02.csv`
- `train_KsponSpeech_02.trn`
- `train_KsponSpeech_02_test.csv`
- `train_KsponSpeech_02_train.csv`
- `train_KsponSpeech_03.csv`
- `train_KsponSpeech_03.trn`
- `train_KsponSpeech_03_test.csv`
- `train_KsponSpeech_03_train.csv`
- `train_KsponSpeech_04.csv`
- `train_KsponSpeech_04.trn`
- `train_KsponSpeech_04_test.csv`
- `train_KsponSpeech_04_train.csv`
- `train_KsponSpeech_05.csv`
- `train_KsponSpeech_05.trn`
- `train_KsponSpeech_05_test.csv`
- `train_KsponSpeech_05_train.csv`
- `train_test.csv`
- `train_train.csv`

## train
학습 관련 추가해야 할 폴더

- model_archive/
- model_finetuned/ 
    - 파인튜닝 완료된 베스트 모델이 저장될 폴더
- model_output/
    - 매 학습이 끝나면 checkpoint 폴더 자동 생성

###  참고사항
- 추론 단계에서 --base-model 옵션을 이용해 파인튜팅 된 모델과 base-model을 맞춰줘야 합니다.
- (예시)
	- 파인 튜닝된 whisper-small 모델 적용 시 --base-model openai/whisper-small 적용
	- 파인 튜닝된 whisper-medium 모델 적용 시 --base-model openai/whisper-medium 적용

