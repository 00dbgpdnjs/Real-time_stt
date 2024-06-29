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
