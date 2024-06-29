'''Utilities for fine-tunning'''
from random import shuffle
import pandas as pd
import pickle
import numpy as np
import librosa as lr
import soundfile as sf
import os
from tqdm import tqdm

class PrepareDataset:
    def __init__(self, audio_dir: str = './data/audio') -> None: 
        self.VOICE_DIR = audio_dir
         
    def pcm2audio(
        self,
        audio_path: str,
        ext: str = 'wav',
        save_file: bool = True,
        remove: bool = False, # (pcm을 wav로 변경했으니까) pcm 삭제하고 싶은지
    ) -> object:
        '''참고 블로그: https://noggame.tistory.com/15'''
        buf = None
        with open(audio_path, 'rb') as tf:
            buf = tf.read()
            # zero (0) padding
            #  경우에 따라 pcm 파일 길이가 8bit(1byte)로
            #  나누어 떨어지지 않는 경우가 있어 0으로 패딩을 더해줌
            #  패딩하지 않으면 numpy나 librosa 사용 시 오류날 수 있음
            buf = buf+b'0' if len(buf)%2 else buf
        pcm_data = np.frombuffer(buf, dtype='int16')
        wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2) # wav로 바뀜
        
        if save_file:
            save_file_name = audio_path.replace('.pcm', f'.{ext}')
            sf.write(
                file=save_file_name,
                data=wav_data,
                samplerate=16000, # whisper에서 1초에 160000 샘플링 한 것만 받음
                format='WAV',
                endian='LITTLE',
                subtype='PCM_16'
            )
            
        if remove:
            if os.path.isfile(audio_path):
                os.remove(audio_path)
        
        return wav_data
    
    def process_audio(
        self,
        source_dir: str,
        remove_original_audio: bool = True
    ) -> None:
        '''.pcm 파일을 .wav 파일로 변환 후 현재 디렉토리에 저장'''
        print(f'source_dir: {source_dir}')
        sub_directories = sorted(os.listdir(source_dir))
        print(f'Processing audios: {len(sub_directories)} diretories')
        for directory in tqdm(sub_directories, desc=f'Processing directory: {source_dir}'):
            # if os.path.isdir(directory):
            if os.path.isdir(os.path.join(source_dir, directory)):
                files = os.listdir(os.path.join(source_dir, directory))
                for file_name in files:
                    if file_name.endswith('.pcm'):
                        self.pcm2audio(
                            audio_path=os.path.join(source_dir, directory, file_name),
                            ext='wav',
                            remove=remove_original_audio,
                        )
            else:
                file_name = directory
                if file_name.endswith('.pcm'):
                    self.pcm2audio(
                        audio_path=os.path.join(source_dir, file_name),
                        ext='wav',
                        remove=remove_original_audio,
                    )               
             
    def convert_text_utf(self, file_path:str) -> None:
        '''파일 인코딩 변경: cp494 -> utf-8로 변환하여 저장'''
        try:
            with open(file_path, 'rt', encoding='cp949') as f:
                lines = f.readlines()
        except:
            with open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        # G 파일의 내용을 utf-8 인코딩으로 저장(덮어쓰기)
        with open(file_path, 'wt', encoding='utf-8') as f: 
            for line in lines:
                f.write(line)
    
    def convert_all_files_to_utf8(self, target_dir: str) -> None:
        '''디렉토리 내부의 모든 텍스트 파일을 utf-8 '''
        print(f'Target directory: {target_dir}')
        sub_directories = sorted(os.listdir(target_dir))
        num_files = 0
        for directory in tqdm(sub_directories, desc='converting cp949 -> utf8'):
            files = sorted(os.listdir(os.path.join(target_dir, directory)))
            for file_name in files:
                if file_name.endswith('.txt'):
                    self.convert_text_utf(
                        os.path.join(target_dir, directory, file_name)
                    )
                    num_files += 1
        print(f'{num_files} txt files are converted.')
    
    def split_whole_data(self, target_file:str) -> None:
        '''전체 데이터 파일 (train.trn)을 그룹별로 구분
        For example, in train.trn file
            KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm :: 'some txt'
                -> this file will be stored in train_KsponSpeech_01.trn
            KsponSpeech_02/KsponSpeech_0125/KsponSpeech_124001.pcm :: 'some txt'
                -> this file will be stored in train_KsponSpeech_02.trn
        '''
        with open(target_file, 'rt') as f: # 이미 utf-8로 인코딩해서(읽을 수 있어서) encoding 파라미터는 전달안해도됨
            lines = f.readlines()
            data_group = set()
            for line in lines:
                data_group.add(line.split('/')[0]) # 1st: ex) KsponSpeech_01
        data_group = sorted(list(data_group))
        data_dic = { group: [] for group in data_group} # 'group: []'는 for문의 'data_dic[group] = []' 과 동일
        for line in lines:
            data_dic[line.split('/')[0]].append(line)
        # Save file seperately
        # target_file: data/info/train.trn
        # 아래 두 라인 보다 os.path.dirname(file_path) 가 더 안전
        save_dir = target_file.split('/')[:-1]
        save_dir = '/'.join(save_dir)
        for group, line_list in data_dic.items():
            file_path = os.path.join(save_dir, f'train_{group}.trn')
            with open(file_path, 'wt', encoding='utf-8') as f:
                for text in line_list:
                    f.write(text)
                print(f'File created -> {file_path}')
        print('Done!')
        
    def get_dataset_dict(self, file_name: str, ext: str = 'wav') -> dict:
        '''path_dir에 있는 파일을 dict 형태로 가공하여 리턴
            return data_dict = {
                'audio' : ['file_path1', 'file_path2', ...],
                'text' : ['text1', 'text2', ...]
        }'''
        data_dic = {'path': [], 'sentence': []}
        print(f'file_name: {file_name}')
        with open(file_name, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                audio, text = line.split('::')
                audio = audio.strip()
                audio = os.path.join( # 파이튜닝 시 데이터 줄 때 절대 경로를 받아서
                    os.getcwd(), # /home/hyw/finetune
                    self.VOICE_DIR.replace('./', ''), # VOICE_DIR가 ./로 시작할 수도 있어서
                    audio
                )
                if audio.endswith('.pcm'):
                    audio = audio.replace('.pcm', f'.{ext}')
                text = text.strip()
                data_dic['path'].append(audio)
                data_dic['sentence'].append(text)
        return data_dic
    
    def save_trn_to_pkl(self, file_name: str) -> None:
        '''.trn 파일을 dict(;json)로 만든 후 .pkl 바이너리로 그냥 저장(dump)'''
        data_dict = self.get_dataset_dict(file_name)
        # pickle file dump
        file_name_pickle = file_name + '.dic.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(data_dict, f)
        print('Dataset is saved via dictionary pickle')
        print(f'Dataset path: {file_name_pickle}')
    
    def save_trn_to_csv(self, file_name: str) -> None:
        '''.trn 파일을 .csv로 저장'''
        data_dic = self.get_dataset_dict(file_name)
        file_name_csv = file_name.split('.')[:-1]
        file_name_csv = ''.join(file_name_csv) + '.csv'
        if file_name.startswith('.'): # 이 경우(ex. ./~/bbb.trn) '', '/~/bbb', 'trn'로 split되기 때문에 맨 앞 점을 보존해주려고. 안그러면 절대경로가 됨
            file_name_csv = '.' + file_name_csv
        data_dic = pd.DataFrame(data_dic)
        data_dic.to_csv(file_name_csv, index=False, header=True) # header : 제목 행(path,sentence) 줄지 -> True 해야함 (whisper data needs header)
        print('Dataset is saved via csv')
        print(f'Dataset path: {file_name_csv}')
    
    def split_train_test(self, target_file: str, train_size: float = 0.8) -> None:
        '''입력 파일(.trn)을 train/test 분류하여 저장
            if train_size = 0.8,
                train:test = 80%:20%
        '''
        with open(target_file, 'rt') as f:
            data = f.readlines()
            train_num = int(len(data) *train_size)
        
        # header (header=True) in csv file need!!
        # - whisper data needs header
        header = None
        if target_file.endswith('.csv'):
            header = data[0]
            data = data[1:]
            train_num = int(len(data) *train_size)
        shuffle(data)
        data_train = sorted(data[0:train_num])
        data_test = sorted(data[train_num:])
        
        # train_set 파일 저장
        # - os.path.splitext를 사용하면 간소화될 듯
        train_file = target_file.split('.')[:-1]
        train_file = ''.join(train_file) + '_train.csv'
        if target_file.startswith('.'):
            train_file = '.' + train_file
        with open(train_file, 'wt', encoding='utf-8') as f:
            if header:
                f.write(header)
            for line in data_train:
                f.write(line)
        print(f'Train_dataset saved -> {train_file} ({train_size*100:.1f}%)')
        
        # test_set 파일 저장
        test_file = target_file.split('.')[:-1]
        test_file = ''.join(test_file) + '_test.csv'
        if target_file.startswith('.'):
            test_file = '.' + test_file
        with open(test_file, 'wt', encoding='utf-8') as f:
            if header:
                f.write(header)
            for line in data_test:
                f.write(line)
        print(f'Test_dataset saved -> {test_file} ({(1.0 - train_size)*100:.1f}%)')
    
    def remove_all_text_files(self, target_dir: str, ext: str = 'txt') -> None:
        '''디렉토리 내부의 모든 특정 형태 파일(in our case, txt) 삭제'''
        print(f'Target directory: {target_dir}')
        sub_directories = sorted(os.listdir(target_dir))
        num_files = 0
        for directory in tqdm(sub_directories, desc=f'Delete all {ext} files'):
            files  = os.listdir(os.path.join(target_dir, directory))
            for file_name in files:
                if file_name.endswith(f'.{ext}'):
                    os.remove(
                        os.path.join(target_dir, directory, file_name)
                    )
                    num_files += 1
        print(f'Removed {num_files} txt files')
    
        
# 아래 각 단락들의 테스트 중 실제 사용될 코드묶음은 argparser로 file 하부 명령어에 등록함
if __name__ == '__main__': # 자기 자신으로 호출됐을 때
    prepareds = PrepareDataset()
    
    # audio = 'data/audio/eval_clean/KsponSpeech_E00001.pcm'
    # prepareds.pcm2audio(audio_path=audio) # 듣고 wav, wav 지우기
    
    # 01~05 다 wav로 변환해야 하는데 이렇게 하지 않고,
    # 터미널에서 입력이 가능한 형태로 바꿀 것임 (argparse 이용) 
    source_dir = 'data/audio/KsponSpeech_02'
    # prepareds.process_audio(source_dir=source_dir)
    
    # 해당 파일 열고 $ python utiles.py 를 실행하여 utf-8로 변환됐는지(읽을 수 있는지) 확인
    # text_file = 'data/audio/KsponSpeech_05/KsponSpeech_0497/KsponSpeech_496001.txt'    
    # prepareds.convert_text_utf(file_path=text_file)
    
    # # 이렇게 하지 않고 argparser로 file 하부 명령어에 등록하여 모든 파일의 utf-8 변환 수행할 것임
    # prepareds.convert_all_files_to_utf8(source_dir)
    
    # target_file = 'data/info/train.trn'
    # prepareds.split_whole_data(target_file)
    
    # target_file = 'data/info/eval_clean.trn'
    # data_dict = prepareds.get_dataset_dict(target_file)
    # print(data_dict)
    
    # prepareds.save_trn_to_pkl(target_file)
    # prepareds.save_trn_to_csv(target_file)
    
    # target_file = './data/info/train_KsponSpeech_02.csv'
    # prepareds.split_train_test(target_file)
    
    # prepareds.remove_all_text_files(source_dir)
    pass