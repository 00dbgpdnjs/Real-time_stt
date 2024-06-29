'''Utilities for fine-tunning'''
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
    
if __name__ == '__main__': # 자기 자신으로 호출됐을 때
    # audio = 'data/audio/eval_clean/KsponSpeech_E00001.pcm'
    # prepareds = PrepareDataset()
    # prepareds.pcm2audio(audio_path=audio) # 듣고 wav, wav 지우기
    
    # 01~05 다 wav로 변환해야 하는데 이렇게 하지 않고,
    # 터미널에서 입력이 가능한 형태로 바꿀 것임 (argparse 이용) 
    # source_dir = 'data/audio/KsponSpeech_02'
    # prepareds = PrepareDataset()
    # prepareds.process_audio(source_dir=source_dir)
    
    # 해당 파일 열고 $ python utiles.py 를 실행하여 utf-8로 변환됐는지(읽을 수 있는지) 확인
    # text_file = 'data/audio/KsponSpeech_05/KsponSpeech_0497/KsponSpeech_496001.txt'
    
    # prepareds = PrepareDataset()
    # prepareds.convert_text_utf(file_path=text_file)
    
    # 이렇게 하지 않고 argparser로 file 하부 명령어에 등록하여 모든 파일의 utf-8 변환 수행할 것임
    target_dir = 'data/audio/KsponSpeech_05'
    prepareds = PrepareDataset()
    prepareds.convert_all_files_to_utf8(target_dir)