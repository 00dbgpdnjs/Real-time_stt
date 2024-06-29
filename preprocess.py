'''
한국어 음성대화 데이터셋 전처리
subcommand
    - audio
    - file
'''

import argparse
from utils import PrepareDataset

def audio_process(config) -> None:
    print('Start audio processing')
    preprocessor = PrepareDataset()
    preprocessor.process_audio(
        source_dir=config.target_dir,
        remove_original_audio=config.remove_original_audio,
    )
    
def file_process(config) -> None:
    print('Start file processing')
    
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='Korean speech dataset pre-processor',
        description = 'Process Korean speech dataset'
    )
    sub_parser = parser.add_subparsers(title='sub-command')
    
    # Parser for sub-command 'audio'
    parser_audio = sub_parser.add_parser(
        'audio',
        help = 'sub-command for audio processing' # 도움말
    )
    parser_audio.add_argument(
        '--target-dir', '-t',
        required=True,
        help='directory of audio files'
    )
    parser_audio.add_argument(
        '--remove-original-audio', '-r',
        action='store_true' # -r 만 주면 (인자 값 필요x) true, 안주면 false
    )   
    parser_audio.set_defaults(func=audio_process) # 파싱 후 해당 함수의 파라미터(config)로 전달됨
    
    # Parser for sub-command 'file'
    parser_file = sub_parser.add_parser(
        'file',
        help = 'handling txt encoding, generate pkl/csv file,\
            or split file (train/test)'
    )
    
    # TODO: file 관련 argument 추가
    
    parser_file.set_defaults(func=file_process)     
    
    #sub cmd 두 개(audio, file)을 하나의 객체로
    config = parser.parse_args()    
    return config

if __name__ == '__main__':
    config = get_parser()
    # 등록한 func로 config를 줘라
    # -ex) file의 경우 file_process로 등록함
    config.func(config) 
