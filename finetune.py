'''Fine-tuning whisper (possible: tiny, small, medium, etc.)
References:
    - Master reference -> https://huggingface.co/blog/fine-tune-whisper
    - Korean blog -> https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
'''

import argparse
from datasets import load_dataset, DatasetDict
from utils import get_unique_directory
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor
)

def get_config() -> argparse.ArgumentParser:
    '''Whisper finetuning args parsing function'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base-model', '-b',
        required=True,
        help='Base model for tokenizer, processor, feature extraction. \
            Ex. "openai/whisper-tiny", "openai/whisper=small", etc. from huggingface'
    )
    parser.add_argument( # fine-tune을 했으면 이어서 할 수 있음
        '--pretrained-model', '-p',
        default='',
        help='Pre-trained model from huggingface or local computer. \
            If not given, we will set the models same to --base-model (-b)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./model_output',
        help='Directory for saving whisper finetune outputs'
    )
    parser.add_argument(
        '--finetuned-model-dir', '-ft',
        required=True,
        help='Directory for saving fine-tuned model (best model after train)'
    )
    ###############
    parser.add_argument(
        '--train-set', '-t',
        required=True,
        help='Train dataset name (file name or file path)'
    )
    parser.add_argument(
        '--valid-set', '-v',
        required=True,
        help='Validation dataset name (file name or file path)'
    )
    parser.add_argument(
        '--test-set', '-e',
        required=True,
        help='Test dataset name (file name or file path)'
    )
    parser.add_argument(
        '--lang',
        default='Korean',
        help='Language for fine-tuning, default: Korean'
    )
    parser.add_argument(
        '--task',
        default='transcribe',
        help='transcribe or translate, default: transcribe'
    )
    config = parser.parse_args()
    return config
    
class Trainer:
    '''Whisper finetune trainer'''
    
    def __init__(self, config) -> None:
        '''Init all required args for whisper finetune'''
        self.config = config
        
        # 사전 학습 모델 - 2개
        #   Base model -> tokenizer, feature_extractor, processor
        #   pre-trained model
        
        if config.pretrained_model:
            self.pretrained_model = config.pretrained_model
        else:
            print('\nPre-trained model is not given...')
            print(f'We will set pre-trained model same to --base-model (-b): {config.base_model}\n')
            self.pretrained_model = config.base_model
        
        self.output_dir = get_unique_directory(
            dir_name=config.output_dir,
            model_name=self.pretrained_model
        )
        self.finetuned_model_dir = get_unique_directory(
            dir_name=config.finetuned_model_dir,
            model_name=self.pretrained_model
        )
        print(f'\nTraining outputs will be saved -> {self.output_dir}')
        print(f'Fine-tuned model will be saved -> {self.finetuned_model_dir}')
        
        # Feature Extractor 등록
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.base_model
        )
        
        self.tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.base_model, 
            language=config.lang, 
            task=config.task
        )
        
        # Processor 등록
        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=config.base_model, 
            language=config.lang, 
            task=config.task
        )


    
    
    def load_dataset(self,) -> DatasetDict:
        '''Build dataset containing train/valid/test sets'''
        dataset = DatasetDict()
        dataset['train'] = load_dataset(
            path='csv',
            name='aihub-ko',
            split='train',
            data_files=self.config.train_set
        )
        dataset['valid'] = load_dataset(
            path='csv',
            name='aihub-ko',
            split='train', # valid로 바꾸면 안됨
            data_files=self.config.valid_set
        )
        dataset['test'] = load_dataset(
            path='csv',
            name='aihub-ko',
            split='train',
            data_files=self.config.test_set
        )
        return dataset
    
    def compute_metrics(self, pred) -> dict:
        '''Prepare evaluation metric (wer, cer, etc.)'''
    
    def prepare_dataset(self, batch):
         '''Get input features with numpy array & sentence labels'''
    
    def process_dataset(self, dataset) -> tuple:
        '''Process loaded dataset applying prepare_dataset()'''
    
    def enforce_fine_tune_lang(self) -> None:
        '''Enforce fine-tune language'''
    
    def create_trainer(self, train, valid) -> None:
        '''Create seq2seq trainer'''
    
    def run(self) -> None:        
        '''Run trainer'''
        

if __name__ == '__main__':
    config = get_config()
    trainer = Trainer(config)
    result = trainer.load_dataset()
    print(result)
    
    print(result['train'][0])
    input_str = result['train'][0]['sentence']
    print(f'\ninput_str: {input_str}')
    labels = trainer.tokenizer(input_str).input_ids
    print(f'\nlabels: {labels}')
    decoded_str_with_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=False)
    print(f'\ndecoded w/ special: \t {decoded_str_with_special_tokens}')
    decoded_str_without_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=True)
    print(f'\ndecoded w/o special: \t {decoded_str_without_special_tokens}')
    print(input_str == decoded_str_without_special_tokens)