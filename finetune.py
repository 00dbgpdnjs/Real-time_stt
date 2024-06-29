'''Fine-tuning whisper (possible: tiny, small, medium, etc.)
References:
    - Master reference -> https://huggingface.co/blog/fine-tune-whisper
    - Korean blog -> https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
'''

import argparse
from datasets import load_dataset, DatasetDict


def get_config() -> argparse.ArgumentParser:
    '''Wisper finetuning args parsing function'''
    parser = argparse.ArgumentParser()
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
    config = parser.parse_args()
    return config
    
class Trainer:
    '''Whisper finetune trainer'''
    
    def __init__(self, config) -> None:
        '''Init all required args for whisper finetune'''
        self.config = config
    
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