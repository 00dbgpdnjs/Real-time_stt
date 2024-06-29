'''Fine-tuning whisper (possible: tiny, small, medium, etc.)
References:
    - Master reference -> https://huggingface.co/blog/fine-tune-whisper
    - Korean blog -> https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
'''

import argparse
from datasets import load_dataset, DatasetDict


def get_config() -> argparse.ArgumentParser:
    '''Wisper finetuning args parsing function'''
    
class Trainer:
    '''Whisper finetune trainer'''
    
    def __init__(self) -> None:
        '''Init all required args for whisper finetune'''
    
    def load_dataset(self,) -> DatasetDict:
        '''Build dataset containing train/valid/test sets'''
    
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
    trainer.run()