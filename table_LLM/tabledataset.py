import random
import typing as tp
import numpy as np
import torch

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from .table_util import DELIMITER, STRUCTURE, CONNECTOR

class TableDataset(Dataset):
    def _init_structure(self):
        self.structure = STRUCTURE
        self.key = STRUCTURE.split(CONNECTOR)[0]
        self.value = STRUCTURE.split(CONNECTOR)[1]
        self.connector = CONNECTOR
        self.delimiter = DELIMITER
        self.eos_token = self.tokenizer.eos_token

    def get_data(self):
        return self._data


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_structure()


    def set_generate_mode(self, mode: str):
        self.generate_mode = mode


    def _getitem(self, key: tp.Union[int, slice, str]) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)
        row_df = row.to_pandas()
        column_names = row_df.columns.tolist()
        values = row_df.values.tolist()[0]

        if self.generate_mode == "train":
            idx = random.sample(range(len(column_names)), len(column_names) - 1)
        text = DELIMITER.join(
        [
            f"{column_names[i]}{self.connector}{str(values[i]).strip()}"
            for i in idx
        ]
        )
        print(text)
        text += self.eos_token
        
        tokenized_text = self.tokenizer(text, padding=True)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)
        
    def get_sample(self, key: tp.Union[int, slice, str]):
        return self.tokenizer.decode(self._getitem(key)["input_ids"], skip_special_tokens=False)
    

class TableDataset_V2(Dataset):
    def _init_structure(self):
        self.structure = STRUCTURE
        self.key = STRUCTURE.split(CONNECTOR)[0]
        self.value = STRUCTURE.split(CONNECTOR)[1]
        self.connector = CONNECTOR
        self.delimiter = DELIMITER
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.generate_mode = "train"

    def get_data(self):
        return self._data


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_structure()


    def set_generate_mode(self, mode: str):
        self.generate_mode = mode
    def custom_encode(self, text):
        # Encode the text using the tokenizer

        # 토큰 타입 정의
        TOKEN_TYPES = {
            'NUMBER': 0,
            'STRING': 1,
            'PUNCTUATION': 2,
            'SPECIAL': 3
        }
        encoding = self.tokenizer.encode(text)
        
        # Get the tokens
        tokens = encoding.tokens
        
        # Initialize token_type_ids
        token_type_ids = []
        number_words = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
        special_tokens = ["<EOS>", "<BOS>", "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
        string_words= ["is", ",", " ", "<num>"]
        for token in tokens:
            if token in special_tokens:
                token_type_ids.append(TOKEN_TYPES['SPECIAL'])
            elif token in number_words:
                token_type_ids.append(TOKEN_TYPES['NUMBER'])
            elif token in string_words:
                token_type_ids.append(TOKEN_TYPES['STRING'])
            elif token in [',']:
                token_type_ids.append(TOKEN_TYPES['PUNCTUATION'])
            else:
                token_type_ids.append(TOKEN_TYPES['STRING'])  # Default to STRING for other tokens
        
        # Add token_type_ids to the encoding
        encoding.token_type_ids = token_type_ids
        
        return encoding


    def _getitem(self, key: tp.Union[int, slice, str]) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)
        row_df = row.to_pandas()
        column_names = row_df.columns.tolist()
        values = row_df.values.tolist()[0]

        if self.generate_mode == "train":
            idx = random.sample(range(len(column_names)), len(column_names) - 1)
        text = DELIMITER.join(
        [
            f"{column_names[i]}{self.connector}{str(values[i]).strip()}"
            for i in idx
        ]
        )
        text += self.eos_token
        
        tokenized_text = self.custom_encode(text)
        print(len(tokenized_text.ids))
        tokenized_text = {
            "input_ids": tokenized_text.ids,
            "attention_mask": tokenized_text.attention_mask,
            "token_type_ids": tokenized_text.token_type_ids
        }
        #pad
        if len(tokenized_text["input_ids"]) < 44:
            tokenized_text["input_ids"] += [self.pad_token_id] * (44 - len(tokenized_text["input_ids"]))
            tokenized_text["attention_mask"] += [0] * (44 - len(tokenized_text["attention_mask"]))
            tokenized_text["token_type_ids"] += [3] * (44 - len(tokenized_text["token_type_ids"]))

        #torch.tensor
        tokenized_text["input_ids"] = torch.tensor(tokenized_text["input_ids"])
        tokenized_text["attention_mask"] = torch.tensor(tokenized_text["attention_mask"])
        tokenized_text["token_type_ids"] = torch.tensor(tokenized_text["token_type_ids"])
        

        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)
        
    def get_sample(self, key: tp.Union[int, slice, str]):
        return self.tokenizer.decode(self._getitem(key)["input_ids"], skip_special_tokens=False)
    
    

@dataclass
class TableLLMDataCollator(DataCollatorWithPadding):
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
