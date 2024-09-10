import random
import typing as tp
import numpy as np

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
    

class TableDataset_v2(Dataset):
    def _init_structure(self):
        self.structure = STRUCTURE
        self.key = STRUCTURE.split(CONNECTOR)[0]
        self.value = STRUCTURE.split(CONNECTOR)[1]
        self.connector = CONNECTOR
        self.delimiter = DELIMITER
        self.eos_token = self.tokenizer.eos_token
        self.pad_token_idx = self.tokenizer.pad_token_idx


    def get_data(self):
        return self._data
    
    def set_length(self, max_length: int):
        self.max_length = max_length


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self._init_structure()


    def set_generate_mode(self, mode: str):
        self.generate_mode = mode


    def _getitem(self, key: tp.Union[int, slice, str]) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)
        idx = list(range(row.num_columns))
        
        random.shuffle(idx)
        text = DELIMITER.join(
        [
            f"{row.column_names[i]}{self.connector}{str(row.columns[i].to_pylist()[0]).strip()}"
            for i in idx
        ]
        ) + self.eos_token
        
        tokenized_text = self.tokenizer(text)
        tokenized_text["input_ids"] = np.pad(
            tokenized_text["input_ids"],
            (0, self.max_length - len(tokenized_text["input_ids"])),
            mode="constant",
            constant_values=self.pad_token_idx,
        )
        tokenized_text["token_type"] = np.pad(
            tokenized_text["token_type"],
            (0, self.max_length - len(tokenized_text["token_type"])),
            mode="constant",
            constant_values=0,
        )
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)
        
    def get_sample(self, key: tp.Union[int, slice, str]):
        return self.tokenizer.decode(self._getitem(key)["input_ids"])
    
    

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
