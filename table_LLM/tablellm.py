import os
import json
import torch

import typing as tp
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from table_LLM.tabledataset import TableDataset, TableLLMDataCollator
from table_LLM.table_trainer import TableLLMTrainer, CustomCallbackForTableLLM
from table_LLM import table_util

class TableLLM:
    """
        LLM for Tabular Data Regression & Classification Task. 
    """

    def __init__(
        self,
        model_name: str,
        device_map: tp.Dict[str, int] = None,
        new_tokens: tp.List[str] = None,
        columns: tp.List[str] = None,
        output_hidden_states: bool = False,
    ):
        
        self.model_name = model_name
        self.new_tokens = new_tokens
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.columns = columns
    
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, attn_implementation="eager", torch_dtype=torch.bfloat16, output_hidden_states=output_hidden_states)

        if self.new_tokens is not None:
            self.tokenizer.add_tokens(self.new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {len(self.new_tokens)} tokens to the tokenizer.")
            print(f"Resized token embeddings to {len(self.tokenizer)}.")

    def train(
        self,
        train_data: TableDataset,
        #eval_data: list,
        save_dir: str,
        training_args: TrainingArguments,
        compute_metrics=None,
    ):  
        print(train_data)
        #print(eval_data)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        trainer = TableLLMTrainer(
            column_names=self.columns,
            result_column="result",
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_data,
            #eval_dataset=eval_data[0],
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=TableLLMDataCollator(tokenizer=self.tokenizer),
            #callbacks=[CustomCallbackForTableLLM(tablellm=self, eval_dataframe=eval_data[1], save_dir=save_dir)],
        )

        trainer.train()
        
        return trainer
    def save(self, path: str):
        """Save GReaT Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        os.makedirs(path, exist_ok=True)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")
            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    @classmethod
    def load_from_dir(cls, path: str):
        """Load GReaT class

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)
            print(attributes)

        # Create new be_great model instance
        great = cls(attributes["model_name"], new_tokens=attributes["new_tokens"], device_map=attributes["device_map"], columns=attributes["columns"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great
    @classmethod
    def load_from_dir_debug(cls, path: str):
        """Load GReaT class

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)
            print(attributes)

        # Create new be_great model instance
        great = cls(attributes["model_name"], new_tokens=attributes["new_tokens"], device_map=attributes["device_map"], columns=attributes["columns"], output_hidden_states=True)

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great
    
    def impute(self, data: pd.DataFrame, max_retries: int = 10, **kwargs) -> pd.DataFrame:
        """
        Generate imputations for missing values in a DataFrame.
        """
        # Copy input data to avoid modifying original
        prompt_list = []
        for index, row in data.iterrows():
            prompt, _ = table_util.transform_row_to_sentense_except_nan(row)
            prompt += ", y is"
            prompt_list.append(prompt)
        imputed_data = self.generate(prompt_list, **kwargs)
        return imputed_data
    
    def generate(self,
        starting_prompts: tp.Union[str, list[str]],
        device: str = "cuda",
        batch_size: int = 128,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate new data from the model.
        """
        if isinstance(starting_prompts, str):
            starting_prompts = [starting_prompts]
        generated_data = []
        temp_data = []
        for i in range(0, len(starting_prompts), batch_size):
            batch = starting_prompts[i : i + batch_size]
            batch = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            temp_data.extend(self.model.generate(
                **batch,
                max_length=100,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            ))
        temp_data = self.tokenizer.batch_decode(temp_data, skip_special_tokens=True)
        for text in temp_data:
            generated_row = table_util.transform_sentense_to_row(text)
            generated_data.append(generated_row)
        generated_data = pd.DataFrame(generated_data)

        return generated_data
    

    def _check_integrity(self, generated_data: str):
        """
        Check the integrity of the generated sentences.
        """
        

        
        
