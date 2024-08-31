import os
import warnings
import json
import logging
import re
import torch
import time
import random
import string

import typing as tp
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from transformers import TrainingArguments
from table_LLM.tabledataset import TableDataset
from table_LLM.tablellm import TableLLM
from table_LLM.upload_notion import NotionDatabase
from table_LLM.table_util import transform_result, transform_result_back



#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
warnings.filterwarnings('ignore')
np.random.seed(26415)
DATABASE_ID = "03183227fd634b679cecfaed4c889f2f"
NOTION_KEY = "secret_2nVjIaYGdiJJbz7VpKwF0kqsdbZyqgjgLHPQWfyEXzF"
NOTION_DB = NotionDatabase(DATABASE_ID, NOTION_KEY)





def main(INIT, TRAINDATAPATH, TRAINDATASET, TESTDATAPATH, TESTDATASET, WORDLIST, RESULTDICT, BATCH_SIZE, EPOCHS, MODEL, MODE):
    print("############################################################################################################")
    print(f"INIT: {INIT}.")
    print(f"TRAINDATAPATH: {TRAINDATAPATH}.")
    print(f"TRAINDATASET: {TRAINDATASET}.")
    print(f"TESTDATAPATH: {TESTDATAPATH}.")
    print(f"TESTDATASET: {TESTDATASET}.")
    print(f"WORDLIST: {WORDLIST}.")
    print(f"RESULTDICT: {RESULTDICT}.")
    print(f"BATCH_SIZE: {BATCH_SIZE}.")
    print(f"EPOCHS: {EPOCHS}.")
    print(f"MODEL: {MODEL}.")
    print(f"MODE: {MODE}.")
    print("############################################################################################################")
    ############################################################################################################
    # New Token
    ###########################################################################################################

    result_list = []
    if MODE == "classification":
        result_dict = json.load(open(RESULTDICT, "r"))
        for key, value in result_dict.items():
            result_list.append(value)
    ############################################################################################################
    ############################################################################################################
    # init
    ###########################################################################################################
    data = pd.read_csv(TRAINDATAPATH)
    if MODE == "classification":
        data['result'] = data['result'].round().astype(int)
        data['result'] = data['result'].apply(lambda x: result_dict[str(x)])
    
    if MODE == "TYPE1":
        data['result'] = [transform_result(x) for x in data['result']]


    ############################################################################################################
    os.makedirs(f"/home/hyun/paper/log/{INIT}", exist_ok=True)
    os.chdir(f"/home/hyun/paper/log/{INIT}")  
    save_dir = f"/home/hyun/paper/log/{INIT}/model"

    # save new token list

    #model = TableLLM.load_from_dir("/home/hyun/bob/table_LLM/base_model_resultX1")
    model = TableLLM(model_name=MODEL, device_map='auto', new_tokens=result_list, columns=["process1", "process2", "process3", "process4", "process5", "process6"])

    # train_data, eval_data = train_test_split(data, test_size=0.2, random_state=26415)
    # eval_data_for_metric = eval_data.copy()
    train_data = TableDataset.from_pandas(data, preserve_index=False)
    train_data.set_tokenizer(model.tokenizer)
    train_data.set_generate_mode("train")
    # eval_data = TableDataset.from_pandas(eval_data, preserve_index=False)
    # eval_data.set_tokenizer(model.tokenizer)
    # eval_data.set_generate_mode("train")
    for i in range(10):
        print(train_data.get_sample(i))

    training_args = TrainingArguments(
        seed=26415,
        data_seed=26415,
        output_dir="experiment",

        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        #label_smoothing_factor=0.1,
        save_strategy="steps", #저장하지 않음
        save_steps=500,
        save_total_limit=1, #최대 1개까지 저장
        # eval_steps=500, #학습하는 동안 총 40회 평가
        # eval_delay=500, #시작 후 300스텝 후에 평가 시작
        # eval_strategy="steps", 
        # load_best_model_at_end=True,
        save_only_model=True,
        logging_steps=100,
        tf32=True,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    
    model.train(train_data=train_data, save_dir=save_dir, training_args=training_args)
    model.save(save_dir)
    #model.load_from_dir("/home/hyun/paper/log/20240824090902/model")
    ############################################################################################################
    # Test
    ############################################################################################################
    test_data_load = pd.read_csv(TESTDATAPATH)
    result = test_data_load['result'].copy()
    if MODE == "classification":
        result_list = []
        result_dict = json.load(open(RESULTDICT, "r"))
        result_dict = {str(value): key for key, value in result_dict.items()}
    # 성능 측정을 위한 리스트 초기화
    test_data_load['result'] = np.nan
    mae_list, mse_list, rmse_list, r2_list = [], [], [], []
    for _ in range(10):
        impute_result = model.impute(
            test_data_load, 
            temperature=0.7, 
            num_beams=1, 
            do_sample=True,
            top_k=12,
            top_p=0.6,
            min_p=0.5,
        )
        print(impute_result)
        data_true = result.tolist()
        data_predict = impute_result['result'].tolist()
        print(data_predict)
        if MODE == "classification":
            data_predict = [result_dict[str(x)] for x in data_predict]
        if MODE == "TYPE1":
            data_predict = [transform_result_back(x) for x in data_predict]

        data_true = list(map(float, data_true))
        data_predict = list(map(float, data_predict))

        # MAE, MSE, RMSE 및 R² 계산
        mae = mean_absolute_error(data_true, data_predict)
        mse = mean_squared_error(data_true, data_predict)
        rmse = root_mean_squared_error(data_true, data_predict)
        r_2 = r2_score(data_true, data_predict)
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r_2)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r_2}")
    mae, mse, rmse, r2 = round(np.mean(mae_list), 4), round(np.mean(mse_list), 4), round(np.mean(rmse_list), 4), round(np.mean(r2_list), 4)
    page_value = {
    'Batch Size': str(BATCH_SIZE),
    'TrainDataset': TRAINDATASET,
    'TestDataset': TESTDATASET,
    'Epoch': EPOCHS,
    'MAE': mae,
    'MSE': mse,
    'Model': MODEL,
    'RMSE': rmse,
    'R²': r2,
    '테스트 일시': NOTION_DB.transform_date(INIT),
    'Result Type' : MODE,
    '메모': "paper - gpt2, regression, 1000개, eval 미적용 "
    }
    NOTION_DB.upload_page_values(page_value)

if __name__ == "__main__":
    for gpu, epochs in enumerate([25, 50, 75, 100, 125, 150, 175, 200]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for batch_size in [2, 4, 8, 16, 32, 64]:
            for MODEL in ["gpt2"]:
                print(f"GPU: {gpu}, EPOCHS: {epochs}, BATCH_SIZE: {batch_size}, MODEL: {MODEL}")
                if gpu != 7:
                    continue
                for traindatapath, testdatapath in [("/home/hyun/paper/dataset/train_data_1000.csv", "/home/hyun/paper/dataset/test_data_125.csv")]:
                    INIT = (datetime.now() + timedelta(hours=9)).strftime('%Y%m%d%H%M%S')
                    TRAINDATAPATH = traindatapath
                    TRAINDATASET = TRAINDATAPATH.split("/")[-1].split(".")[0]
                    TESTDATAPATH = testdatapath
                    TESTDATASET = TESTDATAPATH.split("/")[-1].split(".")[0]
                    WORDLIST = None
                    RESULTDICT = "/home/hyun/paper/dataset/new_token_list.json"
                    BATCH_SIZE = batch_size
                    EPOCHS = epochs
                    MODEL = MODEL
                    MODE = "TYPE1"
                            
                    main(INIT, TRAINDATAPATH, TRAINDATASET, TESTDATAPATH, TESTDATASET, WORDLIST, RESULTDICT, BATCH_SIZE, EPOCHS, MODEL, MODE)
                    torch.cuda.empty_cache()

