import pandas as pd
import random
import typing as tp
import pandas as pd

STRUCTURE = "KEY is VALUE"
DELIMITER = ", "
CONNECTOR = " is "

def get_unique_word(df: pd.DataFrame, column: list):
    """
    Get unique words from the given column
    """
    word_list = []
    
    for col in column:
        word_list.extend(df[col].unique())
    column.append("result is ")
    word_list.extend(column)
    return list(set(word_list))
    
def transform_row_to_sentense(row: pd.Series, impute: bool = False) -> str:
    index_range = list(range(len(row)))
    if impute:
        random.shuffle(index_range)
    text = DELIMITER.join(
        [
            f"{row.index[i]}{CONNECTOR}{str(row.iloc[i]).strip()}"
            for i in index_range
        ]
    )
    return text

def transform_row_to_sentense_except_nan(row: pd.Series, impute: bool = False) -> str:
    index_range = list(range(len(row)))
    if impute:
        random.shuffle(index_range)
    text = DELIMITER.join(
        [
            f"{row.index[i]}{CONNECTOR}{str(row.iloc[i]).strip()}"
            for i in index_range
            if not pd.isnull(row.iloc[i])
        ]
    )
    remain_columns = [
        row.index[i] for i in index_range if pd.isnull(row.iloc[i])
    ]
    return text, remain_columns


def transform_sentense_to_row(text: str) -> pd.Series:
    pairs = [pair.strip() for pair in text.split(DELIMITER)]
    data = {}
    for pair in pairs:
        key, value = [item.strip() for item in pair.split(CONNECTOR, 1)]
        data[key] = value
    return pd.Series(data)

def transform_result(result):
    result = str(result)
    result = result.ljust(5, "0")
    result = result.replace(".", "")
    result = "_".join(result)
    return result

def transform_result_back(result):
    result = result.replace("_", "")
    result = result[:2] + "." + result[2:]
    return result



