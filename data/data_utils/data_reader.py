import pandas as pd
import json


class AnswerReader:
    def __init__(self, file_path, dataset_name):
        self.file_path = file_path
        self.dataset_name = dataset_name

    def to_dataframe(self):
        fp = self.file_path
        name = self.dataset_name
        try:
            if name == 'ASAP':
                df = pd.read_csv(fp).drop_duplicates().dropna()
                return df
            elif name == 'alice':
                with open(fp, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                df = pd.DataFrame(data, columns=['id', 'answer', 'score'])
                return df
            else:
                raise ValueError(f"Unknown dataset: {name}")
        except Exception as e:
            return f"Error loading data: {e}"

