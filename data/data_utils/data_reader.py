import pandas as pd
import json


class AnswerReader:
    def __init__(self, file_path, dataset_name, experiment_setting=None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.experiment_setting = experiment_setting


    def load_alice(self, fp, *, with_question=False, with_rubric=False, with_both=False):
        with open(fp, 'r', encoding='utf-8') as file:
            data = json.load(file)

        df = pd.DataFrame(data)

        def combine_content(row):
            include_q = with_question or with_both
            include_r = with_rubric or with_both

            parts = []
            if include_q and 'question' in row:
                parts.append(f"Question: {row['question']}")
            if include_r and 'rubric' in row:
                parts.append(f"Rubric: {row['rubric']}")

            parts.append(f"Answer: {row['answer']}")
            return "\n\n".join(parts)

        df['answer'] = df.apply(combine_content, axis=1)

        # Handle score mapping
        if '2way' in fp:
            df['score'] = df['score'].map({'Correct': 1, 'Incorrect': 0})
        else:  # '3way'
            df['score'] = df['score'].map({'Correct': 1, 'Incorrect': 0, 'Partially correct': 0.5})

        return df[['id', 'answer', 'score']]
        

    def to_dataframe(self):
        fp = self.file_path
        name = self.dataset_name
        setting = self.experiment_setting
        try:
            if name == 'ASAP':
                df = pd.read_csv(fp).drop_duplicates().dropna()
                return df
            elif name == 'alice':
                kwargs = {
                    'with_question': setting == 'with_question',
                    'with_rubric': setting == 'with_rubric',
                    'with_both': setting == 'with_both'
                }
                return self.load_alice(fp, **kwargs)
            else:
                raise ValueError(f"Unknown dataset: {name}")
        except Exception as e:
            return f"Error loading data: {e}"
