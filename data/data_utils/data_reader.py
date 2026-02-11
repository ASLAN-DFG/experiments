import pandas as pd
import json
import re


class AnswerReader:
    def __init__(self, file_path, dataset_name, experiment_setting=None, question_id=None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.experiment_setting = experiment_setting
        self.question_id = question_id

    def load_alice(self, fp, *, with_question=False, with_rubric=False, with_both=False, question_id=None):
        with open(fp, 'r', encoding='utf-8') as file:
            data = json.load(file)

        df = pd.DataFrame(data)

        if question_id:
            if len(question_id)>2:
                df = df.loc[df['question_id'] == question_id]
                print(str(len(df))+' answers to the question ' + question_id + ' are filtered!')
            else:
                df_1 = df.loc[df['question_id'] == question_id[0]]
                df_2 = df.loc[df['question_id'] == question_id[1]]
                df = pd.concat([df_1, df_2], ignore_index=True)
                print(str(len(df))+' answers to the question ' + ','.join(question_id) + ' are filtered!')

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
        try:
            if self.dataset_name == 'ASAP':
                df = pd.read_csv(self.file_path).drop_duplicates().dropna()
                return df
            elif self.dataset_name == 'alice':
                kwargs = {
                    'with_question': self.experiment_setting == 'with_question',
                    'with_rubric': self.experiment_setting == 'with_rubric',
                    'with_both': self.experiment_setting == 'with_both',
                    'question_id': self.question_id
                }
                return self.load_alice(self.file_path, **kwargs)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        except Exception as e:
            return f"Error loading data: {e}"
