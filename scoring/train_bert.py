import sys
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import softmax
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from scoring.scoring_utils import predictions_data_frame, scores_data_frame, save_data_frames, determine_avg_type, prepare_directories
import pandas as pd
import datetime
from data.data_utils.data_reader import AnswerReader


def lr_schedule(epoch, epochs):
    peak = epochs * 0.1
    if epoch <= peak:
        return epoch / peak
    else:
        return 1 - ((epoch - peak) / (epochs - peak))


def to_tensor_dataset(data_frame, index, input_column, target_column, label2id, tokenizer):
    indices = data_frame.index[index]
    sentences = data_frame.loc[indices, input_column].fillna('').to_list()
    labels = data_frame.loc[indices, target_column].map(label2id).to_list()

    inputs = tokenizer(sentences, return_tensors='pt', max_length=None, padding='max_length', truncation=True)

    return TensorDataset(inputs['input_ids'],
                         inputs['attention_mask'],
                         torch.tensor(labels),
                         torch.tensor(indices))


def generate_train_test_dataset(data_frame, train_idxs, test_idxs, input_col, target_col, label2id, tokenizer):
    train_set = to_tensor_dataset(data_frame, train_idxs, input_col, target_col, label2id, tokenizer)
    test_set = to_tensor_dataset(data_frame, test_idxs, input_col, target_col, label2id, tokenizer)
    return train_set, test_set


def load_bert_model(bert, device, labels, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(bert)
    model = AutoModelForSequenceClassification.from_pretrained(bert).to(device)

    if labels.size > 2:
        model.config.num_labels = labels.size
        model.classifier = Linear(in_features=model.config.hidden_size,
                                  out_features=labels.size,
                                  bias=True).to(device)

    return model, tokenizer


def predict(model, tokens, mask):
    return model(tokens, mask)[0]


def calculate_loss(model, tokens, mask, targets, loss_fn):
    return loss_fn(model(tokens, mask)[0], targets)


def predict_labels(model, tokens, mask):
    _, pred_labels = torch.max(softmax(model(tokens, mask)[0], 1), 1)
    return pred_labels


def compute_class_weights(label2id, y):
    n_samples = len(y)
    n_classes = len(label2id)
    return n_samples / (n_classes * (np.bincount(pd.Series(y).map(label2id)) + 1))


def train(target_path, df_train, df_test=None,
          model_name='bert-base-uncased',
          learning_rate=2e-5,
          batch_size=7,
          epochs=6,
          n_folds=10,
          save_model=False,
          random_seed=42,
          weight_decay=0.01,
          device=None):

    # Set the seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Device Detection Logic (if not explicitly provided)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        # Allows passing string 'cuda' or torch.device object
        device = torch.device(device)

    print(f"Using device: {device}")

    data_dir, out_dir = prepare_directories(target_path)
    input_col, target_cols = df_train.columns[1], [df_train.columns[2]]

    run_train_test = df_test is not None
    df_test_full = df_test

    # Use the passed model_name
    bert = model_name
    feature_name = model_name
    estimator_name = feature_name

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    for target_col in target_cols:
        labels = df_train[target_col].unique()
        labels.sort()
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        avg_type = determine_avg_type(labels)
        weigh_classes = np.all(df_train[target_col].value_counts() > 0)

        # Dataset splitting logic remains the same
        if run_train_test:
            df_test.columns = df_train.columns
            n_train, n_test = len(df_train), len(df_test)
            splits = [(np.arange(n_train), np.arange(n_train, n_train + n_test))]
            start_fold = -2
            df = pd.concat([df_train, df_test], keys=['train', 'test']).reset_index()
            dataset_name = f'train-test'
        else:
            df = df_train
            splits = cv.split(df[input_col], df[target_col])
            start_fold = 1
            dataset_name = f'test-cv'

        for i, idxs in enumerate(splits, start=start_fold):
            train_idx, test_idx = idxs

            # Loss Function Setup
            if weigh_classes:
                y = df.loc[df.index[train_idx], target_col].values
                class_weights = compute_class_weights(label2id, y).astype(np.float32)
                class_weights = torch.from_numpy(class_weights).to(device)
                loss_function = CrossEntropyLoss(class_weights)
            else:
                loss_function = CrossEntropyLoss()

            # Loading Model to Device
            model, tokenizer = load_bert_model(bert, device, labels, id2label, label2id)
            opt = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            train_set, test_set = generate_train_test_dataset(df, train_idx, test_idx,
                                                              input_col, target_col,
                                                              label2id, tokenizer)

            # Training Loop
            model.train()
            for epoch in range(epochs):
                print(f"{datetime.datetime.now()} | Fold {i} | Epoch {epoch + 1}/{epochs}")
                for j, batch in enumerate(DataLoader(train_set, batch_size=batch_size, shuffle=True)):
                    tokens, mask, targets, idx = (x.to(device) for x in batch)
                    opt.zero_grad()
                    loss = calculate_loss(model, tokens, mask, targets, loss_function)
                    loss.backward()
                    opt.step()

            # Evaluation Loop
            # TODO: refactoring in scoring_evaluation.py
            model.eval()
            with torch.no_grad():
                y_true, y_predicted = [], []
                for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False):
                    tokens, mask, targets, idx = (x.to(device) for x in batch)
                    pred_labels = predict_labels(model, tokens, mask)
                    y_true.extend(targets.cpu().tolist())
                    y_predicted.extend(pred_labels.cpu().tolist())

                # Scoring and Saving
                scores = scores_data_frame(y_true, y_predicted, dataset_name, estimator_name, feature_name, i, avg_type)
                df_test_full[target_col] = df_test_full[target_col].astype(int)
                df_test_full["predicted_score"] = y_predicted
                df_test_full["model"] = [model_name] * len(df_test_full)

                save_data_frames(out_dir / target_col, [df_test_full, scores], ['predictions.csv', 'scores.csv'])

            if run_train_test and save_model:
                output_model_name = f'{estimator_name}_train'
                model_dir = data_dir / 'models' / output_model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)

            # Important: explicitly move model to CPU and delete to clear VRAM
            model.to('cpu')
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()