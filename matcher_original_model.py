# cls ; python matcher_original_model.py

from torch.utils import data
from transformers import AutoTokenizer
from datetime import datetime
from ditto_light.summarize import Summarizer
from ditto_light.ditto import evaluate, DittoModel
from ditto_light.dataset import DittoDataset
import json
import os
import pandas as pd
import sklearn.metrics as metrics
import torch

lm = "roberta"
# TASK_NAME = "Textual/Abt-Buy"
TASK_NAME = "Structured/Amazon-Google"
# TASK_NAME = "Structured/Walmart-Amazon"
files = os.listdir(rf"checkpoints\{TASK_NAME}")
model_file_path = os.path.join(rf"checkpoints\{TASK_NAME}", files[0])
threshold = float(model_file_path.split('-')[-1].replace('.pt',''))
dataset_file_path = f"data/er_magellan/{TASK_NAME}"
trainset_path = f"{dataset_file_path}/train.txt"
testset_path = f"{dataset_file_path}/test.txt"


def load_testset():

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[TASK_NAME]

    summarizer = Summarizer(config, lm=lm)

    trainset_summarizer = summarizer.transform_file(trainset_path, max_len=64)
    testset_summarizer = summarizer.transform_file(testset_path, max_len=64)

    train_dataset = DittoDataset(
        trainset_summarizer,
        lm=lm,
        max_len=64,
        size=None,
        da="drop_col",
    )
    test_dataset = DittoDataset(testset_summarizer, lm=lm)
    padder = train_dataset.pad
    test_iter = data.DataLoader(
        dataset=test_dataset,
        batch_size=int(len(test_dataset)/4),
        shuffle=False,
        num_workers=0,
        collate_fn=padder,
    )

    test_len = len(test_dataset)

    return test_iter, test_len


test_iter, test_len = load_testset()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DittoModel(device=device, lm=lm)
saved_state = torch.load(model_file_path, map_location=device)
model.load_state_dict(saved_state['model'])
model.to(device)


def model_evaluate(model, test_iter, threshold):
    all_p = []
    y_true = []
    all_probs = []

    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            y_true += y.cpu().numpy().tolist()

    y_pred = [1 if p > threshold else 0 for p in all_probs]

    print()
    print(metrics.classification_report(y_true, y_pred, zero_division=0, digits=4))

    return y_true, y_pred, all_probs

test_start_time = datetime.now()
y_true, y_pred, all_probs = model_evaluate(
    model,
    test_iter,
    threshold,
)
test_end_time = datetime.now()

def create_output(y_true, y_pred, all_probs, testset_path):
    lines = open(testset_path)
    df = []
    for line in lines:
        s1, s2, label = line.strip().split('\t')
        df.append({'left': s1, 'right': s2, 'label': int(label)})

    df = pd.DataFrame(df)
    df['match'] = y_pred
    df['match_confidence'] = all_probs

    output_filename = testset_path.split('/')[-2]

    df.to_json(f'output/{output_filename}-output.jsonl', orient='records', lines=True)

create_output(y_true, y_pred, all_probs, testset_path)

runtime_filepath = r'C:\Users\rodolfo\Documents\llm-tests\runtime-executions.csv'
df_runtime = pd.read_csv(runtime_filepath)
runtime_row = {
    'datetime': datetime.now(),
    'model': 'ditto',
    'dataset_name': TASK_NAME.split('/')[-1].lower(),
    'dataset_rows': test_len,
    'step': 'test',
    'runtime': test_end_time-test_start_time,
    'runtime_per_sample': (test_end_time-test_start_time)/(test_len),
}
df_runtime.loc[len(df_runtime)] = runtime_row
df_runtime.to_csv(runtime_filepath, index=False)