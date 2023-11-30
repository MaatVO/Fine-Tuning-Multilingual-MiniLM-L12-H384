import datasets
from numba import cuda
import pandas as pd
import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import create_optimizer
from datasets import Dataset
from tqdm import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
import torch.cuda
from typing import Tuple, Dict


# İlk hocanın paperinin yaptığının aynısını dene
# Sonra hugging face transformers kullan.
# Sonra sadece q1 değil q2 yi de eklemek gerekebilir.
# Transfer Learning, Lora, PEFT techniques

def csv_to_df(path: str = None, sample: int = 50) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.loc[df['outcome_class'] == 't', 'outcome_class'] = 'T'
    df.loc[df['outcome_class'] == 'd', 'outcome_class'] = 'F'
    df['q1'] = df['q1'].apply(lambda x: x.replace('\n', ''))
    df = df.rename(columns={'q1': 'sent', 'outcome_class': 'labels'})
    df = df[['sent', 'labels']]
    if sample != 0:
        return df.iloc[:sample, :]
    else:
        return df


def create_dataset_with_crossval(df: pd.DataFrame = None, seed: int = 42, cv: int = 2) \
        -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train = {f'split_{i + 1}': [] for i in range(cv)}
    test = {f'split_{i + 1}': [] for i in range(cv)}
    np.random.seed(seed)

    in_arr = np.arange(0, len(df))
    np.random.shuffle(in_arr)
    in_arr_r = in_arr.reshape((cv, int(len(df) / cv)))  # reshape data (num cv, total_rows/num cv)

    cv_ind = 0
    for k in train.keys():  # train.keys() = split_1, split_2 ...

        ind = in_arr_r[cv_ind, :]  # test indices
        antind = [i for i in in_arr if i not in ind]  # train indices

        intent_train = df[df.index.isin(antind)]
        intent_test = df[df.index.isin(ind)]

        train[k].append(intent_train)
        test[k].append(intent_test)

        train[k] = pd.concat(train[k])
        test[k] = pd.concat(test[k])

        cv_ind += 1

    # train,test: dicts with num cv splits, each split has len(data)*(1-1/cv), len(data)*1/cv
    return train, test


def preprocess_function(examples: datasets.Dataset) -> datasets.Dataset:
    inputs = [ex for ex in examples["sent"]]
    targets = [ex for ex in examples["labels"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, truncation=True)
    return model_inputs


seed = 42
np.random.seed(seed)
numcv = 10
intent_df = csv_to_df('data/sign_events_data_statements.csv')
train, test = create_dataset_with_crossval(df=intent_df, seed=seed, cv=numcv)

results = {}
preds = {}
trus = {}
collect_result = {}
model_size = 'small'
num_epochs = 3

for sp in tqdm(train.keys()):

    data_train = Dataset.from_pandas(train[sp])
    data_test = Dataset.from_pandas(test[sp])

    checkpoint = f"google/flan-t5-{model_size}"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    data_train = data_train.map(preprocess_function, batched=True)
    data_test = data_test.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_size}_scenario_intent",
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    pred = []
    for i in test[sp]['sent']:
        toki = tokenizer(i, return_tensors="pt").input_ids   # return token id's as pytorch tensor
        h = model.generate(toki, return_dict_in_generate=True, output_scores=True)
        decoded_preds = tokenizer.batch_decode(h.sequences, skip_special_tokens=True)
        pred.append(decoded_preds[0])

    pred = np.array(pred)

    preds[sp] = pred
    trus[sp] = np.array(test[sp]['labels'])

    collect_result[sp] = pd.DataFrame.from_dict(
        {"prediction": pred,
         'labels': np.array(test[sp]['labels'])})

    collect_result[sp]['corr'] = collect_result[sp]['prediction'] == collect_result[sp]['labels']
    results[sp] = [(pred == trus[sp]).mean()]
    print("collect results:",collect_result[sp].head(20))
    del model
    del trainer

results = pd.DataFrame.from_dict(results)
print('Average accuracy: ', results.mean(axis=1))
print('Max accuracy: ', results.max(axis=1))
