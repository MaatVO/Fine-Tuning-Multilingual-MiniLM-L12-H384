import pandas as pd
import numpy as np
import os
from transformers import  (AutoModel,AutoModelForSequenceClassification,
                           DataCollatorWithPadding,AutoTokenizer,TrainingArguments,Trainer)
from datasets import Dataset,DatasetDict
import evaluate
accuracy = evaluate.load("accuracy")
import torch
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def csv_to_df(path: str = None,question_type:int=1, sample: int = 100) -> pd.DataFrame:
    """

    :param path: "csv file path
    :param question_type: for q1 ->1, for q2->2, for q1+q2->3
    :param sample: 0 is full dataset, anything else is number of rows
    :return: pandas dataframe
    """
    df = pd.read_csv(path, encoding="ISO-8859-1")

    if sample != 0:
        df = df.sample(n=sample, replace='False')


    df.loc[df['outcome_class'] == 't', 'outcome_class'] = 1
    df.loc[df['outcome_class'] == 'd', 'outcome_class'] = 0
    if question_type == 1:
        df['q'] = df['q1'].apply(lambda x: x.replace('\n', ''))
    elif question_type ==2:
        df['q'] = df['q2'].apply(lambda x: x.replace('\n', ''))
    elif question_type == 3:
        df['q1'] = df['q1'].apply(lambda x: x.replace('\n', ''))
        df['q2'] = df['q2'].apply(lambda x: x.replace('\n', ''))
        df['q'] = df['q1'] + df['q2']

    df = df.rename(columns={'q': 'sent', 'outcome_class': 'label'})
    df = df[['sent', 'label']]
    return df
def tokenize_function(examples: Dataset) -> Dataset:
    model_inputs = tokenizer(examples["sent"],truncation=True)
    return model_inputs

seed = 42
np.random.seed(seed)
data_path = 'data/sign_events_data_statements.csv'
question_type = 3
data_sample_size = 800
intent_df = csv_to_df(path=data_path,
                      question_type=question_type,
                      sample = data_sample_size)

dataset = Dataset.from_pandas(intent_df)
train_testvalid = dataset.train_test_split(test_size= 0.1, shuffle=True)
valid_test = train_testvalid["test"].train_test_split(test_size=0.5,shuffle=True)
dataset_splitted = DatasetDict({
    'train': train_testvalid["train"],
    'valid': valid_test["train"],
    'test': valid_test["test"]
})

checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
path = os.path.join("./models/pretrained",checkpoint)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)
id2label = {0: "F", 1: "T"}
label2id = {"F": 0, "T": 1}

model = AutoModelForSequenceClassification.from_pretrained(
path, num_labels=2, id2label=id2label, label2id=label2id
                                   )
tokenized_dataset = dataset_splitted.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

for param in model.bert.encoder.layer[:3].parameters():  # Freeze the first 3 layers of the encoder
    param.requires_grad = False


num_epochs = 10
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu= False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#trainer.train()

# Prediction
predictions = []
labels = []
for example in tokenized_dataset["test"]:
    # Tokenize input
    inputs = tokenizer(example['sent'], return_tensors='pt')

    # Run inference
    outputs = model(**inputs)

    # Get predictions and labels
    prediction = torch.argmax(outputs.logits).item()
    label = example['label']  # Replace 'label' with the actual key in your dataset

    predictions.append(prediction)
    labels.append(label)

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')