# Save CSV file in a separate file using the following format
# ======================================
# text,label
# "The government passed a new policy on healthcare",neutral
# "Taxes should be lowered to boost the economy",right
# "We need more social programs to help the poor",left
# "Strongly supports the right wing policies",right
# "Strongly supports the left wing policies",left
# "The prime minister addressed the nation",neutral
# "The government passed a new policy on healthcare",neutral
# "Strongly supports the socialist policies",left
# "Conservatives vent only irritable mental gestures which seek to resemble ideas",left
# "The leader of Glasgow City Council has condemned UK asylum policy as a machine that creates homeless refugees",left
# "This Government is much more comfortable with large multi-nationals than smaller enterprises that are the building blocks of our prosperity",right
# "Thousands illegally claiming child benefit while living abroad",right
# "The Department of Class Solidarity arms organizers, activists, and everyday people with the tools to expose the billionaires ransacking our democracy. Welcome to the war room of the working class",left
# "Voters who are sympathetic to Corbyn's party could hold their nose and back Labour to stop a Reform candidate winning in their area",left
# "The current policy is trampling on the rights every other girl and her rights to privacy and protected spaces",right
# ======================================

import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import os
import pandas as pd

# load training data from CSV and replace with full file path including file name
csv_file = "training_data.csv"
df = pd.read_csv(csv_file)

# convert CSV to spaCy format
train_data = [
    (row['text'], {"cats": {
        "neutral": 1.0 if row['label'] == "neutral" else 0.0,
        "right": 1.0 if row['label'] == "right" else 0.0,
        "left": 1.0 if row['label'] == "left" else 0.0
    }})
    for idx, row in df.iterrows()
]

# split into train/test
split = int(0.8 * len(train_data))
train_data, test_data = train_data[:split], train_data[split:]

# create blank model with English tokenizer / pipeline
nlp = spacy.blank("en")

# add TextCategoriser to the pipeline with default config
textcat = nlp.add_pipe("textcat")

# add labels
for label in ["neutral", "right", "left"]:
    textcat.add_label(label)

# train the model
optimizer = nlp.begin_training()
for i in range(30):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=8)
    for batch in batches:
        examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Iteration {i}, Losses: {losses}")

# save the trained model
output_dir = "textcat_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# reload the model
nlp2 = spacy.load(output_dir)

# evaluate on test_data
y_true, y_pred = [], []
for text, annotations in test_data:
    doc = nlp2(text)
    true_label = max(annotations["cats"], key=annotations["cats"].get)
    pred_label = max(doc.cats, key=doc.cats.get)
    y_true.append(true_label)
    y_pred.append(pred_label)
    print(f"Text: {text}\n  Predicted: {pred_label} ({doc.cats})\n  True: {true_label}\n")

# print
print("\nEvaluation Results")
print("------------------")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score (macro):", f1_score(y_true, y_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# test prediction with reloaded model
test_text = "The government policies are right leaning"
doc = nlp2(test_text)
pred_label = max(doc.cats, key=doc.cats.get)
print(f"Test text: {test_text}")
print(f"Predicted label: {pred_label}")
print(f"All category scores: {doc.cats}")
