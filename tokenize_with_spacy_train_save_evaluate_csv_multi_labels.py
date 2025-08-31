#The training data with it's labels should be saved in a separate CSV file:
#text,label
#"The government passed a new policy on healthcare","neutral,politics"
#"Taxes should be lowered to boost the economy","right,economy"
#"We need more social programs to help the poor","left,welfare"
#"Strongly supports the right wing policies","right,politics"
#"Strongly supports the left wing policies","left,politics"
#"The prime minister addressed the nation","neutral,politics"
#"The government passed a new policy on healthcare","neutral,politics"
#"The Government is much more comfortable with large multi-nationals than smaller enterprises that are the building blocks of our prosperity","right,economy"
#"Thousands are illegally claiming child benefit while living abroad","right,welfare"
#"Conservatives vent only irritable mental gestures which seek to resemble ideas","left,politics"
#"The leader of Glasgow City Council has condemned UK asylum policy as a machine that creates homeless refugees","left,politics"
#--------------------------------

import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, jaccard_score
import numpy as np
import random
import os
import pandas as pd

# load training data from CSV and replace with full file path including file name
csv_file = "training_data.csv"
df = pd.read_csv(csv_file)

# extract all unique labels from CSV
all_labels = set()
for labels in df["label"]:
    for lbl in str(labels).split(","):
        all_labels.add(lbl.strip())
all_labels = sorted(all_labels)
print("Detected labels:", all_labels)

# convert CSV to spaCy format
train_data = []
for idx, row in df.iterrows():
    cats = {label: 0.0 for label in all_labels}
    for lbl in str(row["label"]).split(","):
        lbl = lbl.strip()
        if lbl in cats:
            cats[lbl] = 1.0
    train_data.append((row["text"], {"cats": cats}))

# split into train/test
split = int(0.8 * len(train_data))
train_data, test_data = train_data[:split], train_data[split:]

# create blank model with English tokenizer / pipeline
nlp = spacy.blank("en")

# add TextCategoriser (multilabel) to the pipeline
textcat = nlp.add_pipe(
    "textcat_multilabel",
    config={
        "threshold": 0.5, # classification threshold
        "model": {
            "@architectures": "spacy.TextCatEnsemble.v2",
            "tok2vec": {"@architectures": "spacy.Tok2Vec.v2"},
        },
    },
)

# add labels
for label in all_labels:
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
output_dir = "textcat_multilabel_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# reload the model
nlp2 = spacy.load(output_dir)

# evaluate on test_data
y_true, y_pred = [], []
for text, annotations in test_data:
    doc = nlp2(text)
    true_labels = [lbl for lbl, val in annotations["cats"].items() if val == 1.0]
    pred_labels = [lbl for lbl, score in doc.cats.items() if score >= 0.5]  # threshold = 0.5
    y_true.append(true_labels)
    y_pred.append(pred_labels)
    print(f"Text: {text}\n  Predicted: {pred_labels} ({doc.cats})\n  True: {true_labels}\n")

# convert true and predicted labels into binary indicator matrices
y_true_bin = []
y_pred_bin = []

# flatten lists for sklearn metrics
for true, pred in zip(y_true, y_pred):
    true_vec = [1 if label in true else 0 for label in all_labels]
    pred_vec = [1 if label in pred else 0 for label in all_labels]
    y_true_bin.append(true_vec)
    y_pred_bin.append(pred_vec)

y_true_bin = np.array(y_true_bin)
y_pred_bin = np.array(y_pred_bin)

# print
print("\nEvaluation Results")
print("------------------")
print("Subset Accuracy (exact match):", accuracy_score(y_true_bin, y_pred_bin))
print("F1 Score (macro):", f1_score(y_true_bin, y_pred_bin, average="macro"))
print("F1 Score (micro):", f1_score(y_true_bin, y_pred_bin, average="micro"))
print("Hamming Loss:", hamming_loss(y_true_bin, y_pred_bin))
print("Jaccard Score (macro):", jaccard_score(y_true_bin, y_pred_bin, average="macro"))
print("Jaccard Score (micro):", jaccard_score(y_true_bin, y_pred_bin, average="micro"))

# classification report per label
print("\nClassification Report:\n", classification_report(y_true_bin, y_pred_bin, target_names=all_labels))

# test prediction with reloaded model
test_text = "The government policies are right leaning and affect the economy"
doc = nlp2(test_text)
pred_labels = [lbl for lbl, score in doc.cats.items() if score >= 0.5]
print(f"Test text: {test_text}")
print(f"Predicted labels: {pred_labels}")
print(f"All category scores: {doc.cats}")
