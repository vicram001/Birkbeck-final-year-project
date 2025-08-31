import spacy
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
import random
import json
import os
from sklearn.metrics import f1_score, classification_report, hamming_loss, jaccard_score

# load training data from CSV and replace with full file path including file name
csv_file = "training_data.csv"
df = pd.read_csv(csv_file)

# extract all unique labels from CSV
all_labels = sorted({lbl.strip()
                     for labels in df["label"]
                     for lbl in str(labels).split(",")})
print("Detected labels:", all_labels)

# convert CSV to spaCy format
train_data = []
for _, row in df.iterrows():
    cats = {label: 0.0 for label in all_labels}
    for lbl in str(row["label"]).split(","):
        cats[lbl.strip()] = 1.0
    train_data.append((row["text"], {"cats": cats}))

# split into train/test
split = int(0.8 * len(train_data))
train_data, test_data = train_data[:split], train_data[split:]

# create blank model with English tokenizer / pipeline
nlp = spacy.blank("en")

# add TextCategoriser (multilabel) to the pipeline with default config
textcat = nlp.add_pipe("textcat_multilabel")
# add labels
for label in all_labels:
    textcat.add_label(label)

# train the model
optimizer = nlp.begin_training()
epochs = 30
batch_size = 8
for epoch in range(epochs):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=batch_size)
    for batch in batches:
        examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1}, Losses: {losses}")

# save trained model and replace with full file path including file name
output_dir = "textcat_multilabel_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# reload model
nlp2 = spacy.load(output_dir)

# find the best threshold and evaluate on test_data
best_threshold, best_f1 = 0.5, 0.0
for t in [i/100 for i in range(20, 81, 5)]:  # try thresholds 0.20 → 0.80
    y_true, y_pred = [], []
    for text, annotations in test_data:
        doc = nlp2(text)
        true_labels = [lbl for lbl, val in annotations["cats"].items() if val == 1.0]
        pred_labels = [lbl for lbl, score in doc.cats.items() if score >= t]
        y_true.append(true_labels)
        y_pred.append(pred_labels)

    # flatten lists for sklearn metrics
    y_true_flat, y_pred_flat = [], []
    for true, pred in zip(y_true, y_pred):
        for lbl in all_labels:
            y_true_flat.append(1 if lbl in true else 0)
            y_pred_flat.append(1 if lbl in pred else 0)

    f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"\nBest threshold found: {best_threshold:.2f} (F1-macro={best_f1:.4f})")

# compute evaluation metrics with best threshold
y_true, y_pred = [], []
for text, annotations in test_data:
    doc = nlp2(text)
    true_labels = [lbl for lbl, val in annotations["cats"].items() if val == 1.0]
    pred_labels = [lbl for lbl, score in doc.cats.items() if score >= best_threshold]
    y_true.append(true_labels)
    y_pred.append(pred_labels)

# flatten lists
y_true_flat, y_pred_flat = [], []
for true, pred in zip(y_true, y_pred):
    for lbl in all_labels:
        y_true_flat.append(1 if lbl in true else 0)
        y_pred_flat.append(1 if lbl in pred else 0)

# evaluation metrics
subset_accuracy = sum(set(t) == set(p) for t, p in zip(y_true, y_pred)) / len(y_true)
eval_results = {
    "subset_accuracy": subset_accuracy,
    "f1_macro": f1_score(y_true_flat, y_pred_flat, average="macro"),
    "f1_micro": f1_score(y_true_flat, y_pred_flat, average="micro"),
    "hamming_loss": hamming_loss(y_true_flat, y_pred_flat),
    "jaccard_macro": jaccard_score(y_true_flat, y_pred_flat, average="macro"),
    "jaccard_micro": jaccard_score(y_true_flat, y_pred_flat, average="micro")
}

# save threshold metadata to meta.json in model dir
custom_meta_path = os.path.join(output_dir, "model_meta.json")
custom_meta = {
    "best_threshold": best_threshold,
    "labels": all_labels,
    "training_file": csv_file,
    "evaluation": eval_results
}
with open(custom_meta_path, "w") as f:
    json.dump(custom_meta, f, indent=4)
print(f"Saved custom metadata to {custom_meta_path}")

# load model + meta (no CSV needed)
nlp3 = spacy.load(output_dir)
with open(os.path.join(output_dir, "model_meta.json"), "r") as f:
    loaded_meta = json.load(f)

threshold = loaded_meta.get("best_threshold", 0.5)
labels_from_meta = loaded_meta.get("labels", [])
print(f"\n[Loaded model metadata]")
print(f"Threshold: {threshold}")
print(f"Labels: {labels_from_meta}")

# test prediction with reloaded model
test_text = "The government policies are right leaning and affect the economy"
doc = nlp3(test_text)
pred_labels = [lbl for lbl, score in doc.cats.items() if score >= threshold]
print(f"\nTest text: {test_text}")
print(f"Predicted labels: {pred_labels}")
print(f"All category scores: {doc.cats}")
