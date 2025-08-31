import spacy
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
import random
import json
import os
from sklearn.metrics import f1_score, classification_report, hamming_loss, jaccard_score

# ---------------------------
# TRAINING PHASE
# ---------------------------

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

# save the trained model and replace with full file path including file name
output_dir = "textcat_multilabel_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# reload model
nlp2 = spacy.load(output_dir)

# ---------------------------
# THRESHOLD OPTIMISATION
# ---------------------------

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

# save threshold to meta.json in model dir
meta_path = os.path.join(output_dir, "meta.json")
meta_data = {
    "best_threshold": best_threshold,
    "labels": all_labels,
    "training_file": csv_file,
    "evaluation": eval_results
}
with open(meta_path, "w") as f:
    json.dump(meta_data, f, indent=4)
print(f"Saved metadata to {meta_path}")

# load threshold + labels back when reusing model
with open(meta_path, "r") as f:
    loaded_meta = json.load(f)
threshold = loaded_meta.get("best_threshold", 0.5)
labels_from_meta = loaded_meta.get("labels", [])
print(f"Loaded threshold: {threshold}")
print(f"Loaded labels: {labels_from_meta}")

# final evaluation with saved threshold
y_true, y_pred = [], []
for text, annotations in test_data:
    doc = nlp2(text)
    true_labels = [lbl for lbl, val in annotations["cats"].items() if val == 1.0]
    pred_labels = [lbl for lbl, score in doc.cats.items() if score >= threshold]
    y_true.append(true_labels)
    y_pred.append(pred_labels)
    print(f"Text: {text}\n Predicted: {pred_labels} ({doc.cats})\n True: {true_labels}\n")

# flatten lists for sklearn
y_true_flat, y_pred_flat = [], []
for true, pred in zip(y_true, y_pred):
    for lbl in all_labels:
        y_true_flat.append(1 if lbl in true else 0)
        y_pred_flat.append(1 if lbl in pred else 0)

# metrics
print("\nEvaluation Results")
subset_accuracy = sum(set(t) == set(p) for t, p in zip(y_true, y_pred)) / len(y_true)
print("Subset Accuracy (exact match):", subset_accuracy)
print("F1 Score (macro):", f1_score(y_true_flat, y_pred_flat, average="macro"))
print("F1 Score (micro):", f1_score(y_true_flat, y_pred_flat, average="micro"))
print("Hamming Loss:", hamming_loss(y_true_flat, y_pred_flat))
print("Jaccard Score (macro):", jaccard_score(y_true_flat, y_pred_flat, average="macro"))
print("Jaccard Score (micro):", jaccard_score(y_true_flat, y_pred_flat, average="micro"))

# classification report for multi-label provide labels explicitly
print("\nClassification Report:\n")
print(classification_report(
    y_true_flat,
    y_pred_flat,
    labels=list(range(len(all_labels))),
    target_names=all_labels,
    zero_division=0
))

# test prediction with reloaded model and auto threshold
test_text = "The government policies are right leaning and affect the economy"
doc = nlp2(test_text)
pred_labels = [lbl for lbl, score in doc.cats.items() if score >= threshold]
print(f"\nTest text: {test_text}")
print(f"Predicted labels: {pred_labels}")
print(f"All category scores: {doc.cats}")
