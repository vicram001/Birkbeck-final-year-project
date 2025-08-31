import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import os

# training data
TRAIN_DATA = [
    ("The government passed a new policy on healthcare", {"cats": {"neutral": 1.0, "right": 0.0, "left": 0.0}}),
    ("Taxes should be lowered to boost the economy", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("The prime minister addressed the nation", {"cats": {"neutral": 1.0, "right": 0.0, "left": 0.0}}),
    ("Strongly supports the right wing policies", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("We need more social programs to help the poor", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("Strongly supports the left wing policies", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("Strongly supports the socialist policies", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("We need more social programs to help the poor", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("Conservatives vent only irritable mental gestures which seek to resemble ideas", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("The SNP leader of Glasgow City Council has condemned UK asylum policy as a machine that creates homeless refugees", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("This Government is much more comfortable with large multi-nationals than smaller enterprises that are the building blocks of our prosperity", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("Thousands illegally claiming child benefit while living abroad", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("The Department of Class Solidarity arms organizers, activists, and everyday people with the tools to expose the billionaires ransacking our democracy. Welcome to the war room of the working class.", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("voters who are sympathetic to Corbyn's party could hold their nose and back Labour to stop a Reform candidate winning in their area", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("The current policy is trampling on the rights every other girl and her rights to privacy and protected spaces", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}})
]

train_data = TRAIN_DATA
test_data = TRAIN_DATA

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
    batches = minibatch(train_data, size=2)
    for batch in batches:
        examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Iteration {i}, Losses: {losses}")

# save the model
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
