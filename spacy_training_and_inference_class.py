import spacy
from spacy.training import Example
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report
import pandas as pd
import json
import os
import random
import warnings

class TextClassifier:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.nlp = None
        self.threshold = 0.5
        self.labels = []
        self.evaluation = {}

        if model_dir:
            self._load(model_dir)

    def train_from_csv(self, csv_path, output_dir, n_iter=30, test_split=0.2, seed=42):
        random.seed(seed)

        # load training data from CSV and replace with full file path including file name
        df = pd.read_csv("c:/Users/User/Desktop/Study/Birkbeck/Final Year Project/Final Project details/Project details/Python files//training_data3.csv")
        # df = pd.read_csv("c:/Users/User/Desktop/Study/BBK/Final Year Project/Step by Step/Step 2 - Build and Train a Classifier Model/training_data3.csv")
        if "text" not in df.columns or "labels" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'labels' columns")
        df["labels"] = df["labels"].apply(lambda x: [lbl.strip() for lbl in str(x).split(",")])
        self.labels = sorted({lbl for row in df["labels"] for lbl in row})

        # shuffle and split
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_split))
        train_df, test_df = df[:split_idx], df[split_idx:]

        # prepare spaCy pipeline
        nlp = spacy.blank("en")
        textcat = nlp.add_pipe("textcat_multilabel")
        for lbl in self.labels:
            textcat.add_label(lbl)

        # convert to example objects
        def make_examples(dataframe):
            examples = []
            for _, row in dataframe.iterrows():
                doc = nlp.make_doc(row["text"])
                cats = {lbl: (1 if lbl in row["labels"] else 0) for lbl in self.labels}
                examples.append(Example.from_dict(doc, {"cats": cats}))
            return examples

        train_examples = make_examples(train_df)
        test_examples = make_examples(test_df)

        # training loop
        optimizer = nlp.initialize(get_examples=lambda: train_examples)
        for epoch in range(1, n_iter + 1):
            losses = {}
            random.shuffle(train_examples)
            batches = spacy.util.minibatch(train_examples, size=8)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Epoch {epoch}, Losses: {losses}")

        # save the model
        nlp.to_disk(output_dir)
        self.nlp = nlp
        self.model_dir = output_dir

        # find the best threshold
        self.threshold, self.evaluation = self._find_best_threshold(test_examples)

        # save the metadata
        meta = {
            "labels": self.labels,
            "best_threshold": self.threshold,
            "evaluation": self.evaluation
        }
        with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Model trained and saved to {output_dir}")
        print(f"✅ Best threshold: {self.threshold:.2f} (F1-macro={self.evaluation['f1_macro']:.4f})")

    # prediction
    def predict(self, text, return_scores=True):
        doc = self.nlp(text)
        pred_labels = [lbl for lbl, score in doc.cats.items() if score >= self.threshold]

        result = {
            "text": text,
            "predicted_labels": pred_labels,
            "threshold": self.threshold,
            "labels": self.labels
        }
        if return_scores:
            result["all_scores"] = dict(doc.cats)
        return result

    def batch_predict(self, texts, return_scores=True):
        return [self.predict(t, return_scores) for t in texts]

    def info(self):
        return {
            "model_dir": self.model_dir,
            "labels": self.labels,
            "threshold": self.threshold,
            "evaluation": self.evaluation
        }
    
    # helpers
    def _find_best_threshold(self, test_examples):
        thresholds = [i / 100 for i in range(30, 91, 5)]
        y_true = [[ex.reference.cats[lbl] for lbl in self.labels] for ex in test_examples]

        best_thresh = 0.5
        best_f1 = -1
        best_eval = {}

        # run the model on test texts
        test_texts = [ex.reference.text for ex in test_examples]
        docs = list(self.nlp.pipe(test_texts))
        y_pred_base = [{lbl: doc.cats.get(lbl, 0.0) for lbl in self.labels} for doc in docs]

        # suppress warnings for rare labels
        warnings.filterwarnings("ignore", category=UserWarning)

        for t in thresholds:
            y_pred = [[1 if cats[lbl] >= t else 0 for lbl in self.labels] for cats in y_pred_base]

            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
            subset_acc = accuracy_score(y_true, y_pred)
            jaccard_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
            jaccard_micro = jaccard_score(y_true, y_pred, average="micro", zero_division=0)

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_thresh = t
                best_eval = {
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "subset_accuracy": subset_acc,
                    "jaccard_macro": jaccard_macro,
                    "jaccard_micro": jaccard_micro,
                    "per_label": classification_report(
                        y_true, y_pred, target_names=self.labels, zero_division=0, output_dict=True
                    )
                }

        warnings.filterwarnings("default", category=UserWarning)
        return best_thresh, best_eval

    def _load(self, model_dir):
        self.nlp = spacy.load(model_dir)
        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.labels = meta.get("labels", [])
                self.threshold = meta.get("best_threshold", 0.5)
                self.evaluation = meta.get("evaluation", {})

# example usage
if __name__ == "__main__":
    clf = TextClassifier()
    clf.train_from_csv(
        "training_data3.csv",
        output_dir="textcat_multilabel_model",
        n_iter=30
    )

    clf = TextClassifier("textcat_multilabel_model")
    print(clf.info())

    result = clf.predict("The government policies are right leaning and affect the economy")
    print(result)
