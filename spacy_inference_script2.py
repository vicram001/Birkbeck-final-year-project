import spacy
import json
import os

# load a trained spaCy model and its metadata (threshold + labels)
class TextClassifier:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.nlp = spacy.load(model_dir)
        meta_path = os.path.join(model_dir, "model_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Custom metadata not found: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.threshold = meta.get("best_threshold", 0.5)
        self.labels = meta.get("labels", [])

        # optional: store evaluation info if needed
        self.evaluation = meta.get("evaluation", {})
    def predict(self, text, return_scores=True):
        
        # predict labels for a given text using the trained model + saved threshold
        doc = self.nlp(text)
        pred_labels = [lbl for lbl, score in doc.cats.items() if score >= self.threshold]
        result = {
            "text": text,
            "predicted_labels": pred_labels,
            "threshold": self.threshold,
            "labels": self.labels
        }

        if return_scores:
            result["all_scores"] = doc.cats
        return result
    def batch_predict(self, texts, return_scores=True):

        # predict labels for a list of texts
        return [self.predict(text, return_scores=return_scores) for text in texts]
    def info(self):
 
        # return model info: labels, threshold, evaluation metrics (if any)
        return {
            "model_dir": self.model_dir,
            "labels": self.labels,
            "threshold": self.threshold,
            "evaluation": self.evaluation
        }

# usage example, load once and reuse efficiently
clf = TextClassifier("textcat_multilabel_model")

# single prediction
result = clf.predict("The government policies are right leaning and affect the economy")
print(result)

# batch predictions
texts = [
    "Welfare improvements help reduce inequality",
    "The economy is showing signs of recovery"
]
results = clf.batch_predict(texts)
for result in results:
    print(result)

# model info
print(clf.info())
