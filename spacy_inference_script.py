import spacy
import json
import os

def load_model_with_threshold(model_dir):
    # load a trained spaCy model and its custom metadata (threshold + labels)
    nlp = spacy.load(model_dir)
    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    threshold = meta.get("best_threshold", 0.5)
    labels = meta.get("labels", [])
    return nlp, threshold, labels

def predict(text, nlp, threshold):
    # predict labels for a given text using a trained model + saved threshold
    doc = nlp(text)
    pred_labels = [lbl for lbl, score in doc.cats.items() if score >= threshold]
    return {
        "text": text,
        "predicted_labels": pred_labels,
        "all_scores": doc.cats,
        "threshold": threshold,
        "labels": labels
    }

if __name__ == "__main__":
    # load model at startup
    model_dir = "textcat_multilabel_model"
    nlp, threshold, labels = load_model_with_threshold(model_dir)

    # example predictions
    texts = [
        "The government policies are right leaning and affect the economy",
        "Welfare improvements help reduce inequality"
    ]

    for txt in texts:
        result = predict(txt, nlp, threshold)
        print(result)

    # single text line test
    result = predict("The economy is improving due to welfare policies", nlp, threshold)
    print(result)
