# Import spacy, scikit-learn etc.
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load English model
nlp = spacy.load("en_core_web_sm")

# preprocessing function using spaCy
def preprocess_spacy(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

# example dataset (≥ 2 classes, lengths match)
texts = [
    "Tax cuts stimulate economic growth.",
    "Universal healthcare should be guaranteed for everyone.",
    "We need stronger regulation on corporations.",
    "Lower taxes and smaller government increase freedom.",
    "Investing in public services improves equality.",
    "Environmental protections must be strengthened.",
]
labels = ["right", "left", "left", "right", "left", "left"]

assert len(texts) == len(labels), f"{len(texts)} texts vs {len(labels)} labels"

# preprocess all texts with spaCy
processed_texts = [preprocess_spacy(t) for t in texts]

# split into train/test sets
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.33, random_state=42, stratify=labels
)

# vectorize, convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

# train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7) Evaluate and predict on test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 8) Predict a new text
new_text = "The government should cut taxes."
new_processed = preprocess_spacy(new_text)
new_vec = vectorizer.transform([new_processed])
print("Prediction:", clf.predict(new_vec)[0])
