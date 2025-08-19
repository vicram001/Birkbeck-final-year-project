# import Natural Language Toolkit (NLTK), scikit-learn etc.
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# download NLTK resources (run once), and load required tools, downloads the tokenizer model for splitting text into words
nltk.download("punkt")

# downloads the list of common stopwords (e.g. "the", "is", "and") so they can be removed
nltk.download("stopwords")

# define stopwords
stop_words = set(stopwords.words("english"))

# define a preprocessing function: Lowercases the text, tokenizes into words, keeps only alphabetic words (isalpha())
# removes stopwords, joins the cleaned words back into a string for use with the vectorizer.
def preprocess(text: str) -> str:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# multiple texts and labels (>= 2 classes), prepare example texts and labels, raw example sentences
texts = [
    "Tax cuts stimulate economic growth.",
    "More government intervention is required.",
    "Less government intervention is best.",
    "Universal healthcare should be guaranteed for everyone.",
    "We need stronger regulation on corporations.",
    "Lower taxes and smaller government increase freedom.",
    "Investing in public services improves equality.",
    "Environmental protections must be strengthened.",
]

# classification labels (e.g. "left" or "right"), align lengths & provide 2+ classes
labels = ["right", "left", "right", "left", "left", "right", "left", "left"]  # align lengths & provide 2+ classes

# sanity check (helps catch future mismatches)
assert len(texts) == len(labels), f"{len(texts)} texts vs {len(labels)} labels"

# preprocess all text, produces a cleaned, normalized text list ready for vectorization
processed_texts = [preprocess(t) for t in texts]

# split into train/test split sets
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.33, random_state=42, stratify=labels
)

# vectorize, convert text to numbers with TF-IDF features, creates numerical features based on term frequency–inverse document frequency, each document is now represented as a sparse vector of numbers.
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

# train the Logistic Regression classifier, learns a statistical model to predict the class label from the numeric features
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# evaluate and Predict on test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# predict a new text, in the minimal working example, splits the data into train/test sets so that the model can be evaluated on unseen examples
new_text = "The government should cut taxes."
new_processed = preprocess(new_text)
new_vec = vectorizer.transform([new_processed])
print("Prediction:", clf.predict(new_vec)[0])
