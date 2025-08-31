import spacy
from spacy.util import minibatch
from spacy.training import Example
from sklearn.model_selection import train_test_split
import random

# training data
TRAIN_DATA = [
    ("The government passed a new policy on healthcare", {"cats": {"neutral": 1.0, "right": 0.0, "left": 0.0}}),
    ("Taxes should be lowered to boost the economy", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("The prime minister addressed the nation", {"cats": {"neutral": 1.0, "right": 0.0, "left": 0.0}}),
    ("Strongly supports the right wing policies", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("We need more social programs to help the poor", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}}),
    ("The government is drawing up plans to make selling family homes liable for capital gains in a fresh tax raid that could hit pensioners hoping to downsize", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
("the Chancellor is eyeing a radical shake-up of stamp duty and council tax, as well as a fresh inheritance tax raid", {"cats": {"neutral": 0.0, "right": 1.0, "left": 0.0}}),
    ("Strongly supports the left wing policies", {"cats": {"neutral": 0.0, "right": 0.0, "left": 1.0}})
]

# split into train/test
train_data, test_data = train_test_split(TRAIN_DATA, test_size=0.25, random_state=42)

# create blank model with English tokenizer
nlp = spacy.blank("en")

# add TextCategoriser to the pipeline
textcat = nlp.add_pipe("textcat")
textcat.add_label("neutral")
textcat.add_label("right")
textcat.add_label("left")

# train the model
optimizer = nlp.begin_training()
for i in range(10):  # 10 iterations
    random.shuffle(train_data)
    losses = {}
    for batch in minibatch(train_data, size=2):
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Iteration {i}, Losses: {losses}")

# save the trained model to disk
output_dir = "textcat_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# reload the model later
print("Loading saved model...")
nlp2 = spacy.load(output_dir)

# test prediction with reloaded model
test_text = "The government policies are right leaning"
doc = nlp2(test_text)
print(test_text, doc.cats)
