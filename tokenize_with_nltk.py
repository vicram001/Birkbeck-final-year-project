# Import Natural Language Toolkit (NLTK)
import nltk

# Import a piece of text that we wish to comprehend
text = "Personal income levels across the U.S. vary widely, shaped by differences in industries, costs of living, and economic growth. The data, via Visual Capitalist's Pallavi Rao, for this visualization comes from the U.S. Bureau of Economic Analysis, compiled by StatsAmerica. Washington, D.C. holds the highest per capita personal income in the nation at $108,233, boosted by a concentration of high-paying government, legal, and consulting jobs. Massachusetts follows at $93,927, powered by its robust education, healthcare, and tech sectors. Connecticut, with its strong finance and insurance industries, comes in third at $93,235. All three leaders are at nearly twice the income last-ranked Mississippi ($52,017), reflecting the impact of specialized, high-skill industries on local income levels. Meanwhile, New York ($85,733), New Jersey ($84,071), and New Hampshire ($82,878) keep the broader Northeast near the top of the distribution. America’s West: The Tech Juggernaut California ($85,518) and Washington ($83,938) both place in the top 10 states by income. Their high incomes are linked to thriving technology and innovation economies, with major employers like Apple, Microsoft, and Google anchoring the regions. These states also attract high-skilled migrants, further boosting wage levels. American South Incomes Still Underperform, The bottom of the ranking is dominated by Southern states, with Mississippi at $52,017 and West Virginia at $55,138. Lower wages, coupled with economies centered on agriculture and lower-wage manufacturing, contribute to these figures. These same states also have a higher rate of poverty, but also a lower cost of living."

# Word and Sentence Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize

# Apply the word Tokenizer function to our text
word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)

# Apply the sentence Tokenizer to our text
sentence_tokens = sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)

# Stopwords
from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')
#print("Stopwords:", stopwords_list)

# Convert the text to lowercase because library of stopwords are also in lowercase and then Tokenize by words
tokens = word_tokenize(text.lower())

# Create a new list by removing all stopwords from Tokenized collection of words
tokens_wo_stopwords = [t for t in tokens if t not in stopwords_list]

print("Tokens without Stopwords:", tokens_wo_stopwords)

# Define a list of common punctuation
punc = [",", ".", "?", ";", ":", "!", "'"]

# Create a new list by removing common punctuation
tokens_wo_stopwords_punc = [t for t in tokens_wo_stopwords if t not in punc]

print("Tokens without Stopwords and Punctuation:", tokens_wo_stopwords_punc)
