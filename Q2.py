import gensim.downloader as api

# Load pre-trained word embeddings
print("Loading word embedding model...")
model = api.load("glove-wiki-gigaword-50")

# Select input word
word = "technology"

# Find 5 semantically similar words
similar_words = model.most_similar(word, topn=5)

# Display similar words
print("\nSimilar words:")
for w, score in similar_words:
    print(w, ":", round(score, 3))

# Extract similar words
words = [w for w, score in similar_words]

# Original prompt
original_prompt = """
Explain technology in simple terms.
"""

# Enriched prompt
enriched_prompt = f"""
Explain technology in simple terms.
Discuss related concepts such as {', '.join(words)}.
Give examples and explain how these concepts are connected to technology.
"""

print("\nOriginal Prompt:")
print(original_prompt)

print("\nEnriched Prompt:")
print(enriched_prompt)
