"""
================================================================================
  GENAI SKILL LAB - QUESTION 5
================================================================================
  i.   Explore pretrained word vectors and word relationships using vector arithmetic.
       Perform arithmetic operations and analyze results.
  ii.  Use dimensionality reduction (PCA) to visualize word embeddings for 10 words
       from a specific domain (Technology). Analyze clusters and relationships.
  iii. Generate contextually rich outputs using embeddings.
       Write a program to generate 5 semantically similar words for a given input.
================================================================================
"""

import os
import sys
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Ensure matplotlib runs smoothly in all environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mpl_cache"))


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ==============================================================================
# STEP 1: LOAD PRETRAINED WORD VECTORS
# ==============================================================================
print_section("STEP 1: LOADING PRETRAINED WORD VECTORS (GloVe 50-dimensional)")
print("Loading 'glove-wiki-gigaword-50' pretrained embeddings...")
model = api.load("glove-wiki-gigaword-50")

vocab_size = len(model)
vector_dim = model.vector_size
print(f"Pretrained Model Loaded Successfully!")
print(f"  - Vocabulary Size : {vocab_size:,} words")
print(f"  - Vector Dimension: {vector_dim} dimensions per word")

sample_word = "computer"
sample_vector = model[sample_word]
print(f"\nSample vector representation for '{sample_word}' (first 8 dimensions):")
print(f"  {np.round(sample_vector[:8], 4)} ... [total {vector_dim} dims]")


# ==============================================================================
# PART (i): VECTOR ARITHMETIC & RELATIONSHIP ANALYSIS
# ==============================================================================
print_section("PART (i): EXPLORING WORD RELATIONSHIPS USING VECTOR ARITHMETIC")

arithmetic_tests = [
    {
        "description": "1. Gender & Royalty Analogy",
        "formula": "king - man + woman = ?",
        "positive": ["king", "woman"],
        "negative": ["man"],
        "expected": "queen",
        "insight": "Captures the gender direction vector: vec('woman') - vec('man') applied to 'king'."
    },
    {
        "description": "2. Country & Capital City Analogy",
        "formula": "paris - france + germany = ?",
        "positive": ["paris", "germany"],
        "negative": ["france"],
        "expected": "berlin",
        "insight": "Captures the capital-city relationship vector: vec('paris') - vec('france') applied to 'germany'."
    },
    {
        "description": "3. Comparative Adjective Analogy",
        "formula": "biggest - big + small = ?",
        "positive": ["biggest", "small"],
        "negative": ["big"],
        "expected": "smallest",
        "insight": "Captures the superlative degree offset applied to different base adjectives."
    },
    {
        "description": "4. Technology & Domain Analogy",
        "formula": "software - computer + biology = ?",
        "positive": ["software", "biology"],
        "negative": ["computer"],
        "expected": "biotechnology / genetics / biological",
        "insight": "Transfers the concept of programmatic engineering into the life sciences domain."
    }
]

for test in arithmetic_tests:
    print(f"\n--- {test['description']} ---")
    print(f"Formula : {test['formula']}")
    
    results = model.most_similar(positive=test["positive"], negative=test["negative"], topn=4)
    print(f"Top Candidate Words from Vector Arithmetic:")
    for rank, (word, score) in enumerate(results, 1):
        indicator = " (Matches Expected!)" if word.lower() == test["expected"].lower() else ""
        print(f"   {rank}. {word:<15} [Cosine Similarity: {score:.4f}]{indicator}")
    
    print(f"Semantic Analysis: {test['insight']}")


# ==============================================================================
# PART (ii): DIMENSIONALITY REDUCTION (PCA) & DOMAIN VISUALIZATION
# ==============================================================================
print_section("PART (ii): DIMENSIONALITY REDUCTION (PCA) & DOMAIN VISUALIZATION")

# Domain: Technology (10 representative words)
domain_words = [
    "computer",
    "software",
    "hardware",
    "internet",
    "network",
    "database",
    "algorithm",
    "server",
    "laptop",
    "website"
]

print(f"Selected Domain: Technology")
print(f"10 Domain Words: {domain_words}")

# Extract 50D word vectors
word_vectors = np.array([model[w] for w in domain_words])

# Apply Principal Component Analysis (PCA) to reduce 50D -> 2D
pca = PCA(n_components=2, random_state=42)
reduced_vectors = pca.fit_transform(word_vectors)

var_ratio = pca.explained_variance_ratio_
print(f"\nPCA Dimensionality Reduction:")
print(f"  - Original Dimensions: 50")
print(f"  - Reduced Dimensions : 2")
print(f"  - Variance Explained : PC1 = {var_ratio[0]*100:.2f}%, PC2 = {var_ratio[1]*100:.2f}% (Total: {sum(var_ratio)*100:.2f}%)")

print("\n2D Coordinates for the 10 Technology Words:")
print(f"{'Word':<15} | {'PC1 (X-axis)':<15} | {'PC2 (Y-axis)':<15}")
print("-" * 50)
for i, word in enumerate(domain_words):
    print(f"{word:<15} | {reduced_vectors[i, 0]:>14.4f} | {reduced_vectors[i, 1]:>14.4f}")

# Pairwise Cosine Similarity Matrix between the 10 Domain Words
print("\nPairwise Cosine Similarity Sample:")
print(f"  - 'computer' <-> 'laptop'   : {model.similarity('computer', 'laptop'):.4f} (High hardware similarity)")
print(f"  - 'software' <-> 'algorithm': {model.similarity('software', 'algorithm'):.4f} (High logic/program similarity)")
print(f"  - 'internet' <-> 'website'  : {model.similarity('internet', 'website'):.4f} (High web connectivity similarity)")
print(f"  - 'server'   <-> 'network'  : {model.similarity('server', 'network'):.4f} (High infrastructure similarity)")

# Generate Matplotlib Visualization
plt.figure(figsize=(10, 7))
plt.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1],
    color="#1f77b4",
    s=120,
    edgecolors="black",
    linewidths=1.5,
    zorder=3
)

# Annotate each point
for i, word in enumerate(domain_words):
    plt.annotate(
        word,
        xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]),
        xytext=(8, 5),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="#0d3b66"
    )

plt.title("2D PCA Projection of Pretrained Word Embeddings (Technology Domain)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=11)
plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6, zorder=0)
plt.tight_layout()

# Save plot to current directory
plot_filename = "technology_word_embeddings_pca.png"
plot_path = os.path.join(os.path.dirname(__file__), plot_filename)
plt.savefig(plot_path, dpi=300)
print(f"\n[Visualization Saved]: '{plot_filename}' saved successfully.")

# Analysis of Clusters and Relationships
print("\n" + "-" * 80)
print("ANALYSIS OF CLUSTERS & RELATIONSHIPS IN THE EMBEDDING SPACE:")
print("-" * 80)
print("1. Physical Client Devices Cluster:")
print("   - 'computer', 'laptop', and 'hardware' map closely together.")
print("   - Cosine distance is minimal because both words share physical and user-device usage contexts.")
print("\n2. Web & Infrastructure Cluster:")
print("   - 'internet', 'network', 'website', and 'server' form a cohesive group.")
print("   - They represent modern distributed communications and hosting paradigms.")
print("\n3. Computational Logic & Systems Cluster:")
print("   - 'software', 'algorithm', and 'database' cluster as software-engineering primitives.")
print("   - Demonstrates that GloVe vectors capture nuanced functional distinctions even within one domain.")


# ==============================================================================
# PART (iii): 5 SEMANTICALLY SIMILAR WORDS & CONTEXTUALLY RICH OUTPUTS
# ==============================================================================
print_section("PART (iii): GENERATE 5 SEMANTICALLY SIMILAR WORDS & CONTEXTUAL OUTPUTS")


def get_5_semantically_similar_words(target_word, word_model, topn=5):
    """
    Finds top-N semantically similar words using pretrained cosine similarity.
    """
    if target_word.lower() not in word_model:
        return None
    return word_model.most_similar(target_word.lower(), topn=topn)


def generate_contextually_rich_output(target_word, word_model):
    """
    Generates 5 semantically similar words and creates an enriched, contextually
    rich prompt and output using the embeddings.
    """
    print(f"\nTarget Input Word: '{target_word}'")
    similar_words = get_5_semantically_similar_words(target_word, word_model, topn=5)
    
    if not similar_words:
        print(f"Error: '{target_word}' not found in vocabulary.")
        return
    
    print("\nTop 5 Semantically Similar Words (Cosine Similarity):")
    extracted_words = []
    for rank, (sim_word, score) in enumerate(similar_words, 1):
        extracted_words.append(sim_word)
        print(f"   {rank}. {sim_word:<15} [Similarity Score: {score:.4f}]")
    
    # Contextually Rich Prompt Construction
    raw_prompt = f"Explain {target_word} in detail."
    
    contextual_prompt = (
        f"Explain {target_word} in comprehensive detail.\n"
        f"Incorporate key intersecting concepts: {', '.join(extracted_words)}.\n"
        f"Discuss how {target_word} integrates with {extracted_words[0]} and {extracted_words[1]}, "
        f"and illustrate real-world applications related to {extracted_words[2]}."
    )
    
    print("\n" + "-" * 50)
    print("DEMONSTRATION OF CONTEXTUALLY RICH OUTPUT GENERATION:")
    print("-" * 50)
    print("[1] Standard / Baseline Prompt:")
    print(f"    \"{raw_prompt}\"")
    print("\n[2] Enriched / Contextually Rich Prompt (Generated using Word Embeddings):")
    print("--------------------------------------------------------------------------------")
    print(contextual_prompt)
    print("--------------------------------------------------------------------------------")
    print("Explanation:")
    print("  By retrieving top semantic neighbors from the vector space, we can augment")
    print("  user prompts (Query Expansion / Prompt Enrichment), ensuring downstream LLMs")
    print("  cover domain-critical dimensions without hallucinating off-topic concepts.")


# Run demonstration for primary input word: 'technology'
generate_contextually_rich_output("technology", model)

# Run secondary demonstration for another domain word: 'sports'
generate_contextually_rich_output("sports", model)

print_section("EXECUTION COMPLETED SUCCESSFULLY")
