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
from sklearn.cluster import KMeans
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
        "formula": "better - good + bad = ?",
        "positive": ["better", "bad"],
        "negative": ["good"],
        "expected": ["worse"],
        "insight": "Captures the comparative-degree offset (good -> better) applied to 'bad'."
    },
    {
        "description": "4. Technology & Domain Analogy",
        "formula": "software - computer + biology = ?",
        "positive": ["software", "biology"],
        "negative": ["computer"],
        "expected": ["biotechnology", "genetics", "bioinformatics"],
        "insight": "Transfers the concept of programmatic engineering into the life sciences domain."
    },
    {
        "description": "5. Superlative Adjective Analogy (Limitation Case)",
        "formula": "biggest - big + small = ?",
        "positive": ["biggest", "small"],
        "negative": ["big"],
        "expected": ["smallest"],
        "insight": "Tests whether the superlative offset (big -> biggest) transfers to 'small'."
    }
]

analogy_summary = []
for test in arithmetic_tests:
    print(f"\n--- {test['description']} ---")
    print(f"Formula : {test['formula']}")

    results = model.most_similar(positive=test["positive"], negative=test["negative"], topn=4)
    top_words = [w.lower() for w, _ in results]
    print(f"Top Candidate Words from Vector Arithmetic:")
    for rank, (word, score) in enumerate(results, 1):
        indicator = " (Matches Expected!)" if word.lower() in test["expected"] else ""
        print(f"   {rank}. {word:<15} [Cosine Similarity: {score:.4f}]{indicator}")

    print(f"Expected : {' / '.join(test['expected'])}")
    print(f"Semantic Analysis: {test['insight']}")
    if top_words[0] in test["expected"]:
        verdict = "SUCCESS - the expected word is the top result."
    elif any(w in test["expected"] for w in top_words):
        verdict = "PARTIAL - the expected word appears in the top 4, but not at rank 1."
    else:
        verdict = (f"FAILED - the expected word is not in the top 4; the top result is "
                   f"'{results[0][0]}'. The relationship is not encoded as a consistent "
                   f"direction for these words in GloVe-50.")
        nearby = [w for w in top_words if w.startswith(test["positive"][0][:3])]
        if nearby:
            verdict += (f" The result stays close to the input word "
                        f"'{test['positive'][0]}' ({', '.join(nearby)}), showing that "
                        f"its own meaning dominates the arithmetic.")
    print(f"Result   : {verdict}")
    analogy_summary.append((test["formula"], verdict.split(" ")[0]))

print("\nAnalogy Summary:")
for formula, status in analogy_summary:
    print(f"  {formula:<36} {status}")


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

def similarity_label(score):
    if score >= 0.65:
        return "High"
    if score >= 0.45:
        return "Moderate"
    return "Low"


# Pairwise Cosine Similarity between selected pairs (computed in the full 50D space)
print("\nPairwise Cosine Similarity Sample (50D space):")
for a, b in [("computer", "laptop"), ("software", "algorithm"),
             ("internet", "website"), ("server", "network")]:
    s = model.similarity(a, b)
    print(f"  - {a!r:<11}<-> {b!r:<12}: {s:.4f} ({similarity_label(s)} similarity)")

# Cluster the 2D PCA points so the analysis matches what the plot shows
N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
cluster_ids = kmeans.fit_predict(reduced_vectors)
cluster_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# Generate Matplotlib Visualization (points coloured by cluster)
plt.figure(figsize=(10, 7))
plt.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1],
    c=[cluster_colors[c] for c in cluster_ids],
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

# Analysis of Clusters and Relationships (computed from the data, not hard-coded)
print("\n" + "-" * 80)
print("ANALYSIS OF CLUSTERS & RELATIONSHIPS IN THE EMBEDDING SPACE:")
print("-" * 80)

sim_matrix = np.array([[model.similarity(a, b) for b in domain_words] for a in domain_words])

print(f"1. K-Means Clusters on the 2D PCA Projection (k = {N_CLUSTERS}):")
for c in range(N_CLUSTERS):
    members = [w for w, cid in zip(domain_words, cluster_ids) if cid == c]
    idx = [domain_words.index(w) for w in members]
    if len(idx) > 1:
        pair_sims = [sim_matrix[i, j] for i in idx for j in idx if i < j]
        cohesion = f"average 50D similarity within cluster = {np.mean(pair_sims):.4f}"
    else:
        cohesion = "single-word cluster (isolated in the plot)"
    print(f"   - Cluster {c + 1} ({cluster_colors[c].replace('tab:', '')} in plot): {', '.join(members)}")
    print(f"       {cohesion}")

print("\n2. Nearest Neighbour of Each Word (within the 10 words, 50D cosine similarity):")
for i, w in enumerate(domain_words):
    sims = sim_matrix[i].copy()
    sims[i] = -1
    j = int(np.argmax(sims))
    print(f"   - {w:<10} -> {domain_words[j]:<10} ({sims[j]:.4f}, {similarity_label(sims[j])})")

print("\n3. Outlier Word (lowest average similarity to the other 9 words):")
avg_sims = (sim_matrix.sum(axis=1) - 1.0) / (len(domain_words) - 1)
outlier = int(np.argmin(avg_sims))
central = int(np.argmax(avg_sims))
print(f"   - Outlier : '{domain_words[outlier]}' (average similarity {avg_sims[outlier]:.4f})")
print(f"   - Most central word : '{domain_words[central]}' (average similarity {avg_sims[central]:.4f})")
print("   - The outlier tends to appear in different contexts (e.g. mathematics or research")
print("     text) than the everyday technology words, so GloVe places it further away.")

print(f"\n4. Note: the 2D plot keeps only {sum(var_ratio)*100:.1f}% of the variance, so some words")
print("   that look close in 2D may be less similar in the full 50D space (compare the")
print("   cluster memberships above with the 50D nearest neighbours).")


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
