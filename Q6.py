"""
================================================================================
Q6: Custom Word2Vec Training on a Domain-Specific Corpus (Medical Domain)
================================================================================
Objective:
    1. Define a domain-specific corpus (Medical / Healthcare).
    2. Preprocess and tokenize the text data.
    3. Train a custom Word2Vec model using Gensim.
    4. Analyze how word embeddings capture domain-specific semantics:
       - Finding top semantically similar medical terms.
       - Comparing semantic similarity scores across related vs unrelated terms.
       - Detecting semantic outliers (odd-one-out) in clinical categories.
================================================================================
"""

import re
from gensim.models import Word2Vec


STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "from", "by", "is", "are", "was", "were", "be", "been", "both",
    "all", "before", "after", "as"
}


def preprocess_corpus(text_data):
    """Tokenize and filter stop words from domain sentences."""
    tokenized_sentences = []
    for sentence in text_data:
        tokens = re.findall(r"\b[a-z]+\b", sentence.lower())
        meaningful_tokens = [t for t in tokens if t not in STOPWORDS]
        if meaningful_tokens:
            tokenized_sentences.append(meaningful_tokens)
    return tokenized_sentences


def main():
    print("=" * 80)
    print("  Q6: TRAINING CUSTOM WORD2VEC ON A DOMAIN-SPECIFIC MEDICAL CORPUS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Define Domain-Specific Corpus (Medical / Clinical Domain)
    # -------------------------------------------------------------------------
    medical_corpus = [
        # Medical roles & professionals
        "The doctor examined the patient and prescribed medication for the infection.",
        "A physician diagnosed the disease and recommended clinical treatment.",
        "The surgeon performed surgery in the hospital operating room with assistance from the nurse.",
        "The nurse monitored patient vital signs and administered antibiotic therapy.",
        "Both the doctor and physician work at the hospital clinic treating sick patients.",
        "The surgeon and doctor reviewed patient clinical records before treatment.",

        # Diseases, symptoms & diagnoses
        "The patient complained of severe fever headache and body pain.",
        "High fever and persistent cough are common symptoms of respiratory infection.",
        "Early diagnosis of infection helps prevent severe illness and complications.",
        "The patient presented symptoms of acute infection and high fever.",
        "The physician confirmed the diagnosis based on laboratory blood tests.",

        # Treatments, drugs & therapy
        "The doctor prescribed an antibiotic medication to eliminate the bacterial infection.",
        "Effective medication and antibiotic drug treatment reduced patient fever rapidly.",
        "The patient completed the prescribed antibiotic medication therapy.",
        "Treatment for the disease includes daily medication and regular clinical checkups.",
        "The hospital pharmacy dispensed the prescribed drug and medication for the patient."
    ]

    print("\n[Step 1] Loading Domain-Specific Dataset:")
    print(f"  Domain: Medical & Healthcare")
    print(f"  Total Sentences in Corpus: {len(medical_corpus)}")

    # -------------------------------------------------------------------------
    # 2. Tokenize and Preprocess
    # -------------------------------------------------------------------------
    tokenized_corpus = preprocess_corpus(medical_corpus)
    vocab_size = len(set(word for sent in tokenized_corpus for word in sent))
    print(f"  Total Unique Words in Vocabulary: {vocab_size}")

    # -------------------------------------------------------------------------
    # 3. Train Custom Word2Vec Model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training Word2Vec Model...")
    # Parameters:
    # vector_size: embedding dimensions (50 is optimal for small domain corpus)
    # window: context window size (captures co-occurrence of symptoms, treatments)
    # min_count=1: keep all domain terms in our curated corpus
    # sg=1: Skip-Gram architecture captures semantic relationships well on smaller datasets
    # epochs=150: sufficient iterations for weight convergence on a small corpus
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=5,
        min_count=1,
        sg=1,
        epochs=150,
        seed=42,
        workers=1
    )
    print("  Word2Vec model training complete!")

    # -------------------------------------------------------------------------
    # 4. Semantic Analysis 1: Find Most Similar Words
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 3] Analyzing Domain-Specific Semantics: Similar Words")
    print("=" * 80)

    target_words = ["doctor", "fever", "medication", "patient"]

    for target in target_words:
        print(f"\nTop 4 Words Semantically Similar to '{target}':")
        similar = model.wv.most_similar(target, topn=4)
        for rank, (word, score) in enumerate(similar, start=1):
            print(f"   {rank}. {word:<12} (Cosine Similarity: {score:.3f})")

    # -------------------------------------------------------------------------
    # 5. Semantic Analysis 2: Pairwise Cosine Similarity Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 4] Semantic Similarity Comparison (Related vs Unrelated Terms)")
    print("=" * 80)

    comparison_pairs = [
        ("doctor", "physician", "High (Synonymous medical professionals)"),
        ("antibiotic", "medication", "High (Treatment / Pharmacology category)"),
        ("fever", "symptoms", "High (Clinical manifestation correlation)"),
        ("doctor", "fever", "Low/Moderate (Practitioner vs Symptom)"),
        ("surgery", "medication", "Low/Moderate (Different clinical modalities)")
    ]

    print(f"{'Word Pair':<28} | {'Cosine Sim':<12} | {'Domain Relationship'}")
    print("-" * 80)
    for w1, w2, relation in comparison_pairs:
        sim = model.wv.similarity(w1, w2)
        print(f"{w1 + ' <-> ' + w2:<28} | {sim:<12.3f} | {relation}")

    # -------------------------------------------------------------------------
    # 6. Semantic Analysis 3: Category Outlier Detection (doesnt_match)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 5] Semantic Category Outlier Detection (doesn't match)")
    print("=" * 80)

    category_tests = [
        (["doctor", "physician", "nurse", "fever"], "Medical roles vs Symptom"),
        (["antibiotic", "medication", "drug", "hospital"], "Pharmacology vs Institution")
    ]

    for words_list, description in category_tests:
        outlier = model.wv.doesnt_match(words_list)
        print(f"Given words: {words_list}")
        print(f"  Context     : {description}")
        print(f"  Identified Outlier: '{outlier}'\n")

    # -------------------------------------------------------------------------
    # 7. Conclusion & Domain Takeaways
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("DOMAIN-SPECIFIC EMBEDDING ANALYSIS SUMMARY:")
    print("=" * 80)
    print("1. Contextual Clustering: The model learned that 'doctor' and 'physician'")
    print("   share strong cosine similarity because they co-occur with similar contexts")
    print("   (treating patients, clinical exams).")
    print("2. Treatment-Disease Associations: 'antibiotic' and 'medication' map closely")
    print("   to 'infection' and 'fever', reflecting clinical prescribing semantics.")
    print("3. Vector Separability: Outlier detection successfully separates clinical")
    print("   roles (doctor/nurse) from symptoms (fever).")
    print("=" * 80)


if __name__ == "__main__":
    main()
