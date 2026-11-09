"""
GENAI SKILL LAB - QUESTION 2
Use word embeddings to improve prompts for a Generative AI model.
  1. Retrieve similar words using word embeddings (GloVe).
  2. Use the similar words to enrich a GenAI prompt.
  3. Use an AI model to generate responses for the original and enriched prompts.
  4. Compare the outputs in terms of detail and relevance.
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "it", "its", "this", "that", "with", "as", "by", "at", "from", "can",
    "such", "these", "those", "which", "who", "what", "how", "has", "have", "had", "not",
}


def banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def tokens(text):
    return [w.strip(".,;:!?()\"'").lower() for w in text.split() if w.strip(".,;:!?()\"'")]


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        min_new_tokens=30,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def semantic_relevance(emb, text, topic_words):
    """Cosine similarity between the mean embedding of the response's content
    words and the mean embedding of the topic words."""
    content = [w for w in tokens(text) if w not in STOPWORDS and w in emb]
    if not content:
        return 0.0
    resp_vec = np.mean([emb[w] for w in content], axis=0)
    topic_vec = np.mean([emb[w] for w in topic_words if w in emb], axis=0)
    return float(np.dot(resp_vec, topic_vec) /
                 (np.linalg.norm(resp_vec) * np.linalg.norm(topic_vec)))


def analyse(emb, text, word, related):
    toks = tokens(text)
    covered = [w for w in related if w in toks]
    return {
        "words": len(toks),
        "unique": len(set(toks)),
        "sentences": max(1, sum(text.count(p) for p in ".!?")),
        "covered": covered,
        "relevance": semantic_relevance(emb, text, [word] + related),
    }


def main():
    # ---------------- Step 1: retrieve similar words ----------------
    banner("STEP 1: RETRIEVE SIMILAR WORDS USING WORD EMBEDDINGS")
    print("Loading GloVe word embedding model (glove-wiki-gigaword-50)...")
    emb = api.load("glove-wiki-gigaword-50")

    word = "technology"
    similar_words = emb.most_similar(word, topn=5)
    print(f"\nTop 5 words similar to '{word}':")
    for w, score in similar_words:
        print(f"  {w:<15} : {score:.3f}")
    related = [w for w, _ in similar_words]

    # ---------------- Step 2: enrich the prompt ----------------
    banner("STEP 2: ORIGINAL vs ENRICHED PROMPT")
    original_prompt = f"Explain {word} in simple terms."
    enriched_prompt = (
        f"Explain {word} in simple terms. "
        f"Discuss related concepts such as {', '.join(related)}. "
        f"Give examples and explain how these concepts are connected to {word}."
    )
    print(f"Original Prompt:\n  {original_prompt}\n")
    print(f"Enriched Prompt:\n  {enriched_prompt}")

    # ---------------- Step 3: generate responses ----------------
    banner("STEP 3: GENERATING RESPONSES WITH A GENERATIVE AI MODEL")
    model_name = "google/flan-t5-large"
    print(f"Loading Hugging Face model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    original_response = generate(model, tokenizer, original_prompt)
    enriched_response = generate(model, tokenizer, enriched_prompt)

    print("\nResponse to ORIGINAL prompt:")
    print(f"  {original_response}")
    print("\nResponse to ENRICHED prompt:")
    print(f"  {enriched_response}")

    # ---------------- Step 4: compare ----------------
    banner("STEP 4: COMPARISON - DETAIL AND RELEVANCE")
    o = analyse(emb, original_response, word, related)
    e = analyse(emb, enriched_response, word, related)

    print(f"{'Metric':<40}{'Original':>15}{'Enriched':>15}")
    print("-" * 70)
    print(f"{'Detail: word count':<40}{o['words']:>15}{e['words']:>15}")
    print(f"{'Detail: unique words':<40}{o['unique']:>15}{e['unique']:>15}")
    print(f"{'Detail: sentences':<40}{o['sentences']:>15}{e['sentences']:>15}")
    print(f"{'Relevance: related terms covered (/5)':<40}"
          f"{len(o['covered']):>15}{len(e['covered']):>15}")
    print(f"{'Relevance: semantic similarity to topic':<40}"
          f"{o['relevance']:>15.3f}{e['relevance']:>15.3f}")

    print(f"\nRelated terms used (original): {', '.join(o['covered']) or 'none'}")
    print(f"Related terms used (enriched): {', '.join(e['covered']) or 'none'}")

    print("\nConclusion:")
    if e["words"] > o["words"]:
        print(f"  - Detail: the enriched response is more detailed "
              f"({e['words']} vs {o['words']} words).")
    else:
        print(f"  - Detail: the enriched response is not longer "
              f"({e['words']} vs {o['words']} words).")
    if len(e["covered"]) > len(o["covered"]):
        print(f"  - Relevance: the enriched response covers more related concepts "
              f"({len(e['covered'])} vs {len(o['covered'])}).")
    else:
        print(f"  - Relevance: both responses cover a similar number of related concepts "
              f"({len(e['covered'])} vs {len(o['covered'])}).")
    print(f"  - Semantic similarity to the topic: enriched {e['relevance']:.3f}, "
          f"original {o['relevance']:.3f} "
          f"({'enriched' if e['relevance'] > o['relevance'] else 'original'} is higher).")


if __name__ == "__main__":
    main()
