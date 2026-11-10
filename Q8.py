"""
================================================================================
Hugging Face Sentiment Analysis Pipeline
================================================================================
Application Context:
    E-Commerce Customer Feedback & Review Monitoring System.
    Automatically classifies customer reviews as POSITIVE, NEGATIVE or MIXED
    to help prioritize customer support tickets, escalate severe issues,
    highlight top customer praise, and pass mixed feedback to the product team.
================================================================================
"""

import os
import re
import warnings

# Suppress Hugging Face Windows symlink notice and standard user warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import pipeline

# Words and punctuation that usually separate two opposing opinions in one review
# (Semicolons are not split on: they usually join two parts of the SAME opinion,
#  e.g. "battery is incredible; it lasts all day")
CONTRAST_SPLIT = r"\b(?:but|however|although|though|yet|whereas|except)\b|[.!?]"

# A clause must be at least this confident to count as clearly positive/negative
CLAUSE_CONFIDENCE = 0.85

ACTIONS = {
    "POSITIVE": "Promote to 'Verified Positive Highlights' on product page.",
    "NEGATIVE": "URGENT: Route ticket to Customer Resolution Specialist.",
    "MIXED": "Forward to Product Team: review contains both praise and complaints.",
}


def split_clauses(text):
    """Split a review into clauses at contrast words and sentence punctuation."""
    parts = re.split(CONTRAST_SPLIT, text, flags=re.IGNORECASE)
    return [p.strip(" ,") for p in parts if p and len(p.split()) >= 2]


def analyze(classifier, text):
    """
    Classify a review as POSITIVE, NEGATIVE or MIXED.

    The pre-trained model only outputs POSITIVE or NEGATIVE, so MIXED is detected
    by classifying each clause separately: if at least one clause is confidently
    positive and another is confidently negative, the review is MIXED.
    """
    overall = classifier(text)[0]
    clauses = split_clauses(text)
    clause_results = classifier(clauses) if len(clauses) > 1 else []

    strong = {r["label"] for r in clause_results if r["score"] >= CLAUSE_CONFIDENCE}
    if {"POSITIVE", "NEGATIVE"} <= strong:
        label = "MIXED"
        # Confidence for MIXED = how sure the model is about the weaker of the two sides
        score = min(
            max(r["score"] for r in clause_results if r["label"] == side)
            for side in ("POSITIVE", "NEGATIVE")
        )
    else:
        label, score = overall["label"], overall["score"]

    return {
        "label": label,
        "score": score,
        "clauses": list(zip(clauses, clause_results)),
    }


def print_result(idx, review, result):
    print(f"  * Review {idx}: \"{review}\"")
    print(f"    Sentiment : {result['label']} ({result['score'] * 100:.2f}% confidence)")
    if result["label"] == "MIXED":
        for clause, r in result["clauses"]:
            print(f"      - \"{clause}\" -> {r['label']} ({r['score'] * 100:.1f}%)")
    print(f"    Action    : {ACTIONS[result['label']]}\n")


def main():
    print("=" * 80)
    print("  REAL-WORLD APPLICATION: E-COMMERCE CUSTOMER REVIEW SENTIMENT ANALYZER")
    print("=" * 80)
    print("Use Case Description:")
    print("  Online retailers receive thousands of reviews daily. This system processes")
    print("  incoming feedback in real-time, detecting customer sentiment and recommending")
    print("  automated triage actions (e.g., escalating issues to Support or flagging praise).\n")

    # ---------------------------------------------------------
    # Step 1: Load Pre-trained Sentiment Analysis Pipeline
    # ---------------------------------------------------------
    print("[1] Loading Hugging Face Sentiment Analysis Pipeline...")
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    classifier = pipeline(
        task="sentiment-analysis",
        model=model_name
    )
    print(f"    Successfully loaded model: '{model_name}'\n")

    # ---------------------------------------------------------
    # Step 2: Define Real-World Input Sentences (Customer Reviews)
    # ---------------------------------------------------------
    sample_reviews = [
        "The battery life of this laptop is absolutely incredible; it lasts the entire workday without a charge!",
        "Customer support was terrible and the package arrived two weeks late with broken packaging.",
        "Setup was effortless and the sound quality exceeded all my expectations for the price.",
        "The screen started flickering after just three days of use. Extremely disappointed and requesting a full refund.",
        "Decent build quality, but the companion mobile app crashes constantly whenever I try to sync data.",
        "Great price but terrible delivery.",
    ]

    print("[2] Analyzing Sentiments for Sample Customer Reviews:\n")

    # ---------------------------------------------------------
    # Step 3: Run Inference & Display Results
    # ---------------------------------------------------------
    results = [analyze(classifier, review) for review in sample_reviews]

    print("-" * 80)
    print(f"{'#':<3} | {'Sentiment':<10} | {'Confidence':<10} | {'Customer Review':<50}")
    print("-" * 80)

    for idx, (review, result) in enumerate(zip(sample_reviews, results), start=1):
        confidence = f"{result['score'] * 100:.2f}%"
        display_text = (review[:47] + "...") if len(review) > 50 else review
        print(f"{idx:<3} | {result['label']:<10} | {confidence:<10} | {display_text}")

    print("-" * 80)

    # ---------------------------------------------------------
    # Step 4: Real-World Actionable Insights / Business Logic
    # ---------------------------------------------------------
    print("\n[3] Automated Business Triage Decisions:")
    for idx, (review, result) in enumerate(zip(sample_reviews, results), start=1):
        print_result(idx, review, result)

    # ---------------------------------------------------------
    # Step 5: Interactive Mode - User Enters Their Own Sentences
    # ---------------------------------------------------------
    print("=" * 80)
    print("[4] INTERACTIVE MODE: Enter your own review to analyze")
    print("    (Type 'q' or press Enter on an empty line to quit)")
    print("=" * 80)

    count = 0
    while True:
        try:
            text = input("\nEnter a sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"q", "quit", "exit"}:
            break
        count += 1
        print()
        print_result(count, text, analyze(classifier, text))

    print("=" * 80)
    print("Sentiment Analysis Execution Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
