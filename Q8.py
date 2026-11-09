"""
================================================================================
Hugging Face Sentiment Analysis Pipeline
================================================================================
Application Context:
    E-Commerce Customer Feedback & Review Monitoring System.
    Automatically classifies customer reviews into POSITIVE or NEGATIVE sentiment
    to help prioritize customer support tickets, escalate severe issues, and
    highlight top customer praise.
================================================================================
"""

import os
import warnings

# Suppress Hugging Face Windows symlink notice and standard user warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import pipeline


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
        "Decent build quality, but the companion mobile app crashes constantly whenever I try to sync data."
    ]

    print("[2] Analyzing Sentiments for Incoming Customer Reviews:\n")

    # ---------------------------------------------------------
    # Step 3: Run Inference & Display Results
    # ---------------------------------------------------------
    results = classifier(sample_reviews)

    print("-" * 80)
    print(f"{'#':<3} | {'Sentiment':<10} | {'Confidence':<10} | {'Customer Review':<50}")
    print("-" * 80)

    for idx, (review, result) in enumerate(zip(sample_reviews, results), start=1):
        sentiment = result["label"]
        confidence = f"{result['score'] * 100:.2f}%"
        # Truncate review for tabular view if very long
        display_text = (review[:47] + "...") if len(review) > 50 else review
        print(f"{idx:<3} | {sentiment:<10} | {confidence:<10} | {display_text}")

    print("-" * 80)

    # ---------------------------------------------------------
    # Step 4: Real-World Actionable Insights / Business Logic
    # ---------------------------------------------------------
    print("\n[3] Automated Business Triage Decisions:")
    for idx, (review, result) in enumerate(zip(sample_reviews, results), start=1):
        sentiment = result["label"]
        score = result["score"]
        if sentiment == "POSITIVE":
            action = "Promote to 'Verified Positive Highlights' on product page."
        else:
            action = "URGENT: Route ticket to Customer Resolution Specialist."

        print(f"  * Review {idx} ({sentiment}, {score*100:.1f}%):")
        print(f"    Text: \"{review}\"")
        print(f"    Action: {action}\n")

    print("=" * 80)
    print("Sentiment Analysis Execution Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
