"""
================================================================================
  GENAI SKILL LAB - QUESTION 4
================================================================================
  Department Academic Report Summarization Tool
  Objective:
    Summarize long academic research reports into concise, structured summaries
    for streamlined departmental review using a pre-trained Hugging Face model.
================================================================================
"""

import os
import sys
import warnings

# Suppress warnings and Hugging Face notices for clean output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Helper class to log output both to stdout and to the 'Q4-output' file
class Logger:
    def __init__(self, filename="Q4-output"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


sys.stdout = Logger("Q4-output")


def print_banner(title):
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def summarize_text(model, tokenizer, text, max_len=100, min_len=25):
    """
    Generates an abstractive summary of an academic text using Seq2Seq beam search.
    """
    # BART-CNN models are trained only for summarization, so no task prefix is needed
    input_text = " ".join(text.split())
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = summary.strip().replace(" .", ".").replace(" ,", ",")
    return summary


def main():
    print_banner("DEPARTMENT ACADEMIC REPORT SUMMARIZATION TOOL")
    print("Application Context:")
    print("  Academic faculty, committee members, and peer reviewers must evaluate")
    print("  numerous technical research reports. This tool utilizes a pre-trained Hugging")
    print("  Face Transformer to produce fast, high-fidelity executive summaries,")
    print("  substantially reducing reading overhead while preserving key findings.\n")

    # Step 1: Model Selection & Loading
    # BART-Large fine-tuned on the CNN/DailyMail news summarization dataset
    model_name = "facebook/bart-large-cnn"
    print(f"[Step 1] Loading Pre-trained Hugging Face Summarization Model: '{model_name}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    print("         Model and Tokenizer loaded successfully!\n")

    # Step 2: Define Sample Long Academic Report
    academic_report = """
    Recent advances in transformer-based deep neural networks have revolutionized Natural
    Language Processing (NLP) across various biomedical and educational disciplines. In this
    study, the Computer Science and Artificial Intelligence Department evaluated transformer
    models deployed on academic text processing tasks, focusing on automated literature
    review generation, knowledge graph construction, and multi-document synthesis. Over a
    six-month evaluation period involving 1,250 peer-reviewed computer science manuscripts,
    the research team observed that sequence-to-sequence transformer architectures achieved a
    34.6% improvement in citation mapping efficiency and reduced qualitative review latency
    from 48 hours to under 15 minutes per manuscript. Furthermore, student comprehension scores
    increased by 22% when technical literature was augmented with automated abstractive summaries.
    However, challenges concerning token limit truncation and computational resource constraints
    on local edge hardware persist. The investigation concludes that implementing fine-tuned
    seq2seq transformers into university library repositories significantly accelerates scholarly
    discovery, provided that appropriate hallucination auditing frameworks are established.
    """

    print_banner("ORIGINAL ACADEMIC REPORT")
    print(academic_report.strip())
    
    original_words = len(academic_report.split())
    original_chars = len(academic_report.strip())
    print("-" * 80)
    print(f"Original Text Metrics : {original_words} words | {original_chars} characters\n")

    # Step 3: Generate Summary
    print_banner("GENERATING CONCISE EXECUTIVE SUMMARY")
    print("Generating abstractive summary using beam search generation (num_beams=4)...")
    
    summary = summarize_text(model, tokenizer, academic_report, max_len=90, min_len=30)

    summary_words = len(summary.split())
    summary_chars = len(summary)
    compression_ratio = (1.0 - (summary_words / original_words)) * 100.0

    # Check which key facts from the report survive in the summary
    key_facts = ["1,250", "34.6%", "15 minutes", "22%", "hallucination"]
    covered = [f for f in key_facts if f in summary]

    # Check whether the summary is copied verbatim (extractive) or rewritten (abstractive)
    flat_report = " ".join(academic_report.split()).lower()
    summary_sentences = [s.strip().lower() for s in summary.split(".") if s.strip()]
    copied = sum(1 for s in summary_sentences if s in flat_report)

    print("\n" + "=" * 80)
    print("  FINAL CONCISE SUMMARY (EXECUTIVE REVIEW)")
    print("=" * 80)
    print(f"\n\"{summary}\"\n")
    print("-" * 80)
    print("SUMMARY PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"  - Original Length       : {original_words} words ({original_chars} characters)")
    print(f"  - Summary Length        : {summary_words} words ({summary_chars} characters)")
    print(f"  - Word Reduction        : {original_words - summary_words} words eliminated")
    print(f"  - Compression Ratio     : {compression_ratio:.2f}% reduction")
    print(f"  - Key Facts Retained    : {len(covered)}/{len(key_facts)} "
          f"({', '.join(covered) if covered else 'none'})")
    print(f"  - Verbatim Sentences    : {copied}/{len(summary_sentences)} copied word-for-word from the report")
    print("=" * 80)
    print("  SUMMARIZATION PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

