# Document-Search-and-Summarization-Using-LLMs

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that searches a corpus of documents based on a user query and generates a concise summary using a Large Language Model (LLM).

The solution combines:

Traditional Information Retrieval (TF-IDF)

Semantic Search (Sentence Embeddings)

LLM-based Summarization

Evaluation using ROUGE metrics

## Objective

To design a system that:

Retrieves the most relevant documents for a given query

Generates a coherent summary of the retrieved documents

Evaluates both search relevance and summary quality

## System Architecture

User Query
   ↓
Hybrid Document Search
(TF-IDF + Embeddings)
   ↓
Top-N Relevant Documents
   ↓
LLM-based Summarization
   ↓
Final Summary

## Dataset

For demonstration purposes, a small textual corpus is used directly in the code.
This approach is acceptable as the assignment focuses on methodology, not dataset size.

The corpus contains short descriptions related to:

Artificial Intelligence

Machine Learning

Deep Learning

NLP

Large Language Models

## Document Search Methodology
1. TF-IDF Search

Converts documents into numerical vectors

Measures keyword-level similarity using cosine similarity

2. Semantic Search (Embeddings)

Uses Sentence Transformers (all-MiniLM-L6-v2)

Captures semantic meaning beyond keywords

3. Hybrid Scoring

Final relevance score is computed as:


## Document Summarization

Uses Facebook BART Large CNN

Retrieved documents are concatenated

Generates a coherent and concise summary

Summary length is configurable

## Evaluation
**Search Evaluation**

Queries are designed so that relevant documents are known

Retrieval correctness is verified manually

**Summary Evaluation**

Uses ROUGE-1 and ROUGE-L metrics

Compares generated summaries with reference summaries

This provides an objective quality measure.

## Streamlit Interface 

**The Streamlit app allows:**

Query input

Selecting number of documents (Top-N)

Adjusting summary length

Viewing retrieved documents and summary


## Installation
**Prerequisites**

Python 3.9+

**Install dependencies**
pip install scikit-learn sentence-transformers transformers torch rouge-score streamlit

**How to Run**
streamlit run rag_assignment.py

## Scalability

Embeddings can be precomputed

Corpus can be expanded easily

Modular logic supports future extensions (PDFs, APIs, databases)

## Conclusion

This project successfully demonstrates a Retrieval-Augmented Generation pipeline that meets all mandatory requirements of the assignment.
The solution is efficient, scalable, and interview-ready, with a bonus interface for enhanced usability.
