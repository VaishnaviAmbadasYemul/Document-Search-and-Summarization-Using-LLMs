import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rouge_score import rouge_scorer

# -------------------------------
# Sample Corpus
# -------------------------------
CORPUS = [
    "Artificial Intelligence is transforming industries by enabling machines to learn from data.",
    "Machine Learning is a subset of AI focused on statistical methods.",
    "Deep Learning uses neural networks with many layers.",
    "Natural Language Processing enables machines to understand human language.",
    "Large Language Models like GPT are powerful NLP systems."
]

# -------------------------------
# Document Search (TF-IDF + Embeddings)
# -------------------------------
class DocumentSearcher:
    def __init__(self, documents):
        self.documents = documents

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(documents)

    def search(self, query, top_n=2):
        tfidf_query = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix)[0]

        query_embedding = self.embedder.encode([query])
        embed_scores = cosine_similarity(query_embedding, self.embeddings)[0]

        final_scores = (tfidf_scores + embed_scores) / 2
        top_indices = np.argsort(final_scores)[::-1][:top_n]

        return [self.documents[i] for i in top_indices]

# -------------------------------
# Summarization (LLM)
# -------------------------------
class Summarizer:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    def summarize(self, documents, max_len=60):
        text = " ".join(documents)
        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )
        return summary[0]["summary_text"]

# -------------------------------
# Evaluation (ROUGE)
# -------------------------------
def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)

# -------------------------------
# MAIN 
# -------------------------------
def run_cli():
    print("\n--- Document Search & Summarization ---\n")

    query = "What is Artificial Intelligence?"
    print("Query:", query)

    searcher = DocumentSearcher(CORPUS)
    summarizer = Summarizer()

    docs = searcher.search(query, top_n=2)

    print("\nRetrieved Documents:")
    for d in docs:
        print("-", d)

    summary = summarizer.summarize(docs, max_len=60)

    print("\nGenerated Summary:")
    print(summary)

    reference = "Artificial Intelligence allows machines to learn from data."
    scores = evaluate_summary(reference, summary)

    print("\nROUGE Scores:")
    print(scores)

# -------------------------------
# STREAMLIT UI 
# -------------------------------
def run_streamlit():
    import streamlit as st

    st.title("📄 Document Search & Summarization (RAG)")
    st.write("Bonus Interface (as mentioned in assignment)")

    query = st.text_input("Enter your query")
    top_n = st.slider("Top documents", 1, 5, 2)
    summary_len = st.slider("Summary length", 40, 150, 60)

    if st.button("Search & Summarize"):
        searcher = DocumentSearcher(CORPUS)
        summarizer = Summarizer()

        results = searcher.search(query, top_n)

        st.subheader("🔍 Retrieved Documents")
        for i, doc in enumerate(results, 1):
            st.write(f"{i}. {doc}")

        summary = summarizer.summarize(results, summary_len)

        st.subheader("🧠 Summary")
        st.success(summary)


if __name__ == "__main__":
    run_cli()