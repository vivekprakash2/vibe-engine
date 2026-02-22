# 🎵 VibeEngine
### Mapping the architecture of your mood and recommending songs based on your vibes.

**VibeEngine** is a multimodal music discovery platform that moves beyond simple keyword matching to understand the semantic essence of a "vibe". By leveraging **Gemini-powered prompt enrichment** and a high-performance vector search over a corpus of **500,000 songs**, VibeEngine translates messy human emotions, memories, and images into the perfect soundtrack.

---

## ✨ Key Features

* **Multimodal Vibe Extraction:** Users can describe a mood via text or upload an image (e.g., a sunset, a busy street). 
* **Gemini Vision Integration:** The system analyzes images to extract musical attributes like tempo, genre, and atmosphere to guide search queries.
* **Semantic Enrichment:** Raw user prompts are enriched using **Gemini** to generate a "lyrical search vector," significantly improving retrieval accuracy for abstract or poetic queries.
* **Hybrid Ranking Engine:** A custom scoring algorithm that combines semantic similarity with a logarithmic popularity boost to prioritize definitive recordings.
* **AI Insight & Explainability:** Every recommendation features a generated justification, explaining the specific connection between the user's input and the chosen track.
* **Minimalist AI Interface:** A clean, cream-colored UI inspired by modern AI tools like Claude and Gemini, optimized for a seamless discovery experience.

---

## 🛠️ Tech Stack

### **Backend**
* **Language:** Python 3.12
* **API Framework:** FastAPI
* **AI/ML Models:** * **Gemini 1.5 Flash:** Multimodal analysis and prompt enrichment.
    * **Sentence-Transformers (`all-MiniLM-L6-v2`):** Generating high-dimensional song embeddings.
    * **Scikit-Learn:** Optimized Cosine Similarity for vector retrieval.
* **Data Science:** NumPy and Pandas for high-frequency tracking and metadata management.

### **Frontend**
* **Framework:** React + Vite
* **Styling:** Tailwind CSS (v4) with a custom "Cream & Slate" minimalist theme.
* **Icons:** Lucide-React.

---

## 📊 Methodology: Hybrid Ranking

To solve the "Cover Contamination" problem (where low-quality covers crowd out original hits), VibeEngine utilizes a custom scoring function:

$$Score = (Similarity \cdot w_{semantic}) + (log_{10}(Views + 1) \cdot w_{popularity})$$

This ensures that while a cover might have high lyrical similarity, the high-view original version is weighted more heavily in the final ranking.

---

## 🚀 Getting Started

### **Backend Setup**
1.  Navigate to the `/vibe-backend` directory.
2.  Install dependencies:
    ```bash
    pip install requirements.txt
    ```
3.  Set your API Key:
    ```bash
    export GEMINI_API_KEY="your_key_here"
    ```
4.  Run the server:
    ```bash
    python main.py
    ```

### **Frontend Setup**
1.  Navigate to the `/vibe-frontend` directory.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```

---

## 👥 The Team
* **Ajay Suresh**
* **Vivek Prakash**
* **Aaryan Kadam**

---

## 🙏 Acknowledgments
* **Georgia Tech MSA Program** for the statistical and analytical foundations.
* **GT Hacklytics 2026** for the platform to innovate.
