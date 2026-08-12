# 🎵 HarmonyAI: Hybrid Music Recommender

**HarmonyAI** is an intelligent music recommendation system that combines **lyrical meaning**, **audio-style metadata**, and **emotional context** to suggest songs that match your mood or find similar tracks.

It blends modern **Natural Language Understanding (NLP)** with traditional music features to create a hybrid recommendation engine that feels personal and context-aware.

---

## ☁️ Live Demo

Try **HarmonyAI** instantly in your browser: 

🌐 [https://harmonyai-baesxhrdrhaedxwuzxpjd3.streamlit.app/](https://harmonyai-baesxhrdrhaedxwuzxpjd3.streamlit.app/)

No installation required just open the link, describe your mood or select a song, and HarmonyAI will recommend tracks that match your vibe. 🎧

---

## 🌟 Key Features

- 🧠 **Hybrid Embedding Engine** — combines:
  - SBERT (Sentence-BERT) embeddings from song lyrics and metadata  
  - Numeric audio features (energy, danceability, tempo, etc.)  
  - Emotion tags from lyrics
- 🔍 **Two recommendation modes**
  1. **Free-text mood search:** type anything like _"melancholic acoustic ballad"_ or _"energetic 2000s pop rock"_  
  2. **Find similar songs:** pick a song title, and HarmonyAI returns similar tracks
- ⚡ **Real-time search** using **FAISS (Facebook AI Similarity Search)**
- 💻 **Streamlit interface** for an elegant and fast interactive experience
- 🎧 Includes Spotify links for quick listening previews

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Language Model | [`sentence-transformers/all-MiniLM-L6-v2`](https://www.sbert.net/) |
| Embedding Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Interface | [Streamlit](https://streamlit.io) |
| Data Handling | `pandas`, `numpy`, `scikit-learn` |
| Deployment | Streamlit Cloud / local Conda environment |

---

## 🧠 How It Works

1. **Data preprocessing:** merges Spotify song metadata and lyrics  
2. **Embedding generation:**  
   - Lyrics + metadata encoded with Sentence-BERT  
   - Numeric & emotional vectors scaled and concatenated  
3. **FAISS index:** stores all hybrid embeddings for fast cosine similarity search  
4. **Streamlit front-end:** provides an intuitive interface to query and visualize recommendations

---

## 🚀 Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/Iyanuoluwa007/HarmonyAI.git
cd HarmonyAI
```
### 2️⃣ Create a Conda environment
```bash
conda create -n harmonyai python=3.10 -y
conda activate harmonyai
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---
Open your browser to http://localhost:8501 and enjoy HarmonyAI 🎶

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to your GitHub

2. Go to share.streamlit.io

3. Connect your repo and set:

- Main file path: app/streamlit_app.py

4. Deploy 

---

## 📂 Repository Structure

```bash
HarmonyAI/
├── app/
│   ├── streamlit_app.py        # 🎯 Main Streamlit application
│   └── artifacts/              # FAISS index + item metadata (small artifacts)
│
├── data/
│   ├── processed/              # Cleaned CSVs / embeddings
│   └── interim/                # Intermediate data (ignored in git)
│
├── requirements.txt            # Dependencies for Streamlit Cloud
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files and datasets
```

---

## 📈 Example Queries

| Mode | Example Input | Output |
|------|----------------|--------|
| Free-text | melancholic acoustic ballad | recommends soft, minor-key acoustic tracks |
| Free-text | uplifting 90s rock with high energy | energetic guitar-driven songs |
| Similar-song | Select: Freedom 15 — !!! | finds songs with similar lyrical tone and tempo |

---

## 🧰 Requirements

| Library | Version |
|----------|----------|
| Python | 3.10 |
| Streamlit | ≥1.37 |
| sentence-transformers | ≥3.0 |
| FAISS | ≥1.7 |
| pandas, numpy, scikit-learn | latest |

All dependencies are listed in requirements.txt.

---

## 👨‍💻 Author

**Iyanuoluwa Enoch Oke**  
🔗 GitHub: [Iyanuoluwa007](https://github.com/Iyanuoluwa007)  
🔗 LinkedIn: [Iyanuoluwa Enoch Oke](https://www.linkedin.com/in/iyanuoluwa-enoch-oke/)

## Licence and attribution

Released under the MIT Licence. See [LICENSE](LICENSE).

Copyright (c) 2026 Oke Iyanuoluwa Enoch.

You are free to use, modify and build on this work. The licence asks one thing
in return, and it is not optional: keep the copyright notice and the licence
text with any copy or substantial portion of the software. That notice is how
the work stays credited to its author.

If you fork this, publish something derived from it, or use it in a product,
paper or demo, please credit it visibly and link back:

> Built on [HarmonyAI](https://github.com/Iyanuoluwa007/HarmonyAI) by Oke Iyanuoluwa Enoch.

The MIT Licence covers this repository own code only. Any third-party data,
pretrained weights, papers or assets that may be present remain the property of
their owners and are subject to their own terms.