import streamlit as st
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ---------- Page setup ----------
st.set_page_config(page_title="HarmonyAI — Hybrid Recommender", page_icon="🎵", layout="wide")

# ---------- Resolve artifacts relative to this file ----------
APP_DIR = Path(__file__).parent.resolve()
ART_DIR = APP_DIR / "artifacts"

# ---------- Query encoder (rebuilt at runtime to avoid pickle issues) ----------
class HybridQueryEncoder:
    """
    hybrid = [ SUM(SBERT(lyrics), SBERT(meta)) ] + [ numeric ] + [ emotion ]
    query  = (w_lyrics + w_meta) * SBERT(text)  then zero-pad tail to match hybrid dim.
    """
    def __init__(self, model_name: str, w_lyrics: float = 0.6, w_meta: float = 0.2, tail_dim: int = 0):
        self.model = SentenceTransformer(model_name)
        self.w_lyrics = float(w_lyrics)
        self.w_meta = float(w_meta)
        self.tail_dim = int(max(0, tail_dim))

    def encode(self, text: str) -> np.ndarray:
        E = self.model.encode([text], normalize_embeddings=True).astype("float32")  # (1, text_dim)
        q_text = (self.w_lyrics + self.w_meta) * E                                  # (1, text_dim)
        tail = np.zeros((1, self.tail_dim), dtype="float32")
        q = np.hstack([q_text, tail])                                               # (1, hybrid_dim)
        return normalize(q).astype("float32")

# ---------- Load artifacts (items + FAISS) and rebuild encoder ----------
@st.cache_resource
def load_artifacts():
    req = ["items_50k.pkl", "faiss_ip_50k.index"]
    missing = [r for r in req if not (ART_DIR / r).exists()]
    if missing:
        st.error(f"Looked in: {ART_DIR.resolve()}")
        raise FileNotFoundError(f"Missing artifacts in {ART_DIR}: {missing}")

    items = pd.read_pickle(ART_DIR / "items_50k.pkl")
    index = faiss.read_index(str(ART_DIR / "faiss_ip_50k.index"))

    hybrid_dim = int(index.d)
    text_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim = SentenceTransformer(text_model).get_sentence_embedding_dimension()
    tail_dim = max(0, hybrid_dim - text_dim)  # because we SUMMED lyrics+meta

    enc = HybridQueryEncoder(text_model, w_lyrics=0.6, w_meta=0.2, tail_dim=tail_dim)
    return items, index, enc

items, index, enc = load_artifacts()

# ---------- Centered title (no extra caption/dims) ----------
st.markdown(
    "<h1 style='text-align:center; margin-top: 0;'>🎵 HarmonyAI — Hybrid Music Recommender</h1>",
    unsafe_allow_html=True,
)

# ---------- Sidebar preview (silent if no 'popularity') ----------
with st.sidebar:
    if "popularity" in items.columns and items["popularity"].notna().any():
        st.header("Quick peek")
        cols = [c for c in ["track_name", "artist", "album", "genre", "popularity"] if c in items.columns]
        st.dataframe(
            items.sort_values("popularity", ascending=False, na_position="last")[cols].head(10),
            use_container_width=True,
            hide_index=True,
        )
    # else: show nothing (no message)

# ---------- Sample titles expander (helps users discover content) ----------
if len(items):
    samples = items.sample(min(10, len(items)), random_state=42)[["track_name", "artist"]]
    sample_lines = [f"- {t} — {a}" for t, a in zip(samples["track_name"], samples["artist"])]
    with st.expander("Need ideas? Try some titles from your dataset"):
        st.markdown("\n".join(sample_lines))

# ---------- UI Controls ----------
mode = st.radio("Search mode", ["Free-text mood/lyrics", "Similar to a song (by title)"], horizontal=True)
k = st.slider("Number of recommendations", 5, 30, 10)

# ---------- Helpers ----------
def render_results(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)

    def mk_link(row):
        q = f"{row.get('track_name','')} {row.get('artist','')}".strip()
        url = f"https://www.google.com/search?q={q.replace(' ', '+')}+site%3Aopen.spotify.com"
        return f"[🔎 Spotify search]({url})"

    df["link"] = df.apply(mk_link, axis=1)
    cols = [c for c in ["track_name", "artist", "album", "genre", "score", "link"] if c in df.columns]
    st.dataframe(
        df[cols].style.format({"score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

def rec_by_text(q: str, k: int = 10) -> pd.DataFrame:
    qv = enc.encode(q)                    # (1, hybrid_dim)
    scores, ids = index.search(qv, k)     # (1, k)
    out = items.iloc[ids[0]].copy()
    out["score"] = scores[0]
    return out

# ---------- Modes ----------
if mode == "Free-text mood/lyrics":
    q = st.text_input("Describe what you want (e.g., 'uplifting 90s rock with high energy')")
    if st.button("Recommend") and q.strip():
        try:
            recs = rec_by_text(q.strip(), k)
            render_results(recs)
        except Exception as e:
            st.error(f"Recommendation error: {e}")

else:
    title = st.text_input("Type an exact/partial song title", placeholder="e.g., Freedom 15")
    matched = (
        items[items["track_name"].str.contains(title, case=False, na=False)].head(50)
        if title.strip()
        else items.head(0)
    )

    if matched.empty and title.strip():
        st.info("No matches yet — try fewer words or a different part of the title.")

    options = [""] + (matched["track_name"] + " — " + matched["artist"]).tolist()
    pick = st.selectbox("Pick a seed song", options=options)

    if st.button("Find similar") and pick:
        try:
            sel_title, sel_artist = pick.split(" — ", 1)
            row_idx = matched[(matched["track_name"] == sel_title) & (matched["artist"] == sel_artist)].index[0]

            seed_query = f"{items.loc[row_idx, 'track_name']} {items.loc[row_idx, 'artist']}"
            qv = enc.encode(seed_query)
            scores, ids = index.search(qv, k + 1)

            filtered = [i for i in ids[0] if i != row_idx][:k]
            recs = items.iloc[filtered].copy()

            rank = {i: r for r, i in enumerate(ids[0])}
            recs["score"] = [scores[0][rank[i]] for i in filtered]
            render_results(recs)
        except Exception as e:
            st.error(f"Seed search error: {e}")

# ---------- Footer (optional) ----------
st.markdown(
    "<div style='text-align:center; opacity:0.7; padding-top:12px;'>"
    "Built with SBERT + FAISS • HarmonyAI</div>",
    unsafe_allow_html=True,
)