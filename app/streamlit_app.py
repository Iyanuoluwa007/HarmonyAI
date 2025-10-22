import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Try FAISS, fallback to NumPy
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    faiss = None
    FAISS_OK = False

# ---------- Page setup ----------
st.set_page_config(page_title="HarmonyAI — Hybrid Recommender", page_icon="🎵", layout="wide")

APP_DIR = Path(__file__).parent.resolve()
ART_DIR = APP_DIR / "artifacts"

# ---------- Query encoder ----------
class HybridQueryEncoder:
    """Hybrid = (lyrics + meta embeddings) + numeric + emotion"""
    def __init__(self, model_name: str, w_lyrics: float = 0.6, w_meta: float = 0.2, tail_dim: int = 0):
        self.model = SentenceTransformer(model_name)
        self.w_lyrics = float(w_lyrics)
        self.w_meta = float(w_meta)
        self.tail_dim = int(max(0, tail_dim))

    def encode(self, text: str) -> np.ndarray:
        E = self.model.encode([text], normalize_embeddings=True).astype("float32")
        q_text = (self.w_lyrics + self.w_meta) * E
        tail = np.zeros((1, self.tail_dim), dtype="float32")
        q = np.hstack([q_text, tail])
        return normalize(q).astype("float32")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    items_path = ART_DIR / "items_50k.pkl"
    if not items_path.exists():
        st.error(f"Missing: {items_path}")
        raise FileNotFoundError(items_path)
    items = pd.read_pickle(items_path)

    text_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim = SentenceTransformer(text_model).get_sentence_embedding_dimension()

    faiss_path = ART_DIR / "faiss_ip_50k.index"
    if FAISS_OK and faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        hybrid_dim = int(index.d)
        tail_dim = max(0, hybrid_dim - text_dim)
        backend = ("faiss", index)
    else:
        emb_path = ART_DIR / "hybrid_emb_50k.npy"
        if not emb_path.exists():
            raise FileNotFoundError("Neither FAISS nor NumPy embedding matrix available.")
        mat = np.load(emb_path).astype("float32")
        mat = normalize(mat).astype("float32")
        hybrid_dim = mat.shape[1]
        tail_dim = max(0, hybrid_dim - text_dim)
        backend = ("numpy", mat)

    enc = HybridQueryEncoder(text_model, w_lyrics=0.6, w_meta=0.2, tail_dim=tail_dim)
    return items, enc, backend, hybrid_dim, text_dim, tail_dim

items, enc, backend, hybrid_dim, text_dim, tail_dim = load_artifacts()

# ---------- Centered title ----------
st.markdown(
    """
    <h2 style='text-align:center; margin-top: 0; color:#222;'>🎵 HarmonyAI — Hybrid Music Recommender</h2>
    """,
    unsafe_allow_html=True,
)

mode_label = "FAISS (cosine, ANN)" if backend[0] == "faiss" else "NumPy fallback (cosine, exact)"
st.caption(f"Search backend: **{mode_label}**")

# ---------- Sidebar preview ----------
with st.sidebar:
    if "popularity" in items.columns and items["popularity"].notna().any():
        st.markdown("#### Quick peek")
        cols = [c for c in ["track_name", "artist", "album", "genre", "popularity"] if c in items.columns]
        st.dataframe(
            items.sort_values("popularity", ascending=False, na_position="last")[cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

# ---------- Sample titles ----------
if len(items):
    samples = items.sample(min(10, len(items)), random_state=42)[["track_name", "artist"]]
    with st.expander("Need ideas? Try some titles from your dataset"):
        st.markdown("\n".join(f"- {t} — {a}" for t, a in zip(samples["track_name"], samples["artist"])))

# ---------- Controls ----------
mode = st.radio("Search mode", ["Free-text mood/lyrics", "Similar to a song (by title)"], horizontal=True)
k = st.slider("Number of recommendations", 5, 30, 10)

# ---------- Helpers ----------
def make_spotify_md(track: str, artist: str) -> str:
    q = f"{track} {artist}".strip()
    url = f"https://www.google.com/search?q={q.replace(' ', '+')}+site%3Aopen.spotify.com"
    return f"[🔎 Spotify search]({url})"

def render_results(df: pd.DataFrame):
    if df.empty:
        st.warning("No results found.")
        return
    for _, r in df.iterrows():
        st.markdown(
            f"""
            <div style='font-size:0.9rem; line-height:1.4; margin-bottom:0.7em;'>
                <strong>🎵 {r.get('track_name','?')}</strong> — <em>{r.get('artist','?')}</em><br>
                <span style='opacity:0.85;'>Album:</span> {r.get('album','?')} | <span style='opacity:0.85;'>Genre:</span> {r.get('genre','')} | 
                <span style='opacity:0.85;'>Score:</span> {r.get('score',0):.3f}<br>
                <a href="https://www.google.com/search?q={r.get('track_name','').replace(' ', '+')}+{r.get('artist','').replace(' ', '+')}+site%3Aopen.spotify.com" target="_blank" style="color:#1DB954;">🔗 Open on Spotify</a>
            </div>
            <hr style='opacity:0.2; margin:0.5em 0;'>
            """,
            unsafe_allow_html=True,
        )

def rec_by_text(q: str, k: int = 10) -> pd.DataFrame:
    qv = enc.encode(q)
    mode_name, obj = backend
    if mode_name == "faiss":
        scores, ids = obj.search(qv, k)
        out = items.iloc[ids[0]].copy()
        out["score"] = scores[0]
    else:
        mat = obj
        scores = (mat @ qv[0])
        idx = np.argsort(-scores)[:k]
        out = items.iloc[idx].copy()
        out["score"] = scores[idx]
    return out

# ---------- Search modes ----------
if mode == "Free-text mood/lyrics":
    q = st.text_input("Describe what you want (e.g., 'uplifting 90s rock with high energy')")
    if st.button("Recommend") and q.strip():
        recs = rec_by_text(q.strip(), k)
        render_results(recs)

else:
    title = st.text_input("Type an exact or partial song title", placeholder="e.g., Freedom 15")
    matched = items[items["track_name"].str.contains(title, case=False, na=False)].head(50) if title.strip() else items.head(0)
    pick = st.selectbox("Pick a seed song", options=[""] + (matched["track_name"] + " — " + matched["artist"]).tolist())
    if st.button("Find similar") and pick:
        sel_title, sel_artist = pick.split(" — ", 1)
        row_idx = matched[(matched["track_name"] == sel_title) & (matched["artist"] == sel_artist)].index[0]
        seed_query = f"{items.loc[row_idx, 'track_name']} {items.loc[row_idx, 'artist']}"
        recs = rec_by_text(seed_query, k + 1)
        mask = ~((recs["track_name"] == sel_title) & (recs["artist"] == sel_artist))
        recs = recs[mask].head(k)
        render_results(recs)

# ---------- Footer (optional) ----------
st.markdown(
    "<div style='text-align:center; opacity:0.7; padding-top:12px;'>"
    "Built with SBERT + FAISS • HarmonyAI</div>",
    unsafe_allow_html=True,
)
