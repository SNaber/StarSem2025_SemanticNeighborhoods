import json
import numpy as np
from tqdm import tqdm


def _try_import(module_name, package_name=None):
    """Import a module on demand, with a clear error if missing."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg = package_name or module_name
        raise ImportError(
            f"The module '{pkg}' is required for this function. "
            f"Install it with 'pip install {pkg}'."
        )


# ------------------------------------------------------------------
# SentenceTransformer embeddings
# ------------------------------------------------------------------
def build_sentence_kNN(self, model_name, sentence_dict_file, name, n_kNN=10,
                       batch_size=512, normalize=True, device="cuda"):
    """Build kNN space from SentenceTransformer sentence embeddings."""
    SentenceTransformer = _try_import("sentence_transformers").SentenceTransformer
    with open(sentence_dict_file, "r", encoding="utf-8") as f:
        sentence_dict = json.load(f)

    target_keys = list(sentence_dict.keys())
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    embeddings = []
    for word in tqdm(target_keys, desc=f"Encoding sentences ({name})"):
        sentences = sentence_dict[word]
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=normalize)
            all_embs.append(embs)
        avg_emb = np.mean(np.vstack(all_embs), axis=0)
        embeddings.append(avg_emb)

    embeddings = np.vstack(embeddings)
    self._build_knn_space(embeddings, target_keys, target_keys, name, n_kNN, normalize=normalize)
    print(f"Added SentenceTransformer embeddings: {name}")


# ------------------------------------------------------------------
# Image embeddings
# ------------------------------------------------------------------
def build_image_kNN(self, embedding_file, target_file, name, n_kNN, limit=False, limit_n=1):
    """Build kNN space from precomputed image embeddings."""
    torch = _try_import("torch")
    data = torch.load(embedding_file, weights_only=False)
    keys = list(data.keys())
    embs = []
    for key in keys:
        all_embs = [torch.flatten(torch.from_numpy(e.astype(np.float32))).numpy() for e in data[key]]
        if limit:
            all_embs = all_embs[:limit_n]
        embs.append(np.mean(all_embs, axis=0))
    embeddings = np.vstack(embs)
    target_keys = self._load_target_keys(target_file)
    self._build_knn_space(embeddings, keys, target_keys, name, n_kNN)
    print(f"Added image embeddings: {name}")


# ------------------------------------------------------------------
# GloVe embeddings
# ------------------------------------------------------------------
def build_glove_kNN(self, vectors_file, target_file, name, n_kNN):
    """Build kNN space from GloVe vectors."""
    target_keys = self._load_target_keys(target_file)
    target_set = set(target_keys)
    words, vecs = [], []
    with open(vectors_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] in target_set:
                words.append(parts[0])
                vecs.append(np.array(parts[1:], dtype=float))
    self._build_knn_space(np.vstack(vecs), words, target_keys, name, n_kNN)
    print(f"Added GloVe embeddings: {name}")


# ------------------------------------------------------------------
# FastText embeddings
# ------------------------------------------------------------------
def build_fasttext_kNN(self, model_filename, target, name, n_kNN):
    """Build kNN space from FastText model."""
    fasttext = _try_import("fasttext")
    model = fasttext.load_model(model_filename)
    target_keys = self._load_target_keys(target)
    ft_vecs, ft_keys = [], []
    for w in tqdm(target_keys, desc=f"FastText ({name})"):
        try:
            ft_vecs.append(model.get_word_vector(w))
            ft_keys.append(w)
        except Exception:
            print(f"Skipping {w}: not in FastText vocabulary.")
    self._build_knn_space(np.vstack(ft_vecs), ft_keys, target_keys, name, n_kNN)
    print(f"Added FastText embeddings: {name}")


# ------------------------------------------------------------------
# Word2Vec embeddings
# ------------------------------------------------------------------
def build_word2vec_kNN(self, model_file, target_file, name, n_kNN):
    """Build kNN space from a Word2Vec model."""
    gensim = _try_import("gensim")
    model = gensim.models.Word2Vec.load(model_file)
    target_keys = self._load_target_keys(target_file)
    keys, vecs = [], []
    for w in tqdm(target_keys, desc=f"Word2Vec ({name})"):
        if w in model.wv:
            vecs.append(model.wv[w])
            keys.append(w)
    self._build_knn_space(np.vstack(vecs), keys, target_keys, name, n_kNN)
    print(f"Added Word2Vec embeddings: {name}")


# ------------------------------------------------------------------
# CountVector embeddings
# ------------------------------------------------------------------
def build_countvector_kNN(self, count_vector_space, target_file, name, n_kNN):
    """Build kNN space from count-based vectors stored in HDF5."""
    h5py = _try_import("h5py")
    target_keys = self._load_target_keys(target_file)
    with h5py.File(count_vector_space, "r") as hdf:
        keys, vecs = [], []
        for w in target_keys:
            if w in hdf:
                vecs.append(hdf[w][:])
                keys.append(w)
    self._build_knn_space(np.vstack(vecs), keys, target_keys, name, n_kNN)
    print(f"Added CountVector embeddings: {name}")
