import pickle
import json
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from embedding_loaders import (
    build_sentence_kNN,
    build_image_kNN,
    build_glove_kNN,
    build_fasttext_kNN,
    build_word2vec_kNN,
    build_countvector_kNN
)


class SemanticSpaceAnalyzer:
    """
    Analysis framework for comparing semantic neighborhoods
    across representation spaces (e.g., textual vs. visual models).
    """

    def __init__(self, data_filename=None):
        self.data_filename = data_filename
        self.data = {}
        self.vocab = []
        self.word_to_index = {}
        self.index_to_word = {}
        if data_filename:
            self.load(data_filename)
        else:
            print("No data file loaded — initialized empty structure.")

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------
    def save(self, filename: str):
        """
        Save semantic spaces to a binary pickle file (.pkl or .bin).
        Falls back to pickle serialization for other extensions.
        """
        if not self.data:
            raise ValueError("No data to save.")

        # Ensure vocab exists
        if not self.vocab:
            all_keys = set()
            for space in self.data.values():
                all_keys.update(space.keys())
            self.vocab = sorted(all_keys)
            self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
            self.index_to_word = {i: w for i, w in enumerate(self.vocab)}

        save_obj = {"vocab": self.vocab, "spaces": self.data}

        if not filename.endswith((".pkl", ".bin")):
            filename += ".pkl"

        try:
            with open(filename, "wb") as f:
                pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved binary semantic space to {filename} "
                  f"({len(self.vocab)} vocab entries, {len(self.data)} spaces).")
        except Exception as e:
            raise IOError(f"Error saving '{filename}': {e}")

    def load(self, filename: str):
        """
        Load semantic space data.
        Supports:
          - .pkl / .bin  → binary pickle format
          - .json        → legacy JSON formats
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        # Try binary pickle
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "vocab" in obj and "spaces" in obj:
                self.vocab = obj["vocab"]
                self.data = obj["spaces"]
                self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
                self.index_to_word = {i: w for i, w in enumerate(self.vocab)}
                print(f"Loaded binary semantic space "
                      f"({len(self.vocab)} vocab entries, {len(self.data)} spaces).")
                return
        except Exception:
            pass  # Fall back to JSON

        # Try JSON
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"File '{filename}' is not valid pickle or JSON: {e}")

        if "vocab" in data and "spaces" in data:
            # Compact JSON
            self.vocab = data["vocab"]
            self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
            self.index_to_word = {i: w for i, w in enumerate(self.vocab)}
            self.data = {
                name: {int(wid): [(int(nid), float(score)) for nid, score in neighs]
                       for wid, neighs in space.items()}
                for name, space in data["spaces"].items()
            }
            print(f"Loaded compact JSON ({len(self.vocab)} vocab entries, {len(self.data)} spaces).")
        else:
            # Legacy word-based JSON
            all_keys = {w for space in data.values() for w in space.keys()}
            self.vocab = sorted(all_keys)
            self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
            self.index_to_word = {i: w for i, w in enumerate(self.vocab)}
            self.data = {}
            for name, space in data.items():
                self.data[name] = {
                    self.word_to_index[w]: [
                        (self.word_to_index.get(n, -1), float(s))
                        for n, s in neighs if n in self.word_to_index
                    ]
                    for w, neighs in space.items()
                }
            print("Loaded legacy word-based JSON (converted to index format).")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _load_target_keys(target_file: str):
        """Load target concept keys from a .txt or .tsv file."""
        if target_file.endswith(".txt"):
            with open(target_file, "r", encoding="utf-8") as f:
                return [line.strip().split()[0] for line in f if line.strip()]
        elif target_file.endswith("strict.tsv"):
            with open(target_file, "r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="\t")
                return [row[0] for row in reader if row]
        elif target_file.endswith(".tsv"):
            tsv_data = pd.read_csv(target_file, sep="\t")
            if "Word" not in tsv_data.columns:
                raise KeyError(f"'Word' column not found in {target_file}")
            return tsv_data["Word"].tolist()
        else:
            raise ValueError("Target file must be in .txt or .tsv format.")

    def _build_knn_space(self, embeddings, keys, target_keys, name, n_kNN, normalize=True):
        """Construct a kNN dictionary using cosine similarity."""
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        if not self.vocab:
            self.vocab = keys
            self.word_to_index = {w: i for i, w in enumerate(keys)}
            self.index_to_word = {i: w for i, w in enumerate(keys)}

        sim_matrix = embeddings @ embeddings.T
        self.data[name] = {}
        for word in tqdm(target_keys, desc=f"Building kNN space: {name}", ncols=80):
            if word not in self.word_to_index:
                continue
            i = self.word_to_index[word]
            sims = sim_matrix[i]
            nearest = np.argsort(-sims)
            nearest = nearest[nearest != i][:n_kNN]
            self.data[name][i] = [(int(j), float(sims[j])) for j in nearest]
        print(f"Built kNN space '{name}' with {len(self.data[name])} entries.")

    # ------------------------------------------------------------------
    # NAS / NVS
    # ------------------------------------------------------------------
    def calculate_nas_nvs(self, target_file, space1, space2, n_start, k_end, background_size):
        """Compute Normalized Alignment Score (NAS) and NVS between two spaces."""
        target_words = self._load_target_keys(target_file)
        target_ids = [self.word_to_index[w] for w in target_words if w in self.word_to_index]
        if space1 not in self.data or space2 not in self.data:
            raise KeyError(f"Spaces '{space1}' or '{space2}' not found.")

        overlaps = []
        band_size = k_end - n_start + 1
        for wid in tqdm(target_ids, desc="Calculating NAS/NVS", ncols=80):
            if wid not in self.data[space1] or wid not in self.data[space2]:
                continue
            n1 = self.data[space1][wid][n_start - 1:k_end]
            n2 = self.data[space2][wid][n_start - 1:k_end]
            if not n1 or not n2:
                continue
            overlap = len(set(x[0] for x in n1) & set(x[0] for x in n2)) / band_size
            overlaps.append(overlap)

        if not overlaps:
            raise ValueError("No valid overlaps computed.")

        O_avg = np.mean(overlaps)
        sigma_O = np.std(overlaps)
        O_rand = band_size / (background_size - 1)
        NAS = (O_avg - O_rand) / (1 - O_rand) if (1 - O_rand) != 0 else np.nan
        sigma_max = np.sqrt(O_avg * (1 - O_avg))
        sigma_rand = np.sqrt(O_rand * (1 - O_rand))
        denom = max(sigma_rand, sigma_max - sigma_rand)
        NVS_signed = (sigma_rand - sigma_O) / denom if denom > 0 else 0
        return NAS, NVS_signed, overlaps

    # ------------------------------------------------------------------
    # Significance testing
    # ------------------------------------------------------------------
    @staticmethod
    def significance_test(values, n_permutations=10000, random_state=None):
        """Permutation-based significance test."""
        x = np.array(values, dtype=float)
        if x.size == 0:
            raise ValueError("Input 'values' cannot be empty.")
        rng = np.random.default_rng(random_state)
        observed_mean = np.mean(x)
        flips = rng.choice([-1, 1], size=(n_permutations, x.size))
        perm_means = np.mean(flips * x, axis=1)
        return np.mean(np.abs(perm_means) >= abs(observed_mean))

    # ------------------------------------------------------------------
    # Embedding builders
    # ------------------------------------------------------------------
    def calculate_static_sentence_kNN(self, *args, **kwargs): build_sentence_kNN(self, *args, **kwargs)
    def calculate_image_embedding_kNN(self, *args, **kwargs): build_image_kNN(self, *args, **kwargs)
    def calculate_glove_model_kNN(self, *args, **kwargs): build_glove_kNN(self, *args, **kwargs)
    def calculate_fasttext_kNN(self, *args, **kwargs): build_fasttext_kNN(self, *args, **kwargs)
    def calculate_word2vec_kNN(self, *args, **kwargs): build_word2vec_kNN(self, *args, **kwargs)
    def calculate_countvector_kNN(self, *args, **kwargs): build_countvector_kNN(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize_overlap_matrix(self, target_file, spaces, n_kNN_values,
                                 output_prefix="overlap", show=True):
        """Visualize pairwise neighborhood overlaps as heatmaps."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        target_words = self._load_target_keys(target_file)
        target_ids = [self.word_to_index[w] for w in target_words if w in self.word_to_index]

        for n_kNN in n_kNN_values:
            matrix = np.zeros((len(spaces), len(spaces)))
            for i, s1 in enumerate(spaces):
                for j, s2 in enumerate(spaces):
                    if s1 not in self.data or s2 not in self.data:
                        matrix[i, j] = np.nan
                        continue
                    overlaps = []
                    for wid in target_ids:
                        if wid not in self.data[s1] or wid not in self.data[s2]:
                            continue
                        n1 = [x[0] for x in self.data[s1][wid][:n_kNN]]
                        n2 = [x[0] for x in self.data[s2][wid][:n_kNN]]
                        overlap = len(set(n1) & set(n2)) / n_kNN
                        overlaps.append(overlap)
                    matrix[i, j] = np.mean(overlaps) if overlaps else 0

            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm",
                        xticklabels=spaces, yticklabels=spaces)
            plt.title(f"Neighborhood Overlap (n_kNN={n_kNN})")
            plt.xlabel("Space Y")
            plt.ylabel("Space X")
            fname = f"{output_prefix}_{n_kNN}.png"
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            print(f"Saved heatmap to {fname}")
            if show:
                plt.show()
            else:
                plt.close()
