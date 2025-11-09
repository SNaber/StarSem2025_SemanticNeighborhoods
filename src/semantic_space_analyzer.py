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
        except json.JSONDecodeError as e:
            raise ValueError(f"File '{filename}' is not valid pickle: {e}")

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
        for wid in target_ids:
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
    def compare_spaces_nas_matrix(self, target_file, spaces, k_values, background_size, title_suffix=None):
        """
        Compute NAS matrices between all given spaces for specified k values
        and save heatmaps.

        Parameters:
            target_file (str)
            spaces (list[str])
            k_values (list[int])
            background_size (int)
            title_suffix (str|None): e.g., "abstract" to show in the title
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        results = {}
        for k in k_values:
            nas_matrix = np.zeros((len(spaces), len(spaces)))
            for i, s1 in enumerate(spaces):
                for j, s2 in enumerate(spaces):
                    try:
                        nas, _, _ = self.calculate_nas_nvs(
                            target_file=target_file,
                            space1=s1,
                            space2=s2,
                            n_start=1,
                            k_end=k,
                            background_size=background_size,
                        )
                        nas_matrix[i, j] = nas if nas is not None else np.nan
                    except Exception:
                        nas_matrix[i, j] = np.nan

            results[k] = nas_matrix

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                nas_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                cbar=True,
                xticklabels=spaces,
                yticklabels=spaces,
                annot_kws={"fontsize": 12},
            )
            title = f"NAS – Top-{k}"
            if title_suffix:
                title += f" – {title_suffix}"
            plt.title(title, fontsize=18, fontweight='bold')
            plt.xlabel("Space Y", fontsize=14)
            plt.ylabel("Space X", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            fname = f"nas_matrix_top{k}" + (f"_{title_suffix}" if title_suffix else "") + ".png"
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved NAS matrix heatmap → {fname}")

        return results



    def plot_difference_heatmaps(self, results, comparisons, spaces, k_values):
        """
        Plot ΔNAS heatmaps for selected target label pairs using a global symmetric scale.
        Expects: results[target_label][k]['nas'] = matrix
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np


        all_diffs = []
        for k in k_values:
            for name_a, name_b in comparisons:
                mat_a = results[name_a][k]["nas"]
                mat_b = results[name_b][k]["nas"]
                all_diffs.append(mat_a - mat_b)
        global_min = min(np.nanmin(d) for d in all_diffs)
        global_max = max(np.nanmax(d) for d in all_diffs)
        max_abs = max(abs(global_min), abs(global_max))

        for k in k_values:
            for name_a, name_b in comparisons:
                mat_a = results[name_a][k]["nas"]
                mat_b = results[name_b][k]["nas"]
                diff = mat_a - mat_b


                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    diff,
                    annot=True,
                    fmt=".2f",
                    center=0,
                    cmap="RdBu_r",
                    vmin=-max_abs,
                    vmax=max_abs,
                    cbar=True,
                    xticklabels=spaces,
                    yticklabels=spaces,
                    annot_kws={"fontsize": 12},
                )
                plt.title(f"Δ NAS (Top-{k}) – {name_a} minus {name_b}", fontsize=18, fontweight='bold')
                plt.xlabel("Space Y", fontsize=14)
                plt.ylabel("Space X", fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                fname = f"diff_nas_top{k}_{name_a}_vs_{name_b}.png"
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved ΔNAS heatmap → {fname}")



    def plot_alignment_profiles_all(
        self,
        target_files,
        target_labels,
        spaces,
        background_size,
        max_k=100,
        bandwidth=10,
        step=5,
    ):
        """
        Plot NAS alignment profiles for all unique (space1, space2) pairs.
        Span increases with k (n_start=1, k_end=end), matching the original visuals.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        assert len(target_files) == len(target_labels), "Each target file must have a corresponding label"


        for i, space1 in enumerate(spaces):
            for j, space2 in enumerate(spaces):
                if j <= i:
                    continue  # skip self-pairs and duplicates

                plt.figure(figsize=(12, 6))

                for target_file, label in zip(target_files, target_labels):
                    nas_values, x_ticks = [], []

                    # span increases: [1 .. end], we advance 'end' with a sliding center step
                    for start in range(1, max_k - bandwidth + 2, step):
                        end = start + bandwidth - 1
                        try:
                            nas, _, _ = self.calculate_nas_nvs(
                                target_file=target_file,
                                space1=space1,
                                space2=space2,
                                n_start=1,          
                                k_end=end,          
                                background_size=background_size,
                            )
                            nas_values.append(nas)
                            x_ticks.append((start + end) // 2)
                        except Exception:
                            nas_values.append(np.nan)
                            x_ticks.append((start + end) // 2)


                    plt.plot(x_ticks, nas_values, marker="o", label=f"NAS – {label}", linewidth=2)

                plt.xlabel("Neighborhood Rank Center", fontsize=14)
                plt.ylabel("Score", fontsize=14)
                plt.title(f"Alignment Profiles: {space1} vs {space2}", fontsize=18, fontweight='bold')
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(True, which='both', linestyle='--', linewidth=0.6)
                plt.legend(fontsize=12)
                plt.tight_layout()

                out_path = f"alignment_profiles_{space1}_vs_{space2}.png"
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"Saved profile plot for {space1} vs {space2} → {out_path}")
