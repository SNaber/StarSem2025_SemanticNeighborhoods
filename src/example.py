from semantic_space_analyzer import SemanticSpaceAnalyzer

# Load analyzer and data
ex = SemanticSpaceAnalyzer("../data/paper_spaces.pkl")
spaces = list(ex.data.keys())

# Define target sets
target_files = [
    "../data/targets_abstract.txt",
    "../data/targets_concrete.txt"
]
target_labels = ["abstract", "concrete"]

# Compute NAS matrices for each target set
results = {}
for target_file, label in zip(target_files, target_labels):
    matrices = ex.compare_spaces_nas_matrix(
        target_file=target_file,
        spaces=spaces,
        k_values=[10, 25, 50, 100],
        background_size=len(ex.vocab),
        title_suffix=label,
    )

    # Wrap each matrix under {"nas": ...} for compatibility with plotting functions
    results[label] = {k: {"nas": m} for k, m in matrices.items()}

# Plot ΔNAS heatmaps between abstract and concrete
ex.plot_difference_heatmaps(
    results,
    comparisons=[("concrete", "abstract")],
    spaces=spaces,
    k_values=[10, 25, 50, 100]
)

# Plot NAS alignment profiles
ex.plot_alignment_profiles_all(
    target_files=target_files,
    target_labels=target_labels,
    spaces=spaces,
    background_size=len(ex.vocab),
    max_k=100,
    bandwidth=10,
    step=5
)
