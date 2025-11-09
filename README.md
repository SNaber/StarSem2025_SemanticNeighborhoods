# SemanticNeighborhoods

Core code and data for:

> **Evaluating Textual and Visual Semantic Neighborhoods of Abstract and Concrete Concepts**  
> Sven Naber, Diego Frassinelli, Sabine Schulte im Walde  
> *∗SEM / StarSem 2025*  
> [[Paper]](https://aclanthology.org/2025.starsem-1.11/) • [[Dataset on Hugging Face]](https://huggingface.co/datasets/SvenN/StarSem2025_SemanticNeighborhoods)

---

## Contents

- `semantic_space_analyzer.py` – main analysis class for computing semantic neighborhoods and NAS metric  
- `embedding_loaders.py` – model-specific embedding loaders (textual and visual)  
- `example.py` – minimal example script to reproduce NAS, ΔNAS, and alignment profile visualizations  
- `data/targets_*.tsv` – target word lists for abstract, mid, and concrete sets  
- `data/background_set.tsv` – background vocabulary set  
- `img/*` – plots and visualizations from the paper  

---

## Notes

- The precomputed n-knn Neighborhoods of semantic spaces used in the paper can be downloaded from  
  [**Hugging Face: SvenN/StarSem2025_SemanticNeighborhoods**](https://huggingface.co/datasets/SvenN/StarSem2025_SemanticNeighborhoods).  
  Example usage:
  ```python
  ex = SemanticSpaceAnalyzer("paper_spaces.pkl")
