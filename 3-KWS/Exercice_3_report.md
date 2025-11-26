# Pattern-Recognition 
## Exercice 3

In this exercise we implement a query-by-example keyword spotting system for historical handwritten documents, following the instructions of the Pattern Recognition course. We work with the **George Washington dataset**, which provides scanned pages of George Washington’s letters together with polygon annotations for each word and their text transcriptions. Starting from these page images and word polygons, the goal is to retrieve all other instances of a given keyword image using a DTW-based baseline and a learned similarity model, and to compare their performance.

## Results and Analysis

### DTW baseline

For the DTW-only keyword spotting system, we evaluate retrieval performance by treating each validation occurrence of a keyword as a query and ranking all training words by DTW distance. Over **69 query instances**, the DTW baseline achieves a mean Average Precision:

- **DTW mAP:** **0.126** (over 69 queries)

This indicates that, on average, relevant word instances tend to appear somewhat above random in the ranking, but are still often mixed with visually similar but incorrect words.

A small random-keyword experiment using **Precision@5** illustrates this behaviour:

- Keywords tested: `g-r-e-a-t-l-y`, `C-a-p-t-a-i-n`, `w-a-n-t-e-d`  
- Per-keyword Precision@5:
  - `g-r-e-a-t-l-y`: 0.00 (0/5 correct)
  - `C-a-p-t-a-i-n`: 0.40 (2/5 correct)
  - `w-a-n-t-e-d`: 0.00 (0/5 correct)
- **Average Precision@5 (DTW baseline):** **0.13**

Qualitatively, the DTW baseline sometimes retrieves the correct word multiple times in the top-5 (e.g. “Captain”), but for other keywords it returns only visually similar long words that are **not** the correct transcription.

---

### Learned similarity model

The learned similarity model is trained on pairs of word images, using DTW-based features (distance and length features) as input to a logistic regression classifier. On the pairwise validation set, the classifier achieves:

- **Train ROC-AUC:** 0.703  
- **Validation ROC-AUC:** 0.795  
- **Validation accuracy @ 0.5:** 0.774 (on a balanced set of positive/negative pairs)

This shows that, on individual pairs, the model can moderately well distinguish “same word” from “different word”.

However, when the learned model is used for full retrieval (ranking all training words for each query by predicted similarity score), its performance is noticeably weaker than the DTW baseline:

- **Learned model mAP:** **0.058** (over 69 queries)

In a random-keyword Precision@5 experiment with 10 keywords, we obtain:

- Keywords tested: `M-a-j-o-r`, `s_GW`, `g-r-e-a-t-l-y`, `l-e-f-t`, `w-a-n-t-e-d`, `t-h-i-n-g-s`, `V-i-r-g-i-n-i-a`, `t-w-e-l-v-e`, `s_et-c-s_pt`, `O-r-d-e-r-s`
- Per-keyword Precision@5 (selected examples):
  - `V-i-r-g-i-n-i-a`: 0.20 (1/5 correct)
  - `O-r-d-e-r-s`: 0.20 (1/5 correct)
  - Most other keywords: 0.00 (0/5 correct)
- **Average Precision@5 (learned model):** **0.04** over 10 queries

Thus, while the classifier works reasonably well on balanced pair classification, the learned similarity scores do not translate into strong rankings when we must pick only a few relevant words among thousands of candidates.

---

### Comparison and qualitative behaviour

A direct comparison of the main retrieval metrics is:

| Method                  | mAP (69 queries) | Avg Precision@5 (small sample) |
|-------------------------|------------------|--------------------------------|
| DTW baseline            | **0.126**        | **0.13** (3 random keywords)   |
| Learned similarity model| **0.058**        | **0.04** (10 random keywords)  |

Overall, the **DTW baseline** provides better retrieval performance than the learned similarity model in this setup. Both methods tend to return visually similar long words (same writer, similar stroke patterns), but the simple DTW distance is more effective at keeping true matches higher in the ranking.

These results suggest that, with the current feature set and training procedure, the learned model does **not** yet improve over raw DTW for query-by-example keyword spotting on this dataset. In the subsequent discussion section, we comment on potential reasons (e.g. limited feature expressiveness, strong class imbalance in retrieval) and possible directions for improvement.

---

## Discussion

The main objective of this exercise was to build a query-by-example keyword spotting system for historical handwritten documents, starting from page images and word polygons, and to compare a pure DTW-based baseline with a learned similarity model. Our results show that the DTW baseline achieves a modest but clearly non-random retrieval performance (mAP ≈ 0.126), whereas the learned similarity model reaches a lower mAP (≈ 0.058) and weaker Precision@5 in our random keyword experiments. In practical terms, this means that, given a single query example, the system sometimes retrieves correct instances among the top-ranked candidates, but it frequently confuses visually similar but textually different words.

The learned similarity model behaves somewhat paradoxically: on the balanced word-pair validation set it attains a reasonably good ROC-AUC (~0.80) and accuracy (~0.77), indicating that it can distinguish “same word” vs “different word” when positives and negatives are sampled in a controlled way. However, when we switch to the full retrieval scenario, where only a handful of relevant words must be found among thousands of candidates, the learned scores do not provide better rankings than plain DTW. This suggests that the current feature representation and training setup are not strong enough to separate relevant words from many “hard negatives” that are visually similar in George Washington’s handwriting.

Several limitations help explain these findings. First, the sliding-window features are intentionally simple (ink density, vertical profiles, transitions, and basic contour statistics), which captures general stroke structure but not fine-grained letter shapes. In a single-writer setting, many different words share very similar global appearance under such features. Second, the retrieval task is highly imbalanced: each query has only a fe


::contentReference[oaicite:0]{index=0}
