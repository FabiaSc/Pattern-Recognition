# Pattern-Recognition 
## Exercice 3

## Introduction

This project is carried out in the context of Exercise 3 of the Pattern Recognition course, which focuses on **query-by-example keyword spotting** in historical handwritten documents. The goal is to build a system that, given a single example image of a word (the query), can automatically retrieve other instances of the same word from a collection of page images. We work with the **George Washington dataset**, which contains scanned letters of George Washington along with polygon annotations for each word, text transcriptions, and predefined train/validation splits and keyword lists.   

Our main objective is to design and evaluate a keyword spotting pipeline based on **sliding-window features** and **Dynamic Time Warping (DTW)**, as suggested in the course material. Word images are first extracted from the page scans using the provided word polygons, preprocessed, and converted into sequences of feature vectors by sliding a vertical window across each word. DTW is then used to compute a dissimilarity measure between two word sequences, which serves as a baseline similarity model for retrieval. On top of this, we also implement a **learned similarity model**: we build a dataset of word pairs, extract DTW-based pair features, and train a logistic regression classifier to predict whether two word images represent the same transcription.

The notebook implements this full pipeline and evaluates it on a query-by-example keyword spotting task: query word instances are taken from the validation split, and retrieval is performed over the training split. Performance is measured using information-retrieval metrics such as Precision@K and mean Average Precision (mAP), allowing us to compare the DTW baseline with the learned similarity model and to discuss the strengths and limitations of our approach.

---

## Data Overview

The experiments in this project are based on the **George Washington dataset**, a collection of scanned letters written by George Washington. The dataset is provided in a structured format that includes both the page images and detailed annotations for each word occurrence. For this exercise, we use the official train/validation split and a predefined list of keywords.

The data is organised into the following main components:

- **Page images (`images/*.jpg`)**  
  Grayscale  scans of full manuscript pages. All word images used in our experiments are cropped from these page-level images.

- **Word locations (`locations/*.svg`)**  
  For each page, an SVG file specifies polygon outlines for all word segments. These polygons provide precise spatial boundaries that we use to crop individual word images from the corresponding page image.

- **Transcriptions (`transcription.tsv`)**  
  A character-based transcription file that maps each annotated word instance (identified by a unique `word_id`) to its textual content (the word string), along with page and line identifiers.

- **Keyword list (`keywords.tsv`)**  
  A list of words that appear in both the training and validation splits. These serve as the target “keywords” for the query-by-example keyword spotting task.

- **Splits (`train.tsv`, `validation.tsv`)**  
  Files that define which word instances belong to the **training** and **validation** sets. In our implementation, these are merged with the transcription information into a single `words_df` table containing, for each word instance, its `word_id`, `doc_id`, `line_id`, transcription, and split label (`train` or `val`).

In the notebook, we first load all TSV files and assemble a unified dataframe of word instances. Each word can then be associated with:
1. A page image (via `doc_id`),
2. A polygon describing its location on that page (via the SVG `word_id`),
3. Its ground-truth transcription and split assignment.

This structured representation provides the basis for subsequent steps: cropping word images, extracting sliding-window features, and evaluating keyword spotting performance using the train/validation split and the provided keyword list.

---

## Exploratory Data Analysis

Detail the exploratory analysis performed on the dataset. This should include any key findings from visualizing or summarizing the data. Discuss trends, patterns, or anomalies discovered during EDA. For instance, you might highlight distributions of important variables, correlations between features, or any interesting group comparisons.

*(Placeholder: Summarize major insights from EDA, such as distribution characteristics, outliers, correlations, etc. Include references to any figures or tables from the analysis as needed.)*

---

## Methodology

Explain the approach and methods used to analyze the data and solve the problem. If this project involves building models or performing statistical tests, describe those methods here. Outline any algorithms, techniques, or tools applied. For example, if a machine learning model was used, specify the type of model and why it was chosen. If no modeling was done, describe the analytical techniques or workflow steps (such as hypothesis testing, grouping, filtering, etc.).

- **Techniques Used:**  
  *(Placeholder: e.g., regression, classification, clustering, statistical analysis methods.)*

- **Tools/Libraries:**  
  *(Placeholder: e.g., pandas for data manipulation, scikit-learn for modeling, etc.)*

- **Rationale:**  
  *(Placeholder: why these methods or models were chosen.)*

---

## Results and Analysis

Present the results of the analysis or modeling. Describe what was found and interpret the outcomes in the context of the problem. If a model was built, report its performance (using appropriate metrics) and what those metrics mean for the problem. Highlight any important patterns or discoveries. Use bullet points or tables to clearly illustrate results if needed, and explain them in a clear, concise manner.

*(Placeholder: Provide key results such as model accuracy, error rates, significant correlations, or other findings. Explain the significance of these results and how they address the project objectives.)*

---

## Discussion

Discuss the implications of the results. This section should interpret how the findings answer the initial questions or solve the stated problem. Consider what the results mean in practical terms. Are there limitations to the analysis or data that might affect the conclusions? Also, mention any unexpected findings and potential reasons behind them. If relevant, compare the results with initial hypotheses or with findings from similar analyses.

*(Placeholder: Elaborate on the meaning of the results, their reliability, and any insights or real-world implications. Discuss limitations and surprises discovered during the analysis.)*

---

## Conclusion

Summarize the key findings of the project and reflect on whether the objectives were met. Recap the main points from the analysis and what conclusions can be drawn. Additionally, suggest possible next steps or recommendations for future work. This could include improving the model, collecting more data, exploring new questions that arose, or applying the insights in a practical setting.

*(Placeholder: Final summary of findings, conclusions drawn, and recommendations for future improvements or further analysis.)*

::contentReference[oaicite:0]{index=0}
