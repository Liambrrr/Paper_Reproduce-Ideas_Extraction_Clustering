# Paper_Reproduce-Ideas_Extraction_Clustering

## Overview
- Non-LLM part(step2-8): Replication of paper *Fine-grained Main Ideas Extraction and Clustering of Online Course Reviews*.

- LLM part(step9-12): adopted TopicGPT from paper *TopicGPT: A Prompt-based Topic Modeling Framework*.

## Quick Start
### Environment Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Paper Replication Part
python step2_reviews_filtered.py

python step3_semantic_units.py # not recommended version since output is not comprehensible
python step3_mintoken_version.py #or this

python step4_embeddings.py
python step5_umap.py
python step6_hdbscan.py
python step7_weighted_reps.py
python step8_make_table.py

# LLM part
python step9_prepare_topicgpt.py
python step10_run_topicgpt.py
python step11_coherence.py
python step12_irbo.py
```
## Implementation

### Step Instruction
Step 2 - Data preprocessing

Filter according to CourseId: Include either “machine”, or both “data” and “science” in the course names (did not select “machine-design1” course). Course names should be: “big-data-machine-learning, build-data-science-team, data-science-course, data-science-project, datasciencemathskills, executive-data-science-capstone, genomic-data-science-project, intro-data-science-programacion-estadistica-r, machine-learning, machine-learning-data-analysis, practical-machine-learning, real-life-data-science”. Total 12 courses, 9980 unique comments with 246,290 tokens.

Step 3 - Phase-level segmentation

Split each review comment into fine-grained “long phrases or short sentences” instead of using whole documents. First, segment text into short sentences using standard delimiters (e.g., ., ?, !). 
Then, further split these sentences on stopwords to obtain long phrases, but customize the stopword list so that important opinion/negation words such as “don’t”, “not”, and “shouldn’t” are kept (i.e., removed from the stopword list). Use all resulting long phrases/short sentences as the basic semantic units for the rest of the pipeline.

Step 4 - Sentence embedding with Sentence-Transformers

Encode each phrase with two Sentence-Transformers models: all-mpnet-base-v2 and all-MiniLM-L6-v2. Run the full pipeline separately for each model. For each phrase, obtain its embedding vector using the default settings of the Sentence-Transformers library. Store the resulting embedding matrices (one per model) for later dimensionality reduction and clustering.

Step 5 - Dimensionality reduction with UMAP

For each embedding set (per model), apply UMAP to reduce the embedding dimensionality before clustering. Use cosine distance as the metric and keep other UMAP hyperparameters at their default values unless otherwise required. Generate reduced embeddings for the following target dimensions, matching Figure 2 in the paper: 2, 3, 5, 10, 20, 50, and 200; also keep the original (no reduction) case with 768 dimensions as a baseline.

Step 6 - Clustering with HDBSCAN

The paper mainly discusses all-mpnet-base-v2, so use that embedding model.
Focus only on the 5-dimensional and 10-dimensional UMAP outputs.
For each of those, use HDBSCAN with default parameters. Fit it on 5D UMAP embeddings, then on the 10D UMAP embeddings.
We’ll use cluster labels and outlier scores
For both 5D and 10D, retain only the top 10 largest clusters to replicate Table 1.

Step 7 - Compute weighted centroids and representative phrases

For each of the top-10 clusters in the 5D and 10D settings, compute a single “key idea” phrase using NumPy and scikit-learn. Put the phrase embeddings for that cluster into a NumPy array, and the corresponding HDBSCAN outlier scores into another array. 
Set the weighting parameter α = 1, and for each phrase create a weight equal to 1 minus its outlier score (clip negative values to 0). Use NumPy to compute a weighted average of the embeddings with these weights to obtain one centroid embedding for the cluster. 
Then use cosine_similarity from sklearn.metrics.pairwise to measure the cosine similarity between this centroid and every phrase embedding in the cluster, and select the phrase with the highest similarity as the representative “key idea” phrase for that cluster.

Step 8 - Replicate Table 1 (summary of main aspects)

Using the top-10 clusters and their representative phrases (for both 5D and 10D), inspect each representative phrase and a few additional phrases in the same cluster to understand what students are talking about. 
Based on this reading, manually assign a short aspect label to each cluster (e.g., math background, course structure) and record whether that aspect appears in the 5D run, the 10D run, or both. 
Finally, create a table similar to Table 1 in the paper that lists these aspect labels, an example representative phrase for each, and the corresponding dimensionality (5D, 10D, or both), showing that your replication recovers the main themes reported by the authors.

Step 9 - TopicGPT preparation

Use this repository (they also have a Python library): https://github.com/chtmp223/topicGPT
For more information with their paper, if needed: https://aclanthology.org/2024.naacl-long.164.pdf
Use the same phrase-level units as in the baseline pipeline (output of Step 3). Each phrase should be a short sentence or long phrase after preprocessing (lowercasing, cleaned text, customized stopwords). Create a single text file or list where each line (or entry) is one phrase. This will be the input corpus for TopicGPT so that comparisons with the baseline are fair.
If that doesn’t work, you can also try with a full review directly for more context. 
Use the Llama-3.1-405B model for this task.

Step 10 - Run TopicGPT

Use a TopicGPT implementation and feed it the phrase corpus from Step 9. 
Run TopicGPT with Llama-3.1-405B and keep other TopicGPT parameters at their default values.
During the run, save: (a) the topic labels/descriptions produced by TopicGPT, (b) the list of top words (e.g., top 10 words) for each topic, and (c) the assignment of phrases to topics (if provided).

Step 11 - Compute coherence (C_v)

Compute topic coherence (C_v) for both the baseline clusters and the TopicGPT topics, so that they are evaluated under the same metric. Use a standard topic coherence implementation (for example, Gensim’s CoherenceModel with coherence='c_v'). 
For each model and setting:
Baseline: for 5D and 10D, take the top 5 and top 10 largest clusters and extract their top words (e.g., using c-TF-IDF over phrases in each cluster).
TopicGPT: use the top words that TopicGPT outputs.Compute C_v and then report the average C_v per setting (baseline-5, baseline-10, TopicGPT).

Step 12 - Compute IRBO and compare with baseline

Using the same top-word lists, compute IRBO (topic redundancy/diversity) for both the baseline clusters and the TopicGPT topics. Use the same IRBO implementation we used in the MOOC urgent-posts replication (or an equivalent implementation) and keep its parameters at their default values. 
For each setting (baseline-5, baseline-10, TopicGPT), feed the top words for all topics/clusters into the IRBO function and record the resulting IRBO score. 
Finally, compare C_v and IRBO across the four conditions to see whether TopicGPT produces more coherent and less redundant topics than the original phrase-level clustering baseline.

### Notice
Step 3 introduces limitation on the minimal number of tokens, otherwise output by step4-6 is not comprehensible.

Step 10 terminates with 18059/22974 phrases processed due to budget limits. Therefore LLM part work is based on 79% processed phrases.

Step 11 computes baseline-10D top 10 words with the 9th being filtered out since 9th is in Chinese and cannot be handled with alphabetic tokenizer.
## Results

### Non-LLM part(table replication):
| top-n | 5-d Weighted centroid | 5-d Cluster label (interpreted) | 10-d Weighted centroid | 10-d Cluster label (interpreted) |
|---:|---|---|---|---|
| 1 | good programming assignments | **challenging programming exercises** | good programming assignments | **good programming exercises** |
| 2 | math content obviously must lot well explained | **hard math content** | math content obviously must lot well explained | **hard math content** |
| 3 | excelente curso muy bueno explicacion sencilla de las cosas | excellent course | first disappointed homework needs done matlab octave instead | **Octave and Matlab** |
| 4 | course using matlab octave really helpful | **Octave and Matlab** | es el mejor curso de todos gracias por compartir tanto conocimiento | good way of teaching |
| 5 | great course | **great course** | great course | **great course** |
| 6 | helps understand implementation application basics basic yet powerful algorithms | algorithm | andrew expert great teacher | **Andrew good teaching** |
| 7 | andrew expert great teacher | **Andrew good teaching** | excellent course | **execellent course** |
| 8 | good overview data science | good intro to data science | one best courses ever coursera | **best course in coursera** |
| 9 | excellent course | **execellent course** | 老师讲的通俗易懂，很利于入门机器学习。 | good way of teaching |
| 10 | one best courses coursera | **best course in coursera** | taught not sufficient quizzes | quiz questions |

### LLM part:
#### Coherence
| setting             | source    | k  | avg_c_v  |
|---------------------|-----------|----|----------|
| baseline_5D_top5    | baseline  | 5  | 0.5405   |
| baseline_5D_top10   | baseline  | 10 | 0.4689   |
| baseline_10D_top5   | baseline  | 5  | 0.5444   |
| baseline_10D_top10  | baseline  | 10 | 0.4731   |
| topicgpt            | topicgpt  | 1  | 0.4684   |

#### IRBO
| key                 | value   |
|---------------------|---------|
| baseline_5d_top5    | 0.9661  |
| baseline_5d_top10   | 0.9699  |
| baseline_10d_top5   | 0.9661  |
| baseline_10d_top10  | 0.9766  |
| baseline5_avg       | 0.9661  |
| baseline10_avg      | 0.9732  |
| topicgpt_irbo       | 0.9622  |

## Reference
Xiao, C., Shi, L., Cristea, A., Li, Z., & Pan, Z. (2022). Fine-grained Main Ideas Extraction and Clustering of Online Course Reviews. In M. M. Rodrigo, N. Matsuda, A. I. Cristea, & V. Dimitrova (Eds.), Artificial Intelligence in Education (pp. 294-306). Springer, Cham. https://doi.org/10.1007/978-3-031-11644-5_24

Pham, C. M., Hoyle, A., Sun, S., Resnik, P., & Iyyer, M. (2024). TopicGPT: A prompt-based topic modeling framework. arXiv. https://arxiv.org/abs/2311.01449