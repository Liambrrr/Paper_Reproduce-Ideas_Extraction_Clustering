# Paper_Reproduce-Ideas_Extraction_Clustering

## Overview
Non-LLM part(step2-8): Replication of paper *Fine-grained Main Ideas Extraction and Clustering of Online Course Reviews*.

## Quick Start
### Environment Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
python step2_reviews_filtered.py

python step3_semantic_units.py #run this
python step3_mintoken_version.py #or this

python step4_embeddings.py
python step5_umap.py
python step6_hdbscan.py
python step7_weighted_reps.py
python step8_make_table.py
python step9_prepare_topicgpt.py
```

## Results

### Non-LLM part(replication):
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

## Reference
Xiao, C., Shi, L., Cristea, A., Li, Z., & Pan, Z. (2022). Fine-grained Main Ideas Extraction and Clustering of Online Course Reviews. In M. M. Rodrigo, N. Matsuda, A. I. Cristea, & V. Dimitrova (Eds.), Artificial Intelligence in Education (pp. 294-306). Springer, Cham. https://doi.org/10.1007/978-3-031-11644-5_24