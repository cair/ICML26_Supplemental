# Supplementary Material: Viral Genome Sequence Classification with GraphTM

This document provides supplementary experimental results for viral genome sequence classification using the **Graph Tsetlin Machine (GraphTM)**.  
All reported results are **mean ± standard deviation over 5 independent runs**, unless stated otherwise.

---

## Experimental Setup

All experiments were conducted using the GraphTsetlinMachine benchmarking framework with the following configuration unless specified:

- **Model**: Graph Tsetlin Machine (GraphTM)
- **Clauses**: 2000
- **T**: 2000  
- **s**: 1.0  
- **Max included literals**: 200  
- **Number of symbols**: 64 (codons)  
- **Hypervector size**: 512  
- **Message hypervector size**: 512 / 2048
- **Depth**: 2     

### Virus Classes
- SARS-CoV-2  
- Influenza A virus  
- Dengue virus  
- Zika virus  
- Rotavirus  

## Dataset Extraction and Preprocessing

The viral genome dataset is distributed as a **ZIP archive**.

Before running the experiments, extract the dataset:

```bash
cd Dataset
unzip Sequence_Dataset.zip
```
---
## 1. Impact of Clause Count on Classification Accuracy

This experiment evaluates the impact of varying the **number of clauses** under increasing class complexity.  
The model was trained for **10 epochs**.

### Class Configurations
- **2 classes**: SARS-CoV-2, Influenza A virus  
- **3 classes**: SARS-CoV-2, Influenza A virus, Dengue virus  
- **4 classes**: SARS-CoV-2, Influenza A virus, Dengue virus, Zika virus  
- **5 classes**: SARS-CoV-2, Influenza A virus, Dengue virus, Zika virus, Rotavirus  

### Table 1: Scalability Analysis with Increasing Clause Count

| Classes | 500 Clauses | 700 Clauses | 1000 Clauses | 2000 Clauses |
|-------:|------------:|------------:|-------------:|-------------:|
| 2 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 3 | 95.09 ± 0.53 | 96.80 ± 0.71 | 96.66 ± 0.54 | 97.31 ± 0.29 |
| 4 | 89.16 ± 1.69 | 91.96 ± 0.42 | 92.64 ± 0.83 | 94.67 ± 0.33 |
| 5 | 90.52 ± 1.07 | 92.72 ± 0.68 | 93.85 ± 1.08 | 95.14 ± 1.05 |

---

## 2. Accuracy with Varying Sequence Length

A **5-class classification task** was used to analyze performance under increasing sequence length.  
The model was trained for **20 epochs**.

### Table 2: Classification Accuracy vs. Sequence Length

| Sequence Length | 500 | 1000 | 1500 | 2000 | 4000 | 6000 |
|----------------:|----:|-----:|-----:|-----:|-----:|-----:|
| Accuracy (%) | 95.58 ± 1.05 | 95.88 ± 0.76 | 95.15 ± 0.89 | 95.16 ± 0.39 | 93.30 ± 1.27 | 92.99 ± 1.09 |

---

## 3. Training and Testing Time vs. Sequence Length

All times are reported in **seconds**.

### Table 3: Runtime vs. Sequence Length

| Sequence Length | 500 | 1000 | 1500 | 2000 | 4000 | 6000 |
|----------------:|----:|-----:|-----:|-----:|-----:|-----:|
| Train Time (s) | 162.61 ± 2.11 | 272.05 ± 0.75 | 343.86 ± 1.77 | 414.64 ± 2.33 | 645.93 ± 3.02 | 855.59 ± 2.93 |
| Test Time (s) | 31.28 ± 0.40 | 58.34 ± 0.15 | 75.92 ± 0.16 | 93.43 ± 0.91 | 151.17 ± 0.58 | 204.08 ± 0.28 |

---

## 4. Accuracy with Increasing Sample Size

Five-class classification using an **80/20 train/test split**.  
Each class contained **1799 samples**, with training data randomly sampled.  
The model was trained for **10 epochs**.

### Table 4: Accuracy vs. Sample Size

| Samples | 10000 | 15000 | 20000 | 25000 |
|--------:|------:|------:|------:|------:|
| Accuracy (%) | 94.55 ± 0.58 | 94.60 ± 0.65 | 96.23 ± 0.48 | 96.99 ± 0.56 |

---

## 5. Training and Testing Time vs. Sample Size

All times are reported in **seconds**.

### Table 5: Runtime vs. Sample Size

| Samples | 10000 | 15000 | 20000 | 25000 |
|--------:|------:|------:|------:|------:|
| Train Time (s) | 83.86 ± 0.53 | 167.85 ± 0.28 | 224.88 ± 1.09 | 276.61 ± 1.12 |
| Test Time (s) | 15.67 ± 0.16 | 15.81 ± 0.09 | 16.18 ± 0.11 | 16.25 ± 0.11 |

---

## 6. Comparison of GraphTM with Baseline Models

All models were trained for **10 epochs**.

### Table 6: Performance Comparison

| Method | Train Accuracy (%) | Test Accuracy (%) | Training Time (s) |
|------|-------------------:|------------------:|------------------:|
| GraphTM (Depth = 1) | 60.74 ± 0.14 | 59.81 ± 0.21 | 62.47 ± 0.08 |
| GraphTM (Depth = 2) | 95.17 ± 0.47 | 95.14 ± 0.81 | 84.37 ± 0.42 |
| BiLSTM | 94.43 ± 1.19 | 92.69 ± 0.23 | 50.39 ± 2.28 |
| LSTM | 88.70 ± 1.65 | 87.29 ± 0.91 | 26.02 ± 0.49 |
| GRU | 94.68 ± 0.26 | 94.05 ± 1.31 | 25.47 ± 0.20 |
| BiLSTM-CNN | 96.77 ± 0.42 | 95.44 ± 0.52 | 32.65 ± 0.68 |
| GraphCNN | 96.64 ± 0.19 | 96.35 ± 0.13 | 226.36 ± 1.59 |

