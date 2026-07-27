<div align="center">

# 🧬 eDNA Classification System

### AI-Powered Environmental DNA Analysis & Classification Platform

Developed by **Team AGNI**

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
<img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-success">
<img src="https://img.shields.io/badge/PyQt6-GUI-green">
<img src="https://img.shields.io/badge/SQLite-Database-blue">
<img src="https://img.shields.io/badge/Ethereum-Blockchain-purple">
<img src="https://img.shields.io/badge/License-MIT-orange">
</p>

### *Classifying Environmental DNA using Artificial Intelligence, Bioinformatics & Blockchain.*

</div>

---

# Overview

The **eDNA Classification System** is an AI-powered bioinformatics platform that identifies organisms from environmental DNA (eDNA) sequences using Machine Learning.

The platform combines DNA sequence preprocessing, k-mer feature extraction, and an XGBoost classification model to provide accurate biological classification. Along with intelligent prediction, the system offers a modern desktop interface, role-based authentication, audit logging, PDF report generation, and optional blockchain verification for secure scientific records.

---

# Features

| Feature | Description |
|----------|-------------|
| 🧬 DNA Classification | Predict biological class from DNA sequences |
| 🤖 Machine Learning | XGBoost-powered prediction engine |
| 🧩 Feature Extraction | K-mer based sequence encoding |
| 📈 Confidence Analysis | Prediction confidence scoring |
| ❓ Unknown Species Detection | Low-confidence organism identification |
| 🖥 Desktop Application | Interactive PyQt6 GUI |
| 🔐 Role-Based Login | Admin • Scientist • Viewer |
| 📄 PDF Reports | Automatic report generation |
| 🗄 SQLite Database | Prediction history & audit logs |
| ⛓ Blockchain | Ethereum-based prediction verification |

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Machine Learning | XGBoost • Scikit-Learn • NumPy • Pandas |
| GUI | PyQt6 |
| Database | SQLite |
| Blockchain | Ethereum • Web3.py |
| Reports | ReportLab |

---

# System Workflow

```text
Environmental Sample
        │
        ▼
DNA Sequencing
        │
        ▼
Sequence Cleaning
        │
        ▼
K-mer Feature Extraction
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Classification
        │
        ▼
Confidence Evaluation
        │
        ▼
Known / Unknown Detection
        │
        ▼
PDF Report
        │
        ▼
Blockchain Verification
```

---

# Machine Learning Pipeline

### DNA Cleaning

```
Input

ATCGXX12ATCG

↓

Output

ATCGATCG
```

---

### K-mer Generation

```
DNA

ATCGATCG

↓

4-mers

ATCG
TCGA
CGAT
GATC
ATCG
```

---

### Classification

```
Prediction

Actinopterygii

Confidence

92%
```

If confidence falls below the threshold:

```
UNKNOWN ORGANISM
```

---

# Blockchain Verification

Each prediction can optionally be stored on the Ethereum blockchain.

### Stored Information

- DNA Sequence Hash
- Predicted Biological Class
- Confidence Score
- Timestamp

### Technologies

- Ethereum Sepolia Testnet
- Smart Contracts
- SHA-256 Hashing
- Web3.py

---

# Project Structure

```text
eDNA_Classification_System/

├── Dataset/
│
├── GUI/
│
├── Machine_Learning/
│
├── Reports/
│
├── Blockchain/
│
├── Database/
│
├── README.md
│
└── requirements.txt
```

---

# Applications

- Biodiversity Monitoring
- Wildlife Conservation
- Environmental Assessment
- Marine Ecosystem Analysis
- Freshwater Monitoring
- Agricultural Bio-Surveillance
- Ecological Research
- Scientific Laboratories

---

# Future Roadmap

- Deep Learning Models
- REST API
- Cloud Deployment
- Real-time DNA Processing
- Interactive Analytics Dashboard
- Multi-Chain Blockchain Support
- Species Recommendation Engine

---

# Team AGNI

Building intelligent solutions at the intersection of

**Artificial Intelligence**
•
**Bioinformatics**
•
**Machine Learning**
•
**Blockchain Technology**

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository.

</div>
