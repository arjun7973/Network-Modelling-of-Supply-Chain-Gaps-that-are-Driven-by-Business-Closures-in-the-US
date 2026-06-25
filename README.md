# Network Modelling of Supply Chain Gaps Driven by Business Closures in the United States

## Overview

Businesses operate within highly interconnected supply chain networks where disruptions in one sector can propagate across multiple industries. Business closures create gaps in these networks, affecting suppliers, customers, and dependent sectors throughout the economy.

This project investigates business closures as indicators of supply chain disruptions in the United States. Using historical business dynamics, demographic, and economic data, the study develops deep learning models to forecast business closures and identify patterns of co-occurrence across states and sectors. The objective is to help businesses and policymakers anticipate supply chain gaps and develop mitigation strategies.

---

## Research Objectives

- Analyze patterns and trends in business closures across U.S. states and sectors.
- Examine sectoral and geographic co-occurrence of business closures.
- Engineer temporal and economic indicators associated with business failures.
- Develop deep learning models for forecasting business closures.
- Compare LSTM, LSTM Entity Embedding, 1D CNN, and 2D CNN architectures.
- Evaluate the effectiveness of network-based representations in predicting supply chain disruptions.

---

## Data Sources

Three datasets were collected and integrated.

### Business Closure Data

**Source:** U.S. Census Bureau

- 534,888 observations
- 28 attributes
- Period: 1978–2023

### Median Household Income Data

**Source:** Federal Reserve Economic Data (FRED)

- 2,050 observations
- 3 attributes
- Period: 1984–2024

### Annual Estimated Population Data

**Source:** Federal Reserve Economic Data (FRED)

- 6,200 observations
- 3 attributes
- Period: 1900–2024

---

## Feature Engineering

To create a unified modeling dataset, state and year variables were standardized and merged across data sources.

Additional engineered features included:

- Closure Rate
- Closure Rate (Firms)
- Lag 1 Closure Rate
- Lag 2 Closure Rate
- Three-Year Moving Average Closure Rate
- Year-over-Year Income Growth
- Year-over-Year Population Growth
- Employment Density

The final modeling dataset contained:

- 245,779 observations
- 33 features
- Period: 2000–2023

**Target Variable:** Firm Deaths (`closure_target`)

---

## Exploratory Data Analysis

### Temporal Trends

Analysis of closure and employment dynamics revealed:

- Establishment closure rates peaked in:
  - 2009 (highest)
  - 2002
  - 2023

- Job destruction rates peaked in:
  - 2002 (highest)
  - 2009
  - 2021

- Net positive job creation rates were highest in:
  - 2022
  - 2000

A prolonged period of relatively low closure rates was observed between 2014 and 2019.

### Correlation Analysis

The correlation heatmap showed:

- Strong positive correlation between Closure Rate and Closure Rate Firms.
- Strong negative correlation between Job Destruction Rate and Net Job Creation Rate.
- No substantial multicollinearity among the remaining features.

### Sector-Level Analysis

| Metric | Sector |
|----------|----------|
| Highest Closure Intensity | Transport and Warehousing |
| Highest Total Business Closures | Construction |
| Highest Average Job Loss Intensity | Agriculture, Forestry, Fishing and Hunting |
| Highest Relative Closure Intensity | Real Estate and Rental and Leasing |

### Geographic Analysis

State-level analysis showed that:

- California
- Florida
- New York
- Texas

recorded the largest numbers of firm closures.

Business closure distributions across all states were positively skewed, indicating that closures remain exceptional events rather than normal business outcomes.

### Closure Co-Occurrence Network Analysis

To capture interconnected business disruptions, closure co-occurrence patterns were analyzed across states and sectors.

Key findings:

- Most sectors exhibited the strongest closure co-occurrence with California.
- Professional, Scientific and Technical Services in California showed the highest co-occurrence levels.
- The following sectors showed minimal closure co-occurrence with any state:
  - Agriculture, Forestry, Fishing and Hunting
  - Utilities
  - Mining, Quarrying, and Oil and Gas Extraction
  - Management of Companies and Enterprises

These patterns suggest varying levels of interdependence across sectors within the broader supply chain network.

---

## Deep Learning Models

### Long Short-Term Memory (LSTM)

Architecture:

- 2 LSTM Layers
- 2 Dropout Layers
- 2 Dense Layers

Performance:

| Metric | Value |
|----------|----------|
| Training R² | 0.9096 |
| Testing R² | 0.8081 |

The model explained approximately 81% of variation in business closures on unseen data.

### LSTM Entity Embedding Co-Occurrence Model

This model incorporated state-sector co-occurrence information through entity embeddings.

Architecture:

#### Embedding Layers

- State Embedding
- Sector Embedding

#### Sequential Layers

- 1 LSTM Layer
- 3 Linear Layers
- 2 ReLU Layers
- 2 Dropout Layers

Performance:

| Metric | Value |
|----------|----------|
| Training R² | 0.9835 |
| Testing R² | 0.9784 |

The inclusion of co-occurrence embeddings significantly improved predictive performance.

Higher prediction errors were observed for:

- California – Health Care and Social Assistance
- California – Construction
- New York – Professional, Scientific and Technical Services

### One-Dimensional Convolutional Neural Network (1D CNN)

Architecture:

- 2 Conv1D Layers
- 1 Batch Normalization Layer
- 1 MaxPooling1D Layer
- 2 LSTM Layers
- 3 Dropout Layers
- 2 Dense Layers

Performance:

| Metric | Value |
|----------|----------|
| Training R² | 0.8742 |
| Testing R² | 0.7891 |

### Two-Dimensional Convolutional Neural Network (2D CNN)

Architecture:

- 4 Conv2D Layers
- 2 Batch Normalization Layers
- 2 MaxPooling2D Layers
- 4 Dropout Layers
- 1 Flatten Layer
- 4 Dense Layers

Performance:

| Metric | Value |
|----------|----------|
| Training R² | 0.8180 |
| Testing R² | 0.6189 |

---

## Model Comparison

| Model | Training R² | Testing R² |
|---------|---------|---------|
| LSTM | 0.9096 | 0.8081 |
| LSTM Entity Embedding Co-Occurrence | 0.9835 | 0.9784 |
| 1D CNN | 0.8742 | 0.7891 |
| 2D CNN | 0.8180 | 0.6189 |

---

## Key Findings

- Transport and Warehousing exhibited the highest closure intensity.
- Real Estate and Rental and Leasing showed the highest relative closure intensity.
- Agriculture, Forestry, Fishing and Hunting experienced the highest average job loss intensity.
- California dominated closure co-occurrence patterns across most sectors.
- State-sector co-occurrence information substantially improved predictive performance.
- CNN-based architectures did not outperform LSTM-based approaches for business closure forecasting.
- The LSTM Entity Embedding Co-Occurrence model achieved the best predictive performance, explaining approximately 97.84% of variation in business closures within the testing dataset.

---

## Repository Structure

```text
├── Data/
│   ├── Raw Data
│   ├── Processed Data
│   └── Final Modeling Dataset
│
├── Week_2_Data_Practicum_Nagarjuna_Devarayi.ipynb
├── Week_3_Data_Practicum_Nagarjuna_Devarayi.ipynb
├── Week_4_Data_Practicum_Nagarjuna_Devarayi.ipynb
├── Week_5_Data_Practicum_Nagarjuna_Devarayi.ipynb
├── Week_6_Data_Practicum_Nagarjuna_Devarayi.ipynb
├── Week_7_Data_Practicum_Nagarjuna_Devarayi.ipynb
│
└── README.md
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Matplotlib
- Seaborn
- Jupyter Notebook
