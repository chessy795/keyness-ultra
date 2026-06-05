# Keyness Analysis ULTRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)

Standalone keyness analysis: what words distinguish one corpus from another? Log-likelihood, Monroe log-odds, collocations, bootstrap confidence intervals, robustness resampling, dispersion, temporal trends, word embeddings, and co-occurrence networks.

## Features

| Module | Detail |
|--------|--------|
| **Unigram keyness** | Log-likelihood with Benjamini-Hochberg correction, log-ratio effect size |
| **Bigram/trigram keyness** | Same pipeline for n-grams |
| **Collocations** | PMI, t-score via Frequency Count Method |
| **Monroe log-odds** | Smoothed effect size with prior (Monroe et al. 2008) |
| **Bootstrap CI** | Percentile confidence intervals for LL/G² values |
| **Robustness** | 5-fold resampling of target corpus |
| **Dispersion** | Document presence (% of docs containing term) |
| **Temporal trends** | Monthly normalized frequency |
| **Word2Vec + PCA** | Semantic neighbourhood of top key terms |
| **Co-occurrence network** | Community detection via greedy modularity |
| **Volcano plot** | Log-ratio vs significance (HTML interactive) |

## Quick Start

```bash
pip install numpy pandas scikit-learn plotly networkx gensim

# Basic keyness analysis
python keyness_ultra.py target.csv text_col reference.csv

# Full pipeline with everything
python keyness_ultra.py target.csv text_col reference.csv --all

# With embeddings and co-occurrence network
python keyness_ultra.py target.csv text_col reference.csv --embeddings --network
```

## Usage

```
python keyness_ultra.py <target.csv> <text_col> <ref.csv> [options]
```

| Argument | Description |
|----------|-------------|
| `target.csv` | Corpus to analyse |
| `text_col` | Column name containing text |
| `ref.csv` | Reference corpus for comparison |
| `-o, --output` | Output directory (default: output/) |
| `--all` | Run all modules |
| `--embeddings` | Word2Vec + PCA semantic neighbourhood |
| `--network` | Co-occurrence network with communities |
| `--temporal` | Temporal trend analysis (requires date column) |
| `--top-n` | Top N key terms (default: 25) |

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Target CSV  │───→│  Tokenizer      │───→│  Keyness Pipeline  │
│  Reference   │    │  (unigram,      │    │                    │
│  CSV         │    │   bigram,       │    │  ┌──────────────┐  │
│              │    │   trigram)      │    │  │ Log-likelihood│  │
└─────────────┘    └──────────────────┘    │  │ Log-odds     │  │
                                            │  │ Monroe       │  │
                                            │  │ Bootstrap CI │  │
                                            │  │ Robustness   │  │
                                            │  │ Dispersion   │  │
                                            │  └──────┬───────┘  │
                                            └─────────┼──────────┘
                                                       │
                              ┌────────────────────────┼────────────┐
                              │                        │            │
                        ┌─────▼─────┐          ┌──────▼──────┐     │
                        │ Embeddings │          │ Co-occurrence│     │
                        │ Word2Vec   │          │ NetworkX     │     │
                        │ PCA        │          │ Communities  │     │
                        │ Scatter    │          │ Degree       │     │
                        └─────┬─────┘          └──────┬──────┘     │
                              │                        │            │
                        ┌─────▼────────────────────────▼──────┐     │
                        │         Output Artifacts             │     │
                        │  keyness_results.json               │     │
                        │  volcano_plot.html                  │     │
                        │  embedding_scatter.html            │     │
                        │  cooccurrence_network.html          │     │
                        │  temporal_trends.html               │     │
                        │  manifest.json                      │     │
                        └────────────────────────────────────┘     │
                              │                                     │
                        ┌─────▼─────────────────────────────────────▼─┐
                        │  Volcano Plot: log2(fold) vs significance  │
                        └───────────────────────────────────────────┘
```

## Benchmark Results

Measured June 2026. All datasets processed in under 2 seconds.

| Dataset | Processing Time | Output |
|---------|----------------|--------|
| 20 Newsgroups | <2s | Unigram/bigram/trigram keyness, collocations, visualizations |
| IMDb Sentiment | <2s | Unigram/bigram/trigram keyness, collocations, visualizations |
| BBC News | <2s | Unigram/bigram/trigram keyness, collocations, visualizations |
| TripAdvisor HK | <2s | Unigram/bigram/trigram keyness, collocations, visualizations |

**Note:** Significance filtering requires a larger corpus (>500 docs of the same topic) to produce statistically significant keyness results. Smaller corpora still produce valid keyness rankings, but many terms will not pass the significance threshold.

## Output

| File | Description |
|------|-------------|
| `keyness_results.json` | Per-term: LL, log-ratio, p-adjusted, dispersion, bootstrap CI |
| `volcano_plot.html` | Interactive volcano plot of effect size vs significance |
| `embedding_scatter.html` | Word2Vec PCA scatter for top key terms |
| `cooccurrence_network.html` | Network graph with community coloring |
| `temporal_trends.html` | Monthly normalized frequency trends |
| `manifest.json` | Runtime metadata |

## Keyness Measures

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Log-likelihood (G²)** | 2 × ∑(O × ln(O/E)) | General-purpose, follows χ² distribution |
| **Log-ratio** | log₂( freq_target / freq_ref ) | Effect size, interpretable as fold change |
| **Monroe log-odds** | log( (f_target + prior) / (f_ref + prior) ) | Smoothed, handles sparse data |
| **Bootstrap CI** | Percentile 2.5%-97.5% | Confidence interval for LL |
| **Robustness** | 5-fold CV of target corpus | Stability of keyness rankings |

## Evidence Base

| Method | Source |
|--------|--------|
| Log-likelihood keyness | Dunning 1993, Rayson & Garside 2000 |
| Log-ratio | Hardie 2014 |
| Monroe log-odds | Monroe, Colaresi & Quinn 2008 |
| Benjamini-Hochberg | Benjamini & Hochberg 1995 |
| Bootstrap CI | Efron & Tibshirani 1993 |
| Co-occurrence networks | Newman 2010 |

## Citation

```bibtex
@software{pang2026keynessultra,
  author = {Peter Pang},
  title = {Keyness Analysis ULTRA: Standalone Corpus Comparison Toolkit},
  year = {2026},
  url = {https://github.com/chessy795/keyness-ultra}
}
```

## License

MIT
