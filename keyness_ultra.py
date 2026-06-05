#!/usr/bin/env python3
"""
KEYNESS ANALYSIS ULTRA — What Words Distinguish This Corpus?
=============================================================
Standalone, no API keys, no internet after model download.

Modules:
  1. Unigram/bigram/trigram keyness (log-likelihood)
  2. Collocations (PMI, t-score via FCM)
  3. Monroe log-odds (smoothed effect size)
  4. Bootstrap confidence intervals
  5. Robustness (5-fold resampling)
  6. Dispersion (document presence)
  7. Temporal trends (monthly normalized)
  8. Word embeddings (Word2Vec + PCA)
  9. Co-occurrence network (community detection)
  10. Volcano plots

Usage:
  python keyness_ultra.py target.csv text_col ref.csv       # basic
  python keyness_ultra.py target.csv text_col ref.csv --all # everything
  python keyness_ultra.py target.csv text_col ref.csv --embeddings --network

Author: Peter Pang (2026)
"""
import argparse, os, re, sys, time, warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from gensim.models import Word2Vec as GensimWord2Vec
    HAS_WORD2VEC = True
except ImportError:
    HAS_WORD2VEC = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "min_unigram_freq": 10,
    "min_ngram_freq": 5,
    "min_doc_freq": 5,
    "prior": 0.01,              # Monroe log-odds smoothing
    "keyness_ll_min": 25,       # filter for highly distinctive
    "keyness_p_adj_cutoff": 0.001,
    "log2fold_cutoff": 1.5,
    "n_bootstrap": 100,
    "robustness_folds": 5,
    "colloc_window": 5,
    "top_n_output": 25,         # limit output to top N
    "embedding_dim": 100,
    "embedding_window": 5,
    "embedding_iter": 20,
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_text(text: str, stops: set) -> List[str]:
    """Tokenize, lowercase, remove stopwords."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.lower().split()
    return [t for t in tokens if len(t) > 2 and t not in stops]


def monroe_logodds(n_t, N_t, n_r, N_r, prior=0.01):
    """Monroe-style smoothed log-odds ratio (base 2)."""
    a1 = n_t + prior
    b1 = (N_t - n_t) + prior
    a2 = n_r + prior
    b2 = (N_r - n_r) + prior
    return np.log2((a1 / b1) / (a2 / b2))


def log_likelihood(a, b, c, d):
    """Log-likelihood ratio for 2x2 table."""
    N = a + b + c + d
    E = [(a+b)*(a+c)/N, (a+b)*(b+d)/N, (c+d)*(a+c)/N, (c+d)*(b+d)/N]
    obs = [a, b, c, d]
    ll = 0
    for o, e in zip(obs, E):
        if o > 0 and e > 0:
            ll += 2 * o * np.log(o / e)
    return ll


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: KEYNESS (unigram, bigram, trigram)
# ═══════════════════════════════════════════════════════════════════════════════

def run_keyness(target_texts, ref_texts, config=CONFIG, output_dir="output"):
    """Compute keyness: what words distinguish target from reference?"""
    print(f"\n{'='*60}")
    print("KEYNESS ANALYSIS")
    print(f"{'='*60}")

    # Tokenize
    stops = set(stopwords if 'stopwords' in dir() else []) | {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "just", "like",
    }
    target_tokens = [tokenize_text(t, stops) for t in target_texts]
    ref_tokens = [tokenize_text(t, stops) for t in ref_texts]

    # Build unigram frequencies
    target_freq = Counter()
    ref_freq = Counter()
    for tokens in target_tokens:
        target_freq.update(tokens)
    for tokens in ref_tokens:
        ref_freq.update(tokens)

    total_t = sum(target_freq.values())
    total_r = sum(ref_freq.values())

    # Compute keyness for each unigram
    all_words = set(target_freq.keys()) | set(ref_freq.keys())
    results = []
    for word in all_words:
        a = target_freq.get(word, 0)
        b = ref_freq.get(word, 0)
        c = total_t - a
        d = total_r - b

        if a + b < CONFIG["min_unigram_freq"]:
            continue

        ll = log_likelihood(a, b, c, d)
        freq_t_pm = a / total_t * 1e6
        freq_r_pm = b / total_r * 1e6
        log2fold = np.log2((freq_t_pm + 1e-6) / (freq_r_pm + 1e-6))
        monroe = monroe_logodds(a, total_t, b, total_r, CONFIG["prior"])

        # P-value from chi-square
        from scipy.stats import chi2
        p_value = chi2.sf(ll, 1)

        results.append({
            "word": word,
            "f_target": a,
            "f_ref": b,
            "freq_target_pm": round(freq_t_pm, 2),
            "freq_ref_pm": round(freq_r_pm, 2),
            "log_likelihood": round(ll, 2),
            "log2fold": round(log2fold, 3),
            "monroe_logodds": round(monroe, 3),
            "p_value": p_value,
        })

    if not results:
        print("  No terms passed frequency threshold — returning empty results")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("log_likelihood", ascending=False)

    # BH correction
    from statsmodels.stats.multitest import multipletests
    _, results_df["p_adj"], _, _ = multipletests(results_df["p_value"], method="fdr_bh")

    # Filter
    sig_df = results_df[
        (results_df["log_likelihood"] > CONFIG["keyness_ll_min"]) &
        (results_df["p_adj"] < CONFIG["keyness_p_adj_cutoff"])
    ].head(CONFIG["top_n_output"])

    # Bigram keyness
    bigram_results = _compute_ngram_keyness(target_tokens, ref_tokens, total_t, total_r, n=2)
    trigram_results = _compute_ngram_keyness(target_tokens, ref_tokens, total_t, total_r, n=3)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "unigram_keyness.csv"), index=False)
    sig_df.to_csv(os.path.join(output_dir, "unigram_keyness_significant.csv"), index=False)
    if isinstance(bigram_results, pd.DataFrame):
        bigram_results.to_csv(os.path.join(output_dir, "bigram_keyness.csv"), index=False)
    if isinstance(trigram_results, pd.DataFrame):
        trigram_results.to_csv(os.path.join(output_dir, "trigram_keyness.csv"), index=False)

    print(f"  {len(results_df)} unigrams tested, {len(sig_df)} significant")
    print(f"  Top 5 target-rich:")
    for _, r in sig_df.head(5).iterrows():
        print(f"    {r['word']:>15}  LL={r['log_likelihood']:.1f}  log2={r['log2fold']:.2f}")
    print(f"  Top 5 reference-rich:")
    for _, r in sig_df.tail(5).iterrows():
        print(f"    {r['word']:>15}  LL={r['log_likelihood']:.1f}  log2={r['log2fold']:.2f}")

    return results_df, sig_df, bigram_results, trigram_results


def _compute_ngram_keyness(target_tokens, ref_tokens, total_t, total_r, n=2):
    """Compute keyness for n-grams."""
    target_ngrams = Counter()
    ref_ngrams = Counter()
    for tokens in target_tokens:
        for i in range(len(tokens) - n + 1):
            ngram = "_".join(tokens[i:i+n])
            target_ngrams[ngram] += 1
    for tokens in ref_tokens:
        for i in range(len(tokens) - n + 1):
            ngram = "_".join(tokens[i:i+n])
            ref_ngrams[ngram] += 1

    results = []
    all_ngrams = set(target_ngrams.keys()) | set(ref_ngrams.keys())
    for ngram in all_ngrams:
        a = target_ngrams.get(ngram, 0)
        b = ref_ngrams.get(ngram, 0)
        if a + b < CONFIG["min_ngram_freq"]:
            continue
        ll = log_likelihood(a, b, total_t - a, total_r - b)
        from scipy.stats import chi2
        p_value = chi2.sf(ll, 1)
        results.append({
            "ngram": ngram,
            "f_target": a,
            "f_ref": b,
            "log_likelihood": round(ll, 2),
            "p_value": p_value,
        })

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("log_likelihood", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: COLLOCATIONS (PMI, t-score)
# ═══════════════════════════════════════════════════════════════════════════════

def run_collocations(target_texts, config=CONFIG, output_dir="output"):
    """Extract collocations using PMI and t-score."""
    print(f"\n{'='*60}")
    print("COLLOCATIONS (PMI, t-score)")
    print(f"{'='*60}")
    if len(target_texts) < 5:
        print("  Too few documents for collocation analysis")
        return

    stops = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
             "his", "her", "was", "one", "our", "out", "has", "how", "just",
             "like", "get", "got", "would", "really", "im", "ive", "dont"}
    all_tokens = []
    for t in target_texts:
        all_tokens.extend(tokenize_text(t, stops))

    # Bigram counts
    bigram_counts = Counter()
    unigram_counts = Counter(all_tokens)
    N = len(all_tokens) - 1

    for i in range(len(all_tokens) - 1):
        bigram_counts[(all_tokens[i], all_tokens[i+1])] += 1

    results = []
    for (w1, w2), cooc in bigram_counts.items():
        if cooc < 3:
            continue
        freq_w1 = unigram_counts[w1]
        freq_w2 = unigram_counts[w2]
        p_xy = cooc / N
        p_x = freq_w1 / N
        p_y = freq_w2 / N
        pmi = np.log2((p_xy + 1e-12) / (p_x * p_y + 1e-12))
        expected = p_x * freq_w2
        tscore = (cooc - expected) / np.sqrt(cooc + 1e-8)

        results.append({
            "w1": w1, "w2": w2,
            "cooc": cooc,
            "pmi": round(pmi, 3),
            "tscore": round(tscore, 3),
        })

    if not results:
        print("  No collocations found")
        return
    df = pd.DataFrame(results).sort_values("pmi", ascending=False)
    df = df.head(CONFIG["top_n_output"])
    df.to_csv(os.path.join(output_dir, "collocations.csv"), index=False)

    print(f"  {len(df)} collocation pairs")
    for _, r in df.head(5).iterrows():
        print(f"    {r['w1']:>12} + {r['w2']:<12}  PMI={r['pmi']:.2f}  t={r['tscore']:.2f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════════

def run_bootstrap(target_texts, ref_texts, sig_df, config=CONFIG, output_dir="output"):
    """Bootstrap confidence intervals for keyness."""
    if sig_df.empty:
        print("  No significant terms — skipping bootstrap")
        return
    print(f"\n{'='*60}")
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'='*60}")

    stops = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
             "his", "her", "was", "one", "our", "out", "has", "how", "just",
             "like", "get", "got", "would", "really", "im", "ive", "dont"}
    target_token_lists = [tokenize_text(t, stops) for t in target_texts]
    ref_token_lists = [tokenize_text(t, stops) for t in ref_texts]

    target_freqs = [Counter(t) for t in target_token_lists]
    ref_freqs = [Counter(t) for t in ref_token_lists]

    n_bootstrap = CONFIG["n_bootstrap"]
    bootstrap_g2 = {word: [] for word in sig_df["word"]}

    np.random.seed(42)
    for i in range(n_bootstrap):
        # Resample with replacement
        boot_t_indices = np.random.choice(len(target_freqs), len(target_freqs), replace=True)
        boot_r_indices = np.random.choice(len(ref_freqs), len(ref_freqs), replace=True)

        boot_t_freq = Counter()
        boot_r_freq = Counter()
        for idx in boot_t_indices:
            boot_t_freq.update(target_freqs[idx])
        for idx in boot_r_indices:
            boot_r_freq.update(ref_freqs[idx])

        boot_total_t = sum(boot_t_freq.values())
        boot_total_r = sum(boot_r_freq.values())

        for word in bootstrap_g2:
            a = boot_t_freq.get(word, 0)
            b = boot_r_freq.get(word, 0)
            ll = log_likelihood(a, b, boot_total_t - a, boot_total_r - b)
            bootstrap_g2[word].append(ll)

    # Compute CIs
    ci_results = []
    for word in sig_df["word"]:
        scores = bootstrap_g2[word]
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)
        ci_results.append({
            "word": word,
            "g2_original": sig_df[sig_df["word"] == word]["log_likelihood"].values[0],
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "significant": ci_lower > 0,
        })

    ci_df = pd.DataFrame(ci_results).sort_values("g2_original", ascending=False)
    ci_df.to_csv(os.path.join(output_dir, "bootstrap_ci.csv"), index=False)

    n_sig = sum(ci_df["significant"])
    print(f"  {n_bootstrap} bootstrap iterations")
    print(f"  {n_sig}/{len(ci_df)} terms have CIs above zero (reliable)")
    for _, r in ci_df.head(5).iterrows():
        sig = "✓" if r["significant"] else "✗"
        print(f"    {sig} {r['word']:>15}  G²={r['g2_original']:.1f}  CI=[{r['ci_lower']:.1f}, {r['ci_upper']:.1f}]")

    return ci_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: ROBUSTNESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def run_robustness(target_texts, ref_texts, config=CONFIG, output_dir="output"):
    """5-fold resampling to check if key terms are stable."""
    print(f"\n{'='*60}")
    print("ROBUSTNESS CHECK (5-fold resampling)")
    print(f"{'='*60}")

    stops = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
             "his", "her", "was", "one", "our", "out", "has", "how", "just",
             "like", "get", "got", "would", "really", "im", "ive", "dont"}

    target_token_lists = [tokenize_text(t, stops) for t in target_texts]
    ref_token_lists = [tokenize_text(t, stops) for t in ref_texts]

    all_top_words = []
    np.random.seed(42)
    for fold in range(CONFIG["robustness_folds"]):
        boot_t = np.random.choice(len(target_token_lists), len(target_token_lists), replace=True)
        boot_r = np.random.choice(len(ref_token_lists), len(ref_token_lists), replace=True)

        t_freq = Counter()
        r_freq = Counter()
        for idx in boot_t:
            t_freq.update(target_token_lists[idx])
        for idx in boot_r:
            r_freq.update(ref_token_lists[idx])

        total_t = sum(t_freq.values())
        total_r = sum(r_freq.values())

        fold_results = []
        for word in set(t_freq.keys()) | set(r_freq.keys()):
            a = t_freq.get(word, 0)
            b = r_freq.get(word, 0)
            if a + b < CONFIG["min_unigram_freq"]:
                continue
            ll = log_likelihood(a, b, total_t - a, total_r - b)
            fold_results.append((word, ll))

        fold_results.sort(key=lambda x: -x[1])
        top_200 = [w for w, _ in fold_results[:200]]
        all_top_words.append(set(top_200))
        print(f"  Fold {fold+1}: {len(top_200)} terms")

    # Overlap analysis
    intersection = all_top_words[0]
    for s in all_top_words[1:]:
        intersection = intersection & s

    overlap_pct = len(intersection) / 200 * 100
    print(f"\n  Overlap across all folds: {len(intersection)}/200 ({overlap_pct:.1f}%)")
    print(f"  Top stable terms: {', '.join(sorted(intersection)[:15])}")

    # Save
    pd.DataFrame({"stable_terms": sorted(intersection)}).to_csv(
        os.path.join(output_dir, "robustness_stable_terms.csv"), index=False)

    return intersection


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: WORD EMBEDDINGS (Word2Vec)
# ═══════════════════════════════════════════════════════════════════════════════

def run_embeddings(target_texts, sig_df, config=CONFIG, output_dir="output"):
    """Word2Vec embeddings — which key words are semantically similar?"""
    if not HAS_WORD2VEC:
        print("  Skipping: gensim not available (pip install gensim)")
        return pd.DataFrame()
    if sig_df.empty:
        print("  No significant terms — skipping embeddings")
        return

    print(f"\n{'='*60}")
    print("WORD EMBEDDINGS (Word2Vec)")
    print(f"{'='*60}")

    # Train Word2Vec
    texts_for_w2v = [tokenize_text(t, {"the", "and", "for", "are", "but", "not", "you"})
                     for t in target_texts]
    model = GensimWord2Vec(texts_for_w2v, vector_size=config["embedding_dim"],
                           window=config["embedding_window"], epochs=config["embedding_iter"],
                           min_count=5, workers=4)

    # Find similar terms for key words
    key_terms = sig_df["word"].head(10).tolist()
    similarity_results = []

    for term in key_terms:
        if term in model.wv:
            try:
                similar = model.wv.most_similar(term, topn=8)
                for sim_word, sim_score in similar:
                    similarity_results.append({
                        "target": term,
                        "similar": sim_word,
                        "similarity": round(sim_score, 3),
                    })
            except:
                pass

    if similarity_results:
        sim_df = pd.DataFrame(similarity_results)
        sim_df.to_csv(os.path.join(output_dir, "word_similarity.csv"), index=False)

        print(f"  {len(similarity_results)} similarity pairs")
        for target in key_terms[:5]:
            sims = sim_df[sim_df["target"] == target]
            if not sims.empty:
                top_sims = ", ".join(sims["similar"].head(5).tolist())
                print(f"    {target:>15} -> {top_sims}")

    # PCA visualization
    if HAS_PLOTLY:
        terms_for_plot = []
        for target in key_terms[:5]:
            if target in model.wv:
                terms_for_plot.append(target)
                try:
                    similar = model.wv.most_similar(target, topn=3)
                    for sim_word, _ in similar:
                        terms_for_plot.append(sim_word)
                except:
                    pass

        terms_for_plot = list(set(terms_for_plot))
        if len(terms_for_plot) >= 5:
            term_vectors = np.array([model.wv[t] for t in terms_for_plot])
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(term_vectors)

            fig = go.Figure()
            is_key = [t in key_terms for t in terms_for_plot]
            fig.add_trace(go.Scatter(
                x=pca_result[np.array(is_key), 0], y=pca_result[np.array(is_key), 1],
                mode="markers+text", text=[t for t, k in zip(terms_for_plot, is_key) if k],
                marker=dict(size=12, color="#e74c3c"),
                name="Key terms",
            ))
            fig.add_trace(go.Scatter(
                x=pca_result[~np.array(is_key), 0], y=pca_result[~np.array(is_key), 1],
                mode="markers+text", text=[t for t, k in zip(terms_for_plot, is_key) if not k],
                marker=dict(size=8, color="#3498db"),
                name="Similar terms",
            ))
            fig.update_layout(
                title="Semantic Space of Key Terms (Word2Vec PCA)",
                xaxis_title="PC1", yaxis_title="PC2",
                width=800, height=600,
            )
            fig.write_html(os.path.join(output_dir, "embeddings_pca.html"))
            print(f"  Saved: embeddings_pca.html")

    return sim_df if similarity_results else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: CO-OCCURRENCE NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

def run_network(target_texts, sig_df, config=CONFIG, output_dir="output"):
    """Co-occurrence network with community detection."""
    if len(target_texts) < 5:
        print("  Too few documents for network analysis")
        return
    if sig_df.empty:
        print("  No significant terms — skipping network analysis")
        return
    print(f"\n{'='*60}")
    print("CO-OCCURRENCE NETWORK")
    print(f"{'='*60}")

    stops = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
             "his", "her", "was", "one", "our", "out", "has", "how", "just",
             "like", "get", "got", "would", "really", "im", "ive", "dont"}
    all_tokens = []
    for t in target_texts:
        all_tokens.extend(tokenize_text(t, stops))

    # Bigram counts
    bigram_counts = Counter()
    for i in range(len(all_tokens) - 1):
        bigram_counts[(all_tokens[i], all_tokens[i+1])] += 1

    # Build network from significant bigrams
    G = nx.Graph()
    for (w1, w2), count in bigram_counts.items():
        if count >= 5 and len(w1) > 2 and len(w2) > 2:
            G.add_edge(w1, w2, weight=count)

    if len(G) < 5:
        print("  Too few nodes for network")
        return

    # Metrics
    pagerank = nx.pagerank(G, weight="weight")
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")

    # Community detection
    try:
        communities = list(greedy_modularity_communities(G))
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
    except:
        community_map = {}

    # Save metrics
    metrics = []
    for node in G.nodes():
        metrics.append({
            "word": node,
            "pagerank": round(pagerank.get(node, 0), 4),
            "degree": degree.get(node, 0),
            "betweenness": round(betweenness.get(node, 0), 4),
            "community": community_map.get(node, -1),
        })
    metrics_df = pd.DataFrame(metrics).sort_values("pagerank", ascending=False)
    metrics_df.to_csv(os.path.join(output_dir, "network_metrics.csv"), index=False)

    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(set(community_map.values()))} communities")
    print(f"  Top 5 by PageRank:")
    for _, r in metrics_df.head(5).iterrows():
        print(f"    {r['word']:>15}  PR={r['pagerank']:.4f}  comm={r['community']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7: VOLCANO PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_volcano(results_df, output_dir="output"):
    """Generate volcano plot."""
    if results_df.empty or len(results_df) < 3:
        print("  Too few terms for volcano plot")
        return
    print(f"\n{'='*60}")
    print("VOLCANO PLOT")
    print(f"{'='*60}")

    df = results_df.copy()
    df["significant"] = (df["p_adj"] < 0.001) & (df["log2fold"].abs() > 1.0)

    fig = go.Figure()

    # Non-significant
    fig.add_trace(go.Scatter(
        x=df[~df["significant"]]["log2fold"],
        y=df[~df["significant"]]["log_likelihood"],
        mode="markers", marker=dict(size=5, color="#b3b3b3"),
        name="Non-significant",
    ))

    # Significant (target-rich)
    sig_pos = df[df["significant"] & (df["log2fold"] > 0)]
    fig.add_trace(go.Scatter(
        x=sig_pos["log2fold"], y=sig_pos["log_likelihood"],
        mode="markers+text", text=sig_pos["word"],
        marker=dict(size=8, color="firebrick"),
        name="Target-rich",
    ))

    # Significant (reference-rich)
    sig_neg = df[df["significant"] & (df["log2fold"] < 0)]
    fig.add_trace(go.Scatter(
        x=sig_neg["log2fold"], y=sig_neg["log_likelihood"],
        mode="markers+text", text=sig_neg["word"],
        marker=dict(size=8, color="steelblue"),
        name="Reference-rich",
    ))

    fig.update_layout(
        title="Keyness Volcano Plot",
        xaxis_title="log2 fold (target / reference)",
        yaxis_title="Log-likelihood",
        width=900, height=600,
    )
    fig.write_html(os.path.join(output_dir, "volcano_plot.html"))
    print(f"  Saved: volcano_plot.html")
    print(f"  {len(sig_pos)} target-rich, {len(sig_neg)} reference-rich")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Keyness Analysis Ultra — What words distinguish this corpus?"
    )
    parser.add_argument("target", help="Path to target corpus CSV")
    parser.add_argument("text_col", help="Text column name")
    parser.add_argument("reference", help="Path to reference corpus CSV")
    parser.add_argument("--ref-text-col", default=None, help="Reference text column (default: same as target)")
    parser.add_argument("--output", "-o", default="keyness_output", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Sample N docs from each corpus")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--embeddings", action="store_true", help="Word embeddings (needs word2vec)")
    parser.add_argument("--network", action="store_true", help="Co-occurrence network")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap CI (slow)")
    parser.add_argument("--robustness", action="store_true", help="Robustness check (slow)")
    parser.add_argument("--volcano", action="store_true", help="Volcano plot")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t_start = time.time()

    # Load data
    print("Loading data...")
    target_df = pd.read_csv(args.target)
    ref_df = pd.read_csv(args.reference)

    text_col = args.text_col
    ref_col = args.ref_text_col or args.text_col

    target_texts = target_df[text_col].dropna().astype(str).tolist()
    ref_texts = ref_df[ref_col].dropna().astype(str).tolist()

    if args.sample:
        np.random.seed(42)
        target_texts = list(np.random.choice(target_texts, min(args.sample, len(target_texts)), replace=False))
        ref_texts = list(np.random.choice(ref_texts, min(args.sample, len(ref_texts)), replace=False))

    print(f"  Target: {len(target_texts)} docs")
    print(f"  Reference: {len(ref_texts)} docs")

    # Run keyness
    results_df, sig_df, bigram_df, trigram_df = run_keyness(target_texts, ref_texts, output_dir=args.output)

    # Run collocations
    run_collocations(target_texts, output_dir=args.output)

    # Volcano plot
    if args.all or args.volcano:
        run_volcano(results_df, output_dir=args.output)

    # Bootstrap CI
    if args.all or args.bootstrap:
        run_bootstrap(target_texts, ref_texts, sig_df, output_dir=args.output)

    # Robustness
    if args.all or args.robustness:
        run_robustness(target_texts, ref_texts, output_dir=args.output)

    # Word embeddings
    if args.all or args.embeddings:
        run_embeddings(target_texts, sig_df, output_dir=args.output)

    # Network
    if args.all or args.network:
        run_network(target_texts, sig_df, output_dir=args.output)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE ({elapsed:.1f}s)")
    print(f"Output: {args.output}/")
    for f in sorted(os.listdir(args.output)):
        print(f"  {f}")
    print("="*60)


if __name__ == "__main__":
    main()
