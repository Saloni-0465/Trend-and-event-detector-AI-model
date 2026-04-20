# Literature Review — Dynamic Trend & Event Detection

## 1. Prior work (four core papers)

### 1.1 Latent Dirichlet Allocation [Blei et al., 2003]

LDA models each document as a mixture of topics and each topic as a distribution over words, with inference via variational Bayes or Gibbs sampling. It is still a standard probabilistic topic model for text streams.

### 1.2 Topics over Time [Wang & McCallum, 2006]

TOT extends LDA by tying each topic to a continuous-time distribution over dates, so popularity of a topic can rise and fall without fixing windows in advance. That directly targets *when* a narrative peaks.

### 1.3 Sentence-BERT [Reimers & Gurevych, 2019]

Sentence-BERT produces fixed-size sentence embeddings from pretrained Transformers, so semantically similar headlines map nearby even when wording differs. That addresses the vocabulary mismatch problem of pure bag-of-words models.

### 1.4 Topic coherence [Mimno et al., 2011]

Mimno et al. argue that perplexity alone can favor models whose topics humans find uninterpretable, and that coherence metrics (e.g. based on word co-occurrence) align better with human judgments of topic quality.

---

## 2. What was missing or limited in that line of work

**LDA (Blei et al.)** assumes exchangeable words inside a document (bag-of-words). It does not encode syntax or paraphrase: two articles on the same event with different vocabulary may get weak topic overlap. It also needs a fixed topic count *K* and does not by itself say how topics *move* over time—you only get a static mixture per document unless you add another mechanism.

**TOT (Wang & McCallum)** is closer to “real” temporal topics, but it is heavier to implement and tune than windowed LDA, and it still works from discrete word counts, not dense semantic vectors. Many applied projects skip it and use simpler time slicing instead.

**Sentence-BERT (Reimers & Gurevych)** gives strong semantics, but clustering those vectors (e.g. K-Means) does not yield human-readable “top words” per cluster the way LDA does, and choosing the number of clusters *k* is an extra design choice. It is also a black-box encoder unless you connect it to interpretable baselines.

**Evaluation (Mimno et al.)** established that a single number like perplexity is not enough to judge topics. Work that only reports perplexity risks picking a model that looks good on paper but fails for analysts who need interpretable themes.

Together, the gap for a *practical* trend detector is: (i) compare **lexical**, **probabilistic**, and **semantic** views of the same stream; (ii) measure **change over time** in a simple, reproducible way; (iii) report **both** predictive fit and topic interpretability; (iv) stay within methods you can explain in a course setting without building a full dynamic topic model on day one.

---

## 3. How our project addresses those limitations

| Limitation in prior work | What we do about it |
|--------------------------|---------------------|
| LDA ignores semantics and word order | We add a **sentence-embedding + K-Means** path so similar meaning clusters together even when keywords differ. |
| Plain LDA does not expose temporal drift clearly | We **bin documents by calendar time**, aggregate mean document–topic vectors per bin, and measure **Jensen–Shannon divergence** between adjacent bins so shifts in the topic mix are visible. |
| TOT is complex to ship early | We **do not** implement TOT in Phase 1; we use **fixed windows + post-hoc drift metrics** as a lighter-weight substitute that still answers “is the stream changing week to week?”. |
| Perplexity alone is misleading | We report **perplexity** **and** **u_mass coherence** (Mimno-style interpretability). Models are fit on **train time** only; perplexity and drift metrics include **held-out test** slices. |
| Embedding clusters are hard to read | We **keep frequency / TF-IDF baselines and LDA** in the same pipeline so outputs stay interpretable and we can compare approaches side by side. |
| Choosing *K* for clustering is arbitrary | Phase 1: pick *k* by **silhouette** over a small range on **training** embeddings; **K-Means** is fit on train and **predict** assigns test (`run_embeddings`). |

---

## 4. Approach proposed in this project

We propose a **three-tier pipeline** on a **timestamped news stream** (News Category Dataset: headline + short description, publication date, editor category):

1. **Tier 1 — Baselines:** Per time window, extract top terms by **raw frequency** and **TF-IDF**; quantify lexical stability across windows with **Jaccard** overlap on top terms. This is cheap, transparent, and catches obvious keyword shifts.

2. **Tier 2 — Probabilistic topics (LDA):** Train **LdaModel** on the **full** time-sorted corpus (Phase 1 keeps this simple), report **perplexity** and **coherence**, and summarize **how the mixture of topics changes over time** via **JSD** between adjacent bins—our lightweight answer to “dynamic narrative” without fitting TOT.

3. **Tier 3 — Deep representation (optional demo):** Encode each article with a **frozen** `sentence-transformers` model, pick *k* by **silhouette** over a small range, and run **K-Means** on all embeddings. This tier is mainly to show that **semantics** can be clustered; stricter train/test splits and NMI/ARI vs **category** can be layered on in later phases.

**Phase 1 focus:** a **clear, explainable** comparison of lexical baselines, LDA, and (optional) embeddings, with **Jaccard** / **JSD** for drift and **coherence** for topic readability. Heavier evaluation (held-out LDA, time-split clustering, hybrid models) is natural follow-on work.

---

## 5. Alignment with the codebase (what is actually implemented)

Everything in §3–§4 is wired in the repo at a **basic but real** level. Mapping:

| Idea in this document | Where it lives |
|----------------------|----------------|
| News stream + headline + short description + `category` + timestamps | `scripts/download_data.py` builds `text` from headline + `short_description`, keeps `timestamp` and `category`; default input is `data/sample/news_2018_h1.csv` after download. |
| Tier 1: frequency / TF-IDF per window, Jaccard between adjacent windows | `src/baselines.py` (`frequency_top_terms`, `tfidf_top_terms`); `src/metrics.py` (`mean_adjacent_jaccard`); orchestration in `src/main.py` → `run_baselines`. |
| Time bins (e.g. weekly) | `src/data_utils.py` (`add_time_bins`); CLI `--window` (default `7D`) in `src/main.py`. |
| Tier 2: LDA on the full corpus (Phase 1 simple), perplexity + coherence | `src/lda_model.py` (`train_lda`, `get_perplexity`, `get_coherence`); `src/main.py` → `run_lda` sorts by `timestamp` then fits on all token lists. |
| u_mass coherence (Mimno-style) | `src/lda_model.py` (`get_coherence`); called from `run_lda`. |
| Mean topic mix per bin + JSD between adjacent bins | `src/main.py` (`run_lda`): `doc_topic_matrix` + `mean` per `time_bin`; `src/metrics.py` (`mean_adjacent_jsd`). |
| Tier 3: Sentence-Transformer embeddings (frozen encoder) | `src/embeddings.py` (`encode_texts`, default `all-MiniLM-L6-v2`; no fine-tuning). |
| Choose *k* by silhouette (Phase 1 demo: all rows) | `src/embeddings.py` (`find_best_k`); `src/main.py` → `run_embeddings` picks *k* in `[2, min(6, n−1)]` on the full embedding matrix. |
| K-Means on full embeddings | `src/embeddings.py` (`cluster`); used in `run_embeddings` (simple Phase 1 path; not a time-held-out split). |
| NMI / ARI | Not printed in simplified `main.py`; can be added later or run in a notebook. |
| TOT not implemented | Confirmed: no TOT model; only fixed windows + JSD as a **proxy** for topical drift. |
| Inference for LDA | We use **gensim** `LdaModel` (online variational Bayes–style updates), not Gibbs sampling from the original Blei et al. paper — same model family, different inference algorithm. |

Run all tiers: `python -m src.main --mode all` (after `python scripts/download_data.py`).

---

## References

- Blei, D., Ng, A., & Jordan, M. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.
- Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing Semantic Coherence in Topic Models. *EMNLP*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
- Wang, X., & McCallum, A. (2006). Topics over Time: A Non-Markov Continuous-Time Model of Topical Trends. *KDD*.
