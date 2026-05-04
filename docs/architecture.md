# Phase 3 Hybrid Architecture

```mermaid
flowchart LR
    A["News stream CSV<br/>N documents x {timestamp,text,category}"] --> B["Preprocess<br/>tokens: list[list[str]]<br/>time_bin: 7D"]

    B --> C1["Symbolic ML branch<br/>Top-k term sets per bin<br/>shape: T x K"]
    C1 --> D1["Lexical drift<br/>1 - Jaccard(top_terms_t, top_terms_t+1)<br/>shape: T-1"]

    B --> C2["Probabilistic ML branch<br/>Gensim LDA<br/>doc-topic matrix theta<br/>shape: N x num_topics"]
    C2 --> D2["Topic drift<br/>JSD(mean(theta_t), mean(theta_t+1))<br/>shape: T-1"]

    B --> C3["Neural DL branch<br/>SentenceTransformer MiniLM<br/>embedding matrix X<br/>shape: N x 384"]
    C3 --> D3["Semantic drift<br/>1 - cosine(mean(X_t), mean(X_t+1))<br/>shape: T-1"]

    D1 --> E["Fusion layer<br/>min-max normalize each signal<br/>hybrid = .30 lexical + .56 LDA + .14 semantic<br/>shape: T-1"]
    D2 --> E
    D3 --> E

    E --> F["Hybrid event ranking<br/>top percentile transitions"]
    B --> G["Weak validation target<br/>category-distribution JSD<br/>shape: T-1"]
    F --> H["Ablation table<br/>ML-only vs DL-only vs Hybrid<br/>precision / recall / F1"]
    G --> H
```

## Fusion Mechanism

The hybrid score is not a simple vote across disconnected models. Each model
describes a different failure mode:

- Lexical drift catches exact-term spikes but misses paraphrases.
- LDA drift captures interpretable topic-mixture movement but can blur short
  texts.
- MiniLM embedding drift catches semantic changes even when vocabulary changes.

The fusion layer aligns all three signals on identical adjacent time-bin
transitions, normalizes them within the same held-out run, and computes a
weighted score:

```text
Hybrid(t -> t+1) =
  0.30 * normalized_lexical_drift +
  0.56 * normalized_lda_jsd +
  0.14 * normalized_embedding_cosine_drift
```

The ablation mode removes components and evaluates the same transitions against
a weak real-data target: shifts in HuffPost editor category distribution. This
does not pretend categories are perfect topic labels; it gives a reproducible
external signal for whether detected transitions correspond to real editorial
mix changes.
