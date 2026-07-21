# N-gram

## Project Architecture

### Training Pipeline

```text
Raw Corpus
     │
     ▼
tokenize_corpus()
     │
     ▼
build_vocabulary()
     │
     ├──────────────────────────────┐
     ▼                              │
count_unigrams()                    │
     │                              │
     ▼                              │
generate_ngrams(n=2)                │
     │                              │
     ▼                              │
count_ngrams()                      │
     │                              │
     ├───────────────────────┐      │
     ▼                       ▼      ▼
compute_bigram_probabilities()      compute_bigram_probabilities_smoothed()
               │                               │
               └───────────────┬───────────────┘
                               ▼
                  Bigram Probability Dictionary
```

### Sentence Evaluation Pipeline

```text
Sentence
    │
    ▼
tokenize_sentence()
    │
    ▼
compute_sentence_log_probability()
    │
    ├───────────────► Sentence Score
    │
    ▼
compute_perplexity()
```

### Next-Word Prediction Pipeline

```text
Sentence Prefix
      │
      ▼
tokenize_prefix()
      │
      ▼
Bigram Probability Dictionary
      │
      ├─────────────────────┐
      ▼                     ▼
predict_next_word()     predict_top_k()
```