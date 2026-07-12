<style>
/* General note styling */  
  body {
    font-size: 130%;
  }

    img {
    display: block;
    margin: 20px auto;
    max-width: 70%;
    }

    .math {
    text-align: center;
}
</style>

# N-gram language model  

**Naive bayes asks:**  
given the class, how likely it is the word?
P(Word | class)

**N-gram asks:**  
 given the previous word(s) how likely is the next word?  
P(word_t | word_{t-1}, word_{t-2}, ...)  

## Table of contents

- [N-gram language model](#n-gram-language-model)
  - [Table of contents](#table-of-contents)
  - [Phase 1: Dataset \& Tokenization](#phase-1-dataset--tokenization)
  - [Phase 2: vocabulary](#phase-2-vocabulary)
    - [Vocabulary size vs corpus size](#vocabulary-size-vs-corpus-size)
  - [Phase 3: Unigram](#phase-3-unigram)
  - [Phase 4: N-grams function](#phase-4-n-grams-function)
    - [N-gram probabilities](#n-gram-probabilities)
  - [Phase 5: Bigrams](#phase-5-bigrams)
  - [Phase 6: Sentence Scoring](#phase-6-sentence-scoring)
    - [Why Log Probabilities?](#why-log-probabilities)
  - [Phase 7: Smoothing](#phase-7-smoothing)
    - [Why Does the Smoothing Algorithm Change?](#why-does-the-smoothing-algorithm-change)
    - [Analogy: Rolling a Die](#analogy-rolling-a-die)
    - [How Many Probabilities Does the Model Contain?](#how-many-probabilities-does-the-model-contain)
    - [Why Does the Denominator Change?](#why-does-the-denominator-change)
    - [Summary](#summary)


## Phase 1: Dataset & Tokenization

**Sentence boundary tokens**  

`<s>` represents the start of a sentence 
`/<s>` represents the end of a sentence 

For an N-gram model of order n, we usually add n-1 tokens

Bigram model:  
`<s>` I love NLP `</s>`

Trigram model:  
`<s> <s>` I love NLP `</s>`
why?  
Because the model needs enough previous context even at the beginning of the sentence

**Tokenizer**: converts raw text into sequence of tokens that the language model can process.

**Lowercasing reduces vocabulary size.**  

without lowercasing, `Machine` and `machine` are treated as different words, unnecessarily increasing the vocabulary.  

**Tokenization defines what a "word" is.** 

The language model does not understand raw text. It only sees the sequence of tokens produced by the tokenizer. Different tokenization strategies produce different vocabularies and therefore different probability estimates.

## Phase 2: vocabulary

vocabulary is the set of all unique tokens that appear in the corpus.  

### Vocabulary size vs corpus size

**Vocabulary size**: The number of unique words in the corpus.  

**Corpus size**: The number of word occurrence (including duplicates).  

**Example**:  

Corpus:
`["i", "love", "nlp"$$
["i", "love", "python"$$`

Vocabulary Size = 4
`{"i", "love", "nlp", "python"}`

Corpus Size = 6
`i, love, nlp, i, love, python`  

## Phase 3: Unigram

The unigram language model estimates the probability of a word without considering any surrounding context. A ungram count stores how many times each unique word appears in the corpus.  

Vocabulary answers: "what words exist?"  
Unigram counts answer: "How many times each word appear?"

**In Naive Bayes:**  

$$
P(word | class) = \frac{\text{count(word in class)}}{\text{total words in class}}
$$

In unigram we simply remove the conditioning:  
$$
P(word) = \frac{\text{count(word)}}{\text{total words}}
$$

The denominator is the total number of word occurrence (corpus size), not the vocabulary size. 

Example:  
$$
P(learning) = \frac{\text{count(learning)}}{ \sum_{\text{all words}} \text{count(words)}}
$$

Ungram probabilities always sum to 1.  

## Phase 4: N-grams function

### N-gram probabilities

The General rule of computing the probability of (N)-gram, we need:

- the count of the (N)-gram (numerator).
- the count of its (N-1)-gram prefix (denominator).

Examples:

- Unigram

$$
P(w)=\frac{\text{Count}(w)}{\text{Total number of words}}
$$

- Bigram

$$
\frac{\text{Count}(w_{i-1}, w_i)}
{\text{Count}(w_{i-1})}
$$

- Trigram

$$
\frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}
{\text{Count}(w_{i-2}, w_{i-1})}
$$

In general, an (N)-gram language model computes the probability of a word given its preceding context by dividing the count of the complete (N)-gram by the count of its (N-1) gram prefix.

## Phase 5: Bigrams

Formula:

$$
P ( \text{current word} | \text {previous word})
$$

The probability of current word given teh previous word = (the number of times they appear together) / the number of times the previous word appears as a context (as the first bigram word) 

$$
P(w_i | w_{i-1}) = \frac {count (w_{i-1},w_i) } {count(w_{i-1})}
$$

The denominator is not the total number of words anymore.  
In the unigram model we normalized over the entire corpus, while in the bigram mode we normalize within each context.  
i.e. Among all the times the previous word appeared, how often was it followed by the current word?

The core idea of statistical language model: **The probability of the next word depends on the context**

<br>
  
**Unigram vs. Bigram Probabilities**  

A unigram model assumes each word is independent of all other words. The probability of a word depends only on how frequently it appears in the corpus.  

A bigram model assumes that each word depends only on the immediately preceding word. The probability of a word is therefore conditional on its previous word.  

The key difference is that unigram probabilities are unconditional, while bigram probabilities model local context through conditional probability. 

## Phase 6: Sentence Scoring

**The mathematical definition**  

suppose our sentence is:

```
I love machine learning

```

A bigram model says: 
$
P(sentence)=P(i)×P(love∣i)×P(machine∣love)×P(learning∣machine)
$

This is the mathematical definition of the probability of the sentence 

<br>  

**The implementation problem**  

For longer sentence, suppose each probability is around `0.1` a 20 word sentence will becomes `0.1 × 0.1 × 0.1 × ...` 20 times that is `10^-20` which is very tiny.  

for real corpora, it becomes even smaller:  

``` bash
10^-80
10^-150
10^-300
...
```  

Eventually the computer can't represent numbers that small accurately. This is called **numerical underflow**

<br>  

**The solution**  

We use the logarithm identity:

$$
log (a × b × c) = log a + log b + log c
$$

---

### Why Log Probabilities?

The probability of a sentence is the product of the conditional probabilities of its words.

For long sentences, multiplying many small probabilities can produce extremely small numbers that computers cannot represent accurately. This problem is called **numerical underflow**.

To avoid this, language models use logarithms.

Using the identity

$
\log(ab)=\log(a)+\log(b),
$

the product of probabilities becomes a sum of log probabilities.

This preserves the relative ordering of sentence probabilities while providing much better numerical stability.


## Phase 7: Smoothing

Smoothing is a technique that assigns a small non-zero probability to unseen n-grams by adding a constant value to the counts of all possible n-grams before computing their probabilities.

Laplace smoothing:

$$
P(w_i \mid w_{i-1}) = \frac{\operatorname{Count}(w_{i-1},w_i)+k} {\operatorname{Count}(w_{i-1})+kV}
$$

where:

- **k** is the smoothing constant (gives every possible continuation some probability).
- **V** is the vocabulary size (the number of unique words in the vocabulary).

---

### Why Does the Smoothing Algorithm Change?

This was one of the most important conceptual insights of the project.

The algorithm changes **not because of the formula**, but because **the mathematical model itself changes**.

---

**Before Smoothing (Maximum Likelihood Estimation)**  

Suppose the training corpus contains:

```text
love machine
love python
```

The model has observed exactly two bigrams:

```text
(love, machine)
(love, python)
```

Maximum Likelihood Estimation (MLE) asks a very simple question:

> **"What is the probability of the events that I observed?"**  

Nothing more.

Therefore, it is sufficient to iterate only over the observed bigrams.

```text
Observed events
        ↓
Iterate over observed bigrams
```

The implementation naturally looks like:

```python
for bigram in counts_bigram:
    ...
```

because the model only assigns probabilities to events that actually occurred in the training corpus.

---

**What Changes with Laplace Smoothing?**  

Laplace smoothing makes a different assumption.

Instead of saying:

> "If I never observed an event, its probability is zero."

it says:

> **"Just because I never observed an event does not mean it is impossible."**

This is not simply a programming trick.

It is a **change in the mathematical model**.

Instead of modeling only the observed events, the model now considers **every possible event**.

---

### Analogy: Rolling a Die

Suppose a six-sided die is rolled three times and the results are:

```text
1
3
6
```

Maximum Likelihood Estimation would estimate:

```text
P(1) = 1/3
P(3) = 1/3
P(6) = 1/3
P(2) = 0
P(4) = 0
P(5) = 0
```

The model only assigns probability to outcomes that were observed.

---

Laplace smoothing says:

> "Just because a 2 hasn't appeared yet doesn't mean rolling a 2 is impossible."

Therefore, every face of the die receives some probability.

The sample space becomes:

```text
1
2
3
4
5
6
```

instead of only

```text
1
3
6
```

---

**The Same Idea for Bigrams**  

Before smoothing, after the word:

```text
love
```

the model only considers the observed continuations:

```text
machine
python
```

After smoothing, the model considers **every word in the vocabulary** as a possible continuation:

```text
i
love
machine
learning
pizza
...
```

Even if some of these bigrams never appeared in the training corpus, they still receive a small probability.

---

**Why the Algorithm Must Change**  

Before smoothing:

```text
Observed events
        ↓
Iterate over observed bigrams
```

After smoothing:

```text
Possible events
        ↓
Iterate over every possible bigram
```

The algorithm changes because the **probability distribution changes**.

MLE only models what happened.

Laplace smoothing models what **could happen**.

---

### How Many Probabilities Does the Model Contain?

Assume the vocabulary contains **V** unique words.

For a fixed previous word \(w_{i-1}\), every vocabulary word is a possible next word.

Therefore, each previous word defines a conditional probability distribution containing exactly **V** probabilities:

$$
P(\cdot \mid w_{i-1})
$$

For example, if

```text
Vocabulary = {I, love, machine, learning}
```

then after the previous word **love**, the model contains the probabilities

```text
P(I | love)
P(love | love)
P(machine | love)
P(learning | love)
```

Notice that these probabilities form **one conditional probability distribution**, and therefore

$$
\sum_{w_i \in V} P(w_i \mid w_{i-1}) = 1
$$

Since there are **V** possible previous words, the complete bigram model contains

$$
V \times V = V^2
$$

possible **bigram probabilities**.

---

**Without Smoothing (MLE)**  

The model stores probabilities only for **the bigrams that were observed** in the training corpus.

If only **N** distinct bigrams appear in the corpus, then only those **N** probabilities are computed.

```text
Corpus
    ↓
Observed bigrams
    ↓
N stored probabilities
```

---

**With Laplace Smoothing**  

Laplace smoothing assumes that every vocabulary word is a possible continuation, even if that bigram never appeared in the training corpus.

Therefore, the model computes and stores probabilities for **every possible bigram** that can be formed from the vocabulary.

In other words:

> For every previous word, the model computes probabilities for **all V possible next words**.

```text
Vocabulary (V words)
        ↓
Every possible bigram (V²)
        ↓
V² stored probabilities
```

Many of these bigrams may never appear in the training corpus, but each still receives a small non-zero probability.

---

**Key Insight**  

Before smoothing, the probability dictionary represents:

> **Everything that happened.**

After smoothing, the probability dictionary represents:

> **Everything that could happen.**

This is why the implementation changes from iterating over the observed bigrams to iterating over every possible bigram that can be formed from the vocabulary.

The algorithm changes **because the probability model changes**, not because the formula changes.

---

### Why Does the Denominator Change?

Laplace smoothing adds the smoothing constant **k** to **every possible continuation**.

If the vocabulary contains **V** unique words, then each previous word has exactly **V** possible next words.

Adding **k** to every possible continuation increases the total count by

$$
kV
$$

Therefore, the denominator must also increase by **kV** to keep the conditional probabilities normalized.

The smoothed probability becomes

$$
P(w_i \mid w_{i-1}) = \frac{\operatorname{Count}(w_{i-1},w_i)+k}{\operatorname{Count}(w_{i-1})+kV}
$$

where:

- **k** is the smoothing constant.
- **V** is the vocabulary size.

Without increasing the denominator, the conditional probabilities would no longer sum to 1.

---

### Summary

**Maximum Likelihood Estimation (MLE)**  

- Models only the observed events.
- Iterates only over observed n-grams.
- Assigns zero probability to unseen events.
- Stores probabilities only for observed bigrams.

**Laplace Smoothing**  

- Models the entire sample space.
- Iterates over every possible n-gram that can be formed from the vocabulary.
- Assigns a small non-zero probability to unseen events.
- Stores probabilities for every possible bigram (\(V^2\) possible bigram probabilities).
- Preserves a valid probability distribution by increasing the denominator by \(kV\).