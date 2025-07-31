# Masked Language Modeling with BERT in Multi-GPU Settings

[![Run on Kaggle](https://img.shields.io/badge/Kaggle-Run%20Notebook-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/reshalfahsi/mlm-bert-multi-gpu)

This project demonstrates how to train a masked language model using BERT in a multi-GPU environment on Kaggle. The training process is powered by HuggingFace's ``transformers`` and ``accelerate`` libraries, making distributed training seamless and accessible. The model is fine-tuned on the Wikitext dataset provided by Salesforce.

## 🧠 What is BERT?

BERT is a deep bidirectional transformer-based model pre-trained on large corpora using masked language modeling (MLM) and next sentence prediction (NSP). RoBERTa later showed NSP might not be necessary. BERT is based on the Transformer encoder stack:

| Component            | Description                                        |
| -------------------- | -------------------------------------------------- |
| **Layers**           | 12 (base) or 24 (large) Transformer encoder layers |
| **Hidden size**      | 768 (base), 1024 (large)                           |
| **Attention heads**  | 12 (base), 16 (large)                              |
| **Max input length** | 512 tokens                                         |
| **Total parameters** | \~110M (base), \~340M (large)                      |

Each layer consists of:

- Multi-head self-attention (bidirectional)
- Feed-forward layers
- LayerNorm and residual connections

As aforementioned, BERT is trained by means of masked language modeling a.k.a to predict the original token at each masked position. Hence, at the training time:

- Randomly mask 15% of input tokens.
- Of masked tokens:
  - 80% replaced with ``[MASK]``.
  - 10% replaced with random tokens.
  - 10% left unchanged.

In BERT, bidirectionality comes from how self-attention is configured:
➡️ each token can attend to all other tokens, regardless of their position — both before and after in the sequence. By default:

- There’s no causal mask restricting attention.
- That means every token can attend to every other token, including ones to the left and right.


## 🚀 Project Highlights

- Fine-tune a pre-trained BERT model (``bert-base-uncased``) on the Wikitext corpus.
- Use HuggingFace's ``datasets``, ``transformers``, and ``accelerate`` libraries.
- Leverage multiple GPUs (2 × NVIDIA T4) on Kaggle for parallel training.

## Implementation Overview

These are some key snippets from the project. For full implementation, please check the [notebook](./mlm-bert-multi-gpu.ipynb) on [Kaggle](https://www.kaggle.com/code/reshalfahsi/mlm-bert-multi-gpu).

### 📂 Dataset

The dataset used is ``wikitext-2-raw-v1``, accessible via HuggingFace Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

We concatenate and tokenize the text, then split it into chunks suitable for BERT's input size (typically ``512`` tokens).

### 🧰 Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 🤖 Model

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
```

### 🚗 Accelerated Multi-GPU Training

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

## 📊 Results

### Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/mlm-bert-multi-gpu/blob/main/assets/loss_plot.png" alt="loss_plot" > <br /> Loss curves of BERT on the Wikitext dataset. </p>

### Unmasking the Masked Words

Below are some samples of unmasked sentences. There are single and multiple masked input sentences that are being unmasked.

#### Example #1

```
Input sentence: Learning a new language can be [MASK] but rewarding.
Top 5 predictions:
  Token: 'difficult' | Score: 0.4400 | Sequence: 'learning a new language can be difficult but rewarding.'
  Token: 'challenging' | Score: 0.2132 | Sequence: 'learning a new language can be challenging but rewarding.'
  Token: 'painful' | Score: 0.0563 | Sequence: 'learning a new language can be painful but rewarding.'
  Token: 'frustrating' | Score: 0.0398 | Sequence: 'learning a new language can be frustrating but rewarding.'
  Token: 'awkward' | Score: 0.0293 | Sequence: 'learning a new language can be awkward but rewarding.'
```

#### Example #2

```
Input sentence: You are my [MASK] that shines my [MASK].
Top 5 predictions:
  Token: 'light' | Score: 0.4667 | Sequence: 'you are my light that shines my light.'
  Token: 'way' | Score: 0.1478 | Sequence: 'you are my light that shines my way.'
  Token: 'world' | Score: 0.0333 | Sequence: 'you are my light that shines my world.'
  Token: 'life' | Score: 0.0295 | Sequence: 'you are my light that shines my life.'
  Token: 'lamps' | Score: 0.0215 | Sequence: 'you are my light that shines my lamps.'
```

### Example #3

```
Input sentence: This is the [MASK] [MASK] of my [MASK]!
Top 5 predictions:
  Token: 'life' | Score: 0.4139 | Sequence: 'this is the final part of my life!'
  Token: 'story' | Score: 0.0861 | Sequence: 'this is the final part of my story!'
  Token: 'plan' | Score: 0.0726 | Sequence: 'this is the final part of my plan!'
  Token: 'journey' | Score: 0.0359 | Sequence: 'this is the final part of my journey!'
  Token: 'career' | Score: 0.0211 | Sequence: 'this is the final part of my career!'
```

### Example #4

```
Input sentence: You are [MASK] [MASK] [MASK] [MASK].
Top 5 predictions:
  Token: 'person' | Score: 0.4856 | Sequence: 'you are not the same person.'
  Token: 'thing' | Score: 0.1050 | Sequence: 'you are not the same thing.'
  Token: 'man' | Score: 0.0715 | Sequence: 'you are not the same man.'
  Token: 'age' | Score: 0.0244 | Sequence: 'you are not the same age.'
  Token: 'anymore' | Score: 0.0213 | Sequence: 'you are not the same anymore.'
```

### Example #5

```
Input sentence: He [MASK] [MASK] [MASK] [MASK] [MASK].
Top 5 predictions:
  Token: 'man' | Score: 0.2042 | Sequence: 'he had been a good man.'
  Token: 'boy' | Score: 0.1336 | Sequence: 'he had been a good boy.'
  Token: 'friend' | Score: 0.1069 | Sequence: 'he had been a good friend.'
  Token: 'guy' | Score: 0.0900 | Sequence: 'he had been a good guy.'
  Token: 'kid' | Score: 0.0385 | Sequence: 'he had been a good kid.'
```

### Example #6

```
Input sentence: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]!
Top 5 predictions:
  Token: 'life' | Score: 0.1181 | Sequence: 'the best of my own life!'
  Token: 'time' | Score: 0.1100 | Sequence: 'the best of my own time!'
  Token: 'kind' | Score: 0.0519 | Sequence: 'the best of my own kind!'
  Token: 'country' | Score: 0.0284 | Sequence: 'the best of my own country!'
  Token: 'making' | Score: 0.0280 | Sequence: 'the best of my own making!'
```

## **Reference**

1. [J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol.*, vol. 1, pp. 4171–4186, 2019.](https://aclanthology.org/N19-1423.pdf)
2. [S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer Sentinel Mixture Models," *arXiv preprint arXiv:1609.07843*, 2016.](https://arxiv.org/pdf/1609.07843)
3. [HuggingFace's Accelerate](https://github.com/huggingface/accelerate)
4. [Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.](https://arxiv.org/pdf/1907.11692)
5. [A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
6. [T. Wolf et al., "HuggingFace's transformers: State-of-the-art natural language processing," *arXiv preprint arXiv:1910.03771*, 2019.](https://arxiv.org/pdf/1910.03771)