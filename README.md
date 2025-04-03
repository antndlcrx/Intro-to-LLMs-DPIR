# DPIR Spring School 2025

## 🧠 What We’ve Covered So Far

### ✨ What Is a Language Model?  
A **language model (LM)** estimates how likely a sequence of words or tokens is.  
It does this by breaking the sequence into **conditional probabilities**:

$$
P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot ... \cdot P(w_n | w_1, ..., w_{n-1})
$$

➡️ *Why it matters:* If a model can assign high probability to real, grammatical, meaningful text — it likely understands something about how language works.

---

### ✂️ Tokenization  
Before training or inference, text is **tokenized**:  
- Split into chunks (tokens), which could be words, subwords, or characters.  
- Each token is mapped to a unique ID for the model to process.

$$
\text{Text} \rightarrow \text{Tokens} \rightarrow \text{Token IDs} \rightarrow \text{Embeddings}
$$

➡️ *Why it matters:* Tokenization defines the granularity of information the model sees, and can introduce bias or loss of meaning (especially across languages/dialects).

---

### 🧱 Word Embeddings  
Words are represented as **vectors** in a continuous space, learned from co-occurrence patterns in text. These **embeddings** allow models to compute relationships between words (e.g., similarity, analogy).  
➡️ *Why it matters:* Word embeddings are the **first layer** in a language model — every token gets converted into an embedding before any processing begins.

---

### 🔧 Training an LM  
To learn these probabilities, we train a neural network to predict the **next token** in a sequence, given its context.  
This is done via **gradient descent**:  
- The model outputs a probability distribution over the vocabulary.  
- We compute **cross-entropy loss** between predicted and actual tokens.  
- Gradients of this loss are used to update parameters.  
➡️ *Why it matters:* Training teaches the model to encode linguistic patterns, world knowledge, and context dependencies.

---

### 🧠 Architecture: The Neural Network  
At the core of every LM is a neural architecture that turns token embeddings into predictions.  
We started with simple architectures like **RNNs**, which process tokens one at a time. But RNNs struggle with long-range dependencies and parallelization.

➡️ *Why it matters:* The **architecture defines the model's capacity** to learn from context. That’s why Transformers (next!) were such a breakthrough.

---

### 🔀 The Transformer  
Transformers process entire sequences **at once**, using **self-attention** to decide which tokens to pay attention to.  
They consist of repeated blocks with:
- **Multi-head attention**: lets each token learn from all others.
- **Feedforward layers**: refine the token's internal representation.
- **LayerNorm, skip connections, and dropout**: help stabilize and generalize learning.

➡️ *Why it matters:* Nearly all modern LLMs (GPT, BERT, LLaMA) are built on Transformers. It’s the most important architectural advance in NLP.

---

### 🤗 Hugging Face Ecosystem  
We introduced the **Hugging Face Hub**, a platform for sharing and using models, datasets, and evaluation tools. Throughout the course, we will work with several of Hugging Face’s most powerful libraries:

- [`transformers`](https://github.com/huggingface/transformers): For accessing and running pretrained models like GPT-2, BERT, and more.  
- [`datasets`](https://huggingface.co/docs/datasets): For loading, manipulating, and analyzing NLP datasets at scale.  
- [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines): A high-level interface for common tasks like generation, classification, and translation.

These tools allow us to:
- Quickly prototype LLM-based tasks  
- Explore and evaluate bias  
- Analyze model predictions with minimal code

➡️ *Why it matters:* The Hugging Face ecosystem has become the de facto standard for working with LLMs in both research and industry. Mastering it will give you the ability to build powerful workflows, test your ideas, and run real-world analyses.

--- 

### ⚖️ Bias in Language Models  
We discussed how LMs trained on web-scale data can **amplify social bias**:
- From training corpora (representation gaps, slurs)
- From tokenization (some dialects get fragmented)
- From alignment processes (e.g., fine-tuning with biased human feedback)

We introduced **HolisticBias** as a benchmark to evaluate completions across social identities, and used the `pipeline` API to analyze generated text for sentiment.

➡️ *Why it matters:* Social scientists must understand and **audit model behavior**, especially when applying LMs to people-centered domains.

---

## 🔥 Today’s Focus: Transformers & Pretrained Models

---

### 🧩 Prompt Engineering  
Learn how to guide model behavior with well-crafted prompts — especially for tasks like classification, sentiment and stance prediction, and using LMs as proxies for studying human behaviour.

### 🧪 Fine-Tuning  
We’ll explore how to further **train** a model on your own dataset to better match your domain or goals.

### 🧠 Preference Tuning (Instruction Tuning + RLHF)
Learn how models are aligned with human values and goals.
We’ll cover:

Instruction Tuning – training models to follow natural language instructions

RLHF – a method for teaching models to prefer responses humans rate as more helpful, safe, or aligned
These methods are essential for developing assistant-like LMs and understanding how subjective judgments get encoded into model behavior.

### 📊 Social Science Applications  
Use models to:
- Simulate or emulate human responses
- Analyze public opinion

---

## 👀 Coming Up Next

Tomorrow, we shift focus to **applying LLMs to tasks**:


### 🧠 Reasoning and Alignment  
We’ll explore:
- When models can reason, and why
- What is reasoning to begin with and where can it be helpful? 
- How to endow models with tools

---


# DPIR Spring School 2024
This repository contains material for the workshops taught at the DPIR Spring School 2024

- [Learning Python](https://github.com/antndlcrx/oss_2024/blob/main/tutorials/oss_python_intro.ipynb)
- [Introduction to LLMs](https://github.com/antndlcrx/oss_2024/blob/main/tutorials/oss_python_intro.ipynb)


