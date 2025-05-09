# DPIR Introduction to LLMs Course

## 👋 Welcome! 
This is a repository with the collection of teaching materials for the (Large) Language Models for Social Sciences course, thought at the DPIR Oxford in Spring 2025.

The course introduces LLMs from a social science perspective, focusing on how they work and applying them effectively to real-
world research tasks. To that end, the course provides both intuitions into key concepts and techniques behind current LLM-based systems (chat-bots, agents), and provides hands-on coding examples and exercises in Python, to help you strengthen your understanding and develop practical skills. 

## ⚠️ Disclaimers: 

### 🕒 We cover much more than we can fit in 8h total of class time 

- We have more material than we can cover 
- The best way for you to learn is to engage with material and suggested resources on your own (over several iterations)

### 🤔 The course is fairly advanced

- You need to be able to read and write Python code. If you don't, check out the [Quick Python Intro](https://github.com/antndlcrx/Intro-to-LLMs-DPIR/blob/main/preliminaries/quick_python_intro.ipynb) and the [Introduction to Python Course](https://github.com/antndlcrx/Intro-to-Python-DPIR).
- You need to have basic understanding of Deep Learning and PyTorch. If you don't, check out the [Quick Deep Learning and PyTorch Intro](https://github.com/antndlcrx/Intro-to-LLMs-DPIR/blob/main/preliminaries/quick_dl_torch_intro.ipynb). 
- You need to have a good understanding of basic NLP concepts, particularly word embeddings. If you don't, check out [Quick Embedding Intro](https://github.com/antndlcrx/Intro-to-LLMs-DPIR/blob/main/preliminaries/quick_embedding_intro.ipynb).
- If you lack either, **you will struggle**, but **do not get discouraged**! Learning is an iterative process: each time you go over the material, you identify aspects that you need to understand better, you work on them, and get back. After several iterations, you will develop good intuitions and confidence! 

### 🛠️ This course is still largely in development 

- I am still working on optimising the sequence of sessions, and materials themselves 
- Your feedback is greatly appreciated! 

## 🔭 Outlook: 

Course has four main sessions: 

1. 🏛️ LLM Fundamentals 
2. ✍️ Prompting, Classification, LLM Bias
3. 🧰 Equipping LLMs with tools: RAG and Agents 
4. 🧩 LLM Reasoning
 

## 🏛️ LLM Fundamentals 

This is a key session in which you will learn what language models are, how they are build, and how they generate text sequences. When you are interacting with, say, a chat-bot, there are multiple fascinating concepts and ideas at play (what is a meaning of a word? what is a meaning of a sentence? how do we encode them?), as well as smart and elegant (and at times redundant) algorithms (tokenization, attention, backpropagation, etc.) that put these concepts to life.

Their particular implementations determine how well a model perfoms, both as a general model of language and as a tool for solving your task. For instance, choosing appropriate way to tokenize text can make or break model's ability to write code. Therefore, having a good understanding of each of the building blocks of an LLM is of great importance. 

This session introduces: 
- What does it mean to build a model of a language? 
- What do we practically need to do to build a language model?
- How to process human readable text input into machine readable input and back?
- Model architecture (transformer and attention)
- How does a language model generate text 

All of these concepts are introduced in [**this notebook**](https://colab.research.google.com/drive/1YWB9K0iPn9A-HTW1Dw93UKiINodFFVI5?usp=sharing). 

To create it, I relied on a selection of an amazing set of educational ressources, which I list below and highly reccomend to you that you read and interact with them: 

### 📚 Core ressources 

- [Language Modelling NLP Course for You](https://lena-voita.github.io/nlp_course/language_modeling.html).
- [Raschka, S. (2024). Build a Large Language Model (From Scratch)](https://learning.oreilly.com/library/view/build-a-large/9781633437166/) 
- [Hugging Face NLP Course (Chapter 6): Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/en/chapter6/5).
- [Explore Tokenizers via Tiktokenizer app](https://tiktokenizer.vercel.app/).
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)


### 🧠 Extra ressources 
- [Perplexity of fixed-length models
by 🤗](https://huggingface.co/docs/transformers/en/perplexity).
- [Andrej Karpathy's "Let's Build a GPT Tokenizer" video](https://www.youtube.com/watch?v=zduSFxRajkE).
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [LLM Visualisation](https://bbycroft.net/llm)
- [Transformer Explainer (Polo Club)](https://poloclub.github.io/transformer-explainer/)

### 📄 Papers 

- ["Attention is all you need" by Vaswani et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- ["Neural Machine Translation of Rare Words with Subword Units"
by Sennrich et al. (2015)](https://arxiv.org/abs/1508.07909)

## ✍️ Prompting, Classification, LLM Bias

In this session, we will talk about different ways to do classification (text annotation), which is arguably the most common social-science use case for LLMs. 
We will play with two different approaches to classification: fine-tuning (where we change model parameters to perform our specific task), and in-context learning, where we give model instructions (prompts) to perform our tasks, without touching any model parameters. 

[Classification Notebook](https://colab.research.google.com/drive/1cjm1nVQo6xa5KlJKJbiHUDwl-XsGtR71?usp=sharing).

We further address model bias, where it comes from, how to detect it, and which strategies we can use to mitigate it. 

[Bias Notebook](https://colab.research.google.com/drive/1ijNbhEaj_f-3Tg3MHuqcdLxXtmy2yj6w?usp=sharing).

We will introduce the key set of tools, which is the 🤗 [Hugging Face Ecosystem](https://huggingface.co/).
Particularly, we will work with the following libraries: 
- [`transformers`](https://github.com/huggingface/transformers): For accessing and running pretrained models like GPT-2, BERT, and more.  
- [`datasets`](https://huggingface.co/docs/datasets): For loading, manipulating, and analyzing NLP datasets at scale. 


### Papers

- ["SentimentisNotStance:Target-AwareOpinion
 Classification for Political Text Analysis" by Bestvater and Monroe (2022)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/743A9DD62DF3F2F448E199BDD1C37C8D/S1047198722000109a.pdf/sentiment-is-not-stance-target-aware-opinion-classification-for-political-text-analysis.pdf) 
- ["Bias and Fairness in Large Language Models: A Survey" by Gallegos et al. (2024)](https://aclanthology.org/2024.cl-3.8/)
- ["From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models" by Feng et al. (2023)](https://arxiv.org/pdf/2305.08283.pdf)
- [""I'm sorry to hear that": Finding New Biases in Language Models with a Holistic Descriptor Dataset" by Smith et al. 2022](https://arxiv.org/abs/2205.09209)


## DPIR Methods Spring School Materials 

### DPIR Methods Spring School 2025

#### 🧠 Preference Tuning (Instruction Tuning + RLHF)  

[Notebook Link](https://colab.research.google.com/drive/1ijNbhEaj_f-3Tg3MHuqcdLxXtmy2yj6w?usp=sharing)

#### 🔍 Retrieval-Augmented Generation (RAG)  

[Notebook Link](https://colab.research.google.com/drive/1AqmADxZYeOtsFNrskiJFuit9w3Dtiil-?usp=sharing)

#### 🧠 Reasoning  

[Notebook Link](https://colab.research.google.com/drive/1nnm1R7rdIRt1iKvBKFqzKv5HlNelYTul?usp=sharing)


### DPIR Spring School 2024
This repository contains material for the workshops taught at the DPIR Spring School 2024

- [Learning Python](https://github.com/antndlcrx/oss_2024/blob/main/tutorials/oss_python_intro.ipynb)
- [Introduction to LLMs](https://github.com/antndlcrx/oss_2024/blob/main/tutorials/oss_python_intro.ipynb)
