---
layout: post
title: "document-retrieval-system"
date: 2019-08-27
tag: deep-learning
---

This will be a very quick post on document retrieval system, where we see, how we can retrieve the similar set of document on the basis of queries.

## Problem statement:
Let's say, we have `1000` documents unlabeled document. Our objective is to return the response of `query` document, which will be the most similar document (`semantic-wise`) This problem is similar to topic modelling.

### Few Approaches:
There are some topic modelling approach, which we will talk about later.
- Latent Semantic Analysis (LSA)
- Prob-LSA
- Latent Dirichlet Allocation(`Bayesian version of PSLA`)


## Document Retrieval Approach using deep learning.
We will go thorough step by step procedure to prepare the pipeline.
1. we prepare a `bag-of-words` for these document. For text cleaning, we remove punctuation, stop words  and spaces. We can even use stemming(`Removing the suffix such as -ize, -s, -es etc`).
2. We prepare count vector of `top-words` from all document. To use in neural network as fearture, we can prepare `tfidf` features(`avvreviated as term frequency inverse document frequency which is calculated as = term-freq * log(inv-doc-freq)`, where `term-freq = count-of-word / total-word` and `idf = log(toal-document / #document-in-which-word-appear )`)
3. Prepare a auto-encoder model, with very small latent dimensions. For example, if we select `5000` top-word, we can have choose following configuration:
    `5000 -> 1000 -> 200 -> 10 -> 200 -> 1000 -> 5000`
Note: our loss function is to reconstruct the original features
4. Now, we have latent features(`10 dims`), we extract this features, for each document and as well as the query. Now only step remaining is to find `cosine-similarity` between each document with the query. If we want to `categorize` each document, we can do the same using `similarity` metrics.
**Note**: For this we need to compare each pair, which can be huge (`NC2 combinations`)

That's it, we can get similar document as the query set, if NN is trained well.

### Applications of information retrieval
- better label of product on amazon (let's say, we have some wooden chair for child, it is labeled as furniture, now using the above method, we can find other product similar to this, from there we can have history of buyer and their other product and can estimate its more better label as `baby-product`)
- dicovering similar neughbourhood (for house price estimation, viloent/crime forecasting etc)
- structring web search results (categorize the result, for example we search for `watson`, it will show `ibm-watson`, `emma-watson` or other things, so we can display these result structurally based on categories)
- meta feature to train another model
