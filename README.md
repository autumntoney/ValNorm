# ValNorm
Code and validation datasets used to generate ValNorm scores from ValNorm: A New Word Embedding Intrinsic Evaluation Method Reveals Valence Biases are Consistent Across Languages and Over Decades [on arXiv](https://arxiv.org/abs/2006.03950). 

@article{toney2020valnorm,
  title={ValNorm: A New Word Embedding Intrinsic Evaluation Method Reveals Valence Biases are Consistent Across Languages and Over Decades},
  author={Toney, Autumn and Caliskan, Aylin},
  journal={arXiv preprint arXiv:2006.03950},
  year={2020}
}

## ValNorm.py 
To generate ValNorm scores use the WordEmbeddingFactualAssociation() function and provide the word embedding set (semanticModel) and the vocabulary list to test (vocabToTest). WordEmbeddingFactualAssociation() returns a table with each row containing a vocabulary word, the effect size (ValNorm score), and the p-value.

We use 3 different vocabulary lists:
1. [Bellezza's Lexicon](https://github.com/autumntoney/ValNorm/blob/master/Validation%20Datasets/Bellezza_Lexicon.csv)
2. [ANEW](https://github.com/autumntoney/ValNorm/blob/master/Validation%20Datasets/ANEW.csv)
3. [Warriner's Lexicon](https://github.com/autumntoney/ValNorm/blob/master/Validation%20Datasets/Warriner_Lexicon.csv)

The non-English vocabulary lists are located in the [non-English Validation Datasets](https://github.com/autumntoney/ValNorm/tree/master/non-English%20Validation%20Datasets) folder

We use 7 different word embedding sets:
1. [FastText Wikipedia & Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)
2. [FastText Common Crawl](https://fasttext.cc/docs/en/english-vectors.html)
3. [FastText OpenSubtitles (subs2vec)](https://github.com/jvparidon/subs2vec)
4. [GloVe Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)](https://nlp.stanford.edu/projects/glove/)
5. [GloVe Twitter](https://nlp.stanford.edu/projects/glove/)
6. [Word2Vec Google News](https://code.google.com/archive/p/word2vec/)
7. [ConceptNet Numberbatch (version 19.08)](https://github.com/commonsense/conceptnet-numberbatch) 
