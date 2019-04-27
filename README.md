 
• Classic NLP Approach
Norvig’s Spelling Corrector
The idea is if we artificially generate all terms within maximum edit distance from the misspelled term, then the correct term must be among them. We have to look all of them up in the dictionary until we have a match. So all possible combinations of the 4 spelling error types (insert, delete, replace and adjacent switch) are generated. This is quite expensive with e.g. 114,324 candidate term generated for a word of length=9 and edit distance=2.

• SymSpell Algorithm
SymsSpell is an algorithm to find all strings within a maximum edit distance from a huge list of strings in very short time. It can be used for spelling correction. SymSpell derives its speed from the Symmetric Delete spelling correction algorithm and keeps its memory requirement in check by prefix indexing.
The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster (than the standard approach with deletes + transposes + replaces + inserts) and language independent.

• Machine Learning Approach
Word2Vec
It is an adaptation of Peter Norvig's spell checker. It uses word2vec ordering of words to approximate word probabilities. Indeed, Google word2vec apparently orders words in decreasing order of frequency in the training corpus. This kernel requires to download Google's word2vec: https://github.com/mmihaltz/word2vec-GoogleNews-vectors mmihaltz/word2vec-GoogleNews-vectors word2vec Google News model . Contribute to mmihaltz/word2vec-GoogleNews-vectors
