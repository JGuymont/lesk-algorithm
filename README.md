# COMP550-assignment03

- COMP 550, Fall 2018
- TA: Kian Kenyon-Dean
- Due: November 12th, 2018 at 11:59pm.

## Question 2: Lesk's Algorithm (80 points)
Implement and apply Lesk's algorithm to the publicly available data set of SemEval 2013 Shared Task
\#12 (Navigli and Jurgens, 2013), using NLTK's interface to WordNet v3.0 as your lexical resource.
(Be sure you are using WordNet v3.0!) The relevant les are available on the course website. Starter
code is also provided to help you load the data. More information on the data set can be found at
https://www.cs.york.ac.uk/semeval-2013/task12/.\\

The provided code will load all of the cases that you are to resolve, along with their sentential context.
Apply word tokenization and lemmatization (you have code to do this from A1) as necessary, and remove
stop words.\\

1. [ ] As a first step, compare the following two methods for WSD:

    1. [ ] The most frequent sense baseline: this is the sense indicated as #1 in the synset according toWordNet
    2. [ ] NLTK's implementation of Lesk's algorithm (nltk.wsd.lesk) Use accuracy as the evaluation measure. There is sometimes more than one correct sense annotated in the key. If that is the case, you may consider an automatic system correct if it resolves the word to any one of those senses. What do you observe about the results?

2. Develop two additional methods to solve this problem. 

    1. [ ] One of them must combine **distributional information** about the frequency of word senses, and the standard Lesk's algorithm. 

    2. [ ] The other may be any other method of your design. The two methods must be substantially different; they may not be simply the same method with a different parameter value. Make and justify decisions about any other parameters to the algorithms, such as what exactly to include in the sense and context representations, how to compute overlap, and how to trade of the distributional and the Lesk signal, with the use of the development set, which the starter code will load for you. You may use any heuristic, probabilistic model, or other statistical method that we have discussed in class in order to combine these two sources of information.

    For the last method of your design, you may use external corpora or lexical resources if you wish (e.g., thesauri, or WordNet in other languages, etc.), though it is not required. Feel free to use your creativity to find ways to improve performance!

Some issues and points to watch out for:
 The gold standard key presents solutions using lemma sense keys, which are distinct from the synset numbers that we have seen in class. You will need to convert between them to perform the evaluation. This [webpage](https://wordnet.princeton.edu/man/senseidx.5WN.html) explains what lemma sense keys are.
 The data set contains multi-word phrases, which should be resolved as one entity (e.g., latin america).
Make sure that you are converting between underscores and spaces correctly, and check that you
are dealing with upper- vs lower-case appropriately.
 We are using instances with id beginning with d001 as the dev set, and the remaining cases as the
test set, for simplicity. This is dierent from the setting in the original SemEval evaluation, so the
results are not directly comparable.

Discuss the results of your experiments with the three models. Also include a discussion of the successes
and diculties faced by the models. Include sample output, some analysis, and suggestions for improve-
ments. The entire report, including the description of your model, must be no longer than two pages.
Going beyond this length will result in a deduction of marks.
Your grade will depend on whether you adequately followed the guidelines above, whether you followed
standard model design and experimental procedure during the development of your method, and on the
quality of your report (both linguistic, and content).
What To Submit

Electronically: Submit a .pdf containing the written answers to Question 1 as well as the report part
of Question 2, called `a3-written.pdf'. For the programming part of Question 2, you should submit one
zip le called `a3-q2.zip' with your source code to MyCourses under Assignment 3.
Page 2