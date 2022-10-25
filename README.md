# Lost in Transfer and Translation

Sentiment analysis has been a prominent topic of research in numerous areas of application like journalism, marketing, finance, etc. In recent years, data sources rich with opinions have become widely available due to the prominence of social networks. However, some data reflects highly-resourced languages like English, but is found to be lacking in other low-resource languages. As the world becomes more interconnected through global communications, it is becoming more and more imperative to analyze data from all over the world. Lexicons are not always available in other languages (marking them "low-resource languages") and it remains an expensive task to construct and label them. 

This motivates me to build upon some existing approaches and run a sentiment analysis for different languages using relatively small datasets. For the sake of this project, the focus will be on Slavic languages.

Previous efforts in sentiment analysis for multilingual corpora entailed the translation of the corpora into English to then be analyzed for opinions. Researchers have used language-specific pre-trained models to vectorize English sentences for analysis (using models such as Word2Vec). These embeddings are then passed through supervised machine learning classifiers such as Random Forest, Support Vector Machines, and Gradient Boosting Trees--which provide decent accuracies on out-of-sample data.  In one study by Galshchuk, Jourdan, and Qiu in 2019, an accuracy of about 86\% was achieved using this method. 

I would like to explore whether there are any semantic details lost in the translation from a source language to English. It is often hard to capture the full connotative meanings of words between languages even in person-to-person communication. Therefore, I will be experimenting with transfer learning. Instead of translating from a source to English and risking the loss of some semantic detail in that translation, I propose using BERT to transfer knowledge from a pre-trained English paradigm and then fine-tuning the embeddings for a source language to then be analyzed for sentiment. This will be a domain-adaptation task, where I take a pre-trained BERT and freeze several of the first layers (trained on an English corpus) and re-train the final few layers on a low-resource language. This differs from multilingual BERT since that model was completely trained in other source languages, using high amounts of data from Wikipedia articles in those languages, whereas I would like to see the effects of transfer learning between two different languages. I would also like to perform several ablation studies, perhaps narrowing down which layers are more imperative for each language, whether there are some correlations between English and other languages that can be exploited, etc. This method is often referred to as cross-lingual transfer.


## Anticipated Schedule:
1. Week 1 (10/30/2022) Process Data (clean text) and implement tokenization.
2. Week 2 (11/06/2022) Implement BERT without frozen layers. Acquire baseline performance.
2
3. Week 3 (11/13/2022) Implement BERT with varying levels of frozen layers.
4. Week 4 (11/20/2022) Evaluate performance on different numbers of unfrozen layers for Tweet
sentiment analysis with low-resource languages.
5. Week 5 (11/27/2022) Debugging and/or further ablation studies space.
6. Week 6 (12/04/2022) Test mBERT on low-resource language Tweet Sentiment Analysis.
Test ELECTRA on low-resource language Tweet Sentiment.
7. Week 7 (12/11/2022) Finish Evaluations and prepare final report.
