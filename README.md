# Sentiment Analysis on Twitter Data

In this project, we have applied sentiment analysis and four statistical machine learning models, KNN, Naive Bayes Classifier, Random Forest and Support Vector Regression. These models are used to depict the correlation between the tweets which are extracted from twitter. We have performed sentiment analysis of the twitter data. The purpose of this repo is to compute the sentiment of tweets posted recently on twitter about any field.

TABLE OF CONTENTS

	Background
	Dataset
	Requirement
	Technical Approach
	Limitations and Future Improvements
	Reference
	
# Background
	
Sentiment Analysis is a branch of Natural Language Processing (NLP) that allows us to determine algorithmically whether a statement or document is “positive” or “negative”. It's a technology of increasing importance in the modern society as it allows individuals and organizations to detect trends in public opinion by analyzing social media content. Keeping abreast of socio-political developments is especially important during periods of policy shifts such as election years, when both electoral candidates and companies can benefit from sentiment analysis by making appropriate changes to their campaigning and business strategies respectively.

# Dataset

	TweetList.csv: classified Twitter data containing a set of tweets which have been analyzed and scored for their sentiment
	TweetList_Cleaned.csv: Twitter data containing a set of tweets after removing all the unneccessary characters and words.

# Requirement

	Numpy, Scipy, Scikit, Matplotlib, Pandas, NLTK.


# Technical-Approach
  
	 1.Data cleaning: Design a procedure that prepares the Twitter data for analysis
		Remove all html tags and attributes (i.e., /<[^>]+>/)
		Replace Html character codes (i.e., &...;) with an ASCII equivalent
		Remove all URLs
		Remove all characters in the text are in lowercase
		Remove all stop words are removed
		Preserve empty tweet after pre-processing


	2. Exploratory analysis
		Determine the tweets from twitter.
		Visualization

	3. Model preparation
		Classification algorithms: k-NN, Naive Bayes, SVM, Random Forest
		Features: Bag of Words (word Cloud),TF-IDF, CountVectorizer

	4. Model implementation and tuning
		Train classification model to predict the sentiment value (positive or negative)
		Train multi-class classification models to predict the reason for the negative tweets.

# Limitations and Future Improvements

	Try word embeddings (https://en.wikipedia.org/wiki/Word_embedding) as feature engineering techniques

	Explore Deep Learning algorithms

	Add more explanations (Requirement, algorithm)
	

# REFERENCES

	1. G. Gautam and D. Yadav. (2014), “Sentiment Analysis of Twitter Data Using Machine Learning Approaches and Semantic Analysis” IEEE 2014.
	2. Neethu M S, and R. R. “Sentiment Analysis in Twitter using Machine Learning Techniques” IEEE 2013.
	3. Mittal and S. A. (2016), “Sentiment Analysis of E-Commerce and Social Networking Sites.” IEEE, pp. 2300-2305.
	4. Paul and R. (2017), “Big Data Analysis of Indian Premier League using Hadoop and MapReduce” IEEE, (pp. 1-6). 
	5. Saragih, M. H. (2017), “Sentiment Analysis of Customer Engagement on Social Media in Transport Online” IEEE, pp. 24-29. 
	6. Shahare, F. F. (2017), “Sentiment Analysis for the News Data Based on the social Media” IEEE, pp. 1365-1370.
	7. B. Gokulkrishnan, P. Priyanthan, T. Ragavan, N. Prasath and A. Perera, “Opinion Mining and Sentiment Analysis on a Twitter Data Stream” IEEE 2012.
	8. http://twitter4j.org/en/
	9. https://en.wikipedia.org/wiki/Vector_quantization
	10. https://en.wikipedia.org/wiki/Apriori_algorithm
	11. Phillip Tichaona Sumbureru. “Analysis of Tweets for Prediction of Indian Stock Markets” IJSR 2013.
	12. Xing Fang and Justin Zhan, “Sentiment analysis using product review data. Journal of Big Data 2015”
	13. Gilad Mishne, “Experiments with Mood Classification in Blog Posts. Live Journal 2005”
	14. S. A Kanade and S. Shibu and Abhishek Chauhan, “Review of Aspect Based Opinion Polling”IJREST 2014.
	15. H. Saif, Y. He, and H. Alani, “Semantic Sentiment Analysis of Twitter,” pp. 508–524, 2012.
