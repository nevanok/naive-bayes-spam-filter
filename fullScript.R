#############################
# Set up.
# Needed to install and load multiple packages into memory
# e1071 contains the Naive Bayes classification algorithm that we will use.
#install.packages("e1071")
library(e1071)
# Wordcloud is a text visualisation library which will be used to look at most common words
#install.packages("wordcloud")
library(wordcloud)
# gmodels contains a CrossTable function we will use to evaluate results of classification
#install.packages("gmodels")
library(gmodels)
# tm package contains a lot of tools for manipulating text (removing punctuation, removing unimportant words)
#install.packages("tm")
library(tm)

# Read in the csv file for analysis.
# Used stringsAsFactors to ensure the text was read in as text rather than factors
# Used header parameter to ensure first row was read in as data rather than column names
sms = read.csv('C:/Users/Nevn/OneDrive/Documents/Unstructed data/Week 11 - Bayes/SMSSpamCollection.csv', 
               stringsAsFactors = FALSE, col.names = c('type','text'), header = FALSE)

# Calculate percentage of spam and ham.
# Table generates a freq table, prop.table creates a table of proportions and then round to one decimal place.
# It's a good idea to explore the data like this before going into further analysis
round(prop.table(table(sms$type))*100, digits = 1)

# Convert type column from string to factor for classification
sms$type = factor(sms$type)

# Create a corpus object using tm library
# First use vector source which creates a vector of documents, then Corpus.
sms_corpus = Corpus(VectorSource(sms$text))
# Print shows metadata of corpus, inspect will print individual documents.
print(sms_corpus)
inspect(sms_corpus[1:3])


#############################
# Data Cleaning
# Need to clean the text to make it easier to work.
# Cleaning text using tm. tm_map allows us to apply functions to a corpus.
# Convert all letters in corpus to lower case
corpus_clean = tm_map(sms_corpus, tolower)
# Remove all numbers from each document in corpus
corpus_clean = tm_map(corpus_clean, removeNumbers)
# Remove all stop words from corpus, these are words such as "and", "but" etc.
# These words are removed because they are very common and will not improve our classification algorithm
corpus_clean = tm_map(corpus_clean, removeWords, stopwords())
# Remove all punctuation from corpus.
corpus_clean = tm_map(corpus_clean, removePunctuation)
# Remove all unnecessary spaces between words, spaces are reduced to 1 unit.
corpus_clean = tm_map(corpus_clean, stripWhitespace)          
inspect(corpus_clean[1:3])


#############################
# Following code was included in online tutorial but I removed it becuase it generates an error and is not
# necessary.
# Seems to have been converting the corpus to a plain text document.
#corpus_clean = tm_map(corpus_clean, PlainTextDocument) # this is a tm API necessity


#############################
# Create document term matrix for analysis.
# A document-term matrix is a matrix that describes the frequency of words in each document in a corpus.
# There are 5574 rows, each representing a document.
# There are 7943 columsn, each representing a different word.
dtm = DocumentTermMatrix(corpus_clean) 
str(dtm) 


#############################
# Split into training and test sets. The model will be trained on the training set and its performance will be 
# evaluated on the test set.
# Training set will be 75% of the data, and test set 25%
# Split the raw data first
sms.train = sms[1:4200, ] # about 75%
sms.test  = sms[4201:5574, ] # the rest

# Then split the document-term matrix
dtm.train = dtm[1:4200, ]
dtm.test  = dtm[4201:5574, ]

# Lastly, split the corpus
corpus.train = corpus_clean[1:4200]
corpus.test  = corpus_clean[4201:5574]

# Want to check the proportions of spam/ham again to ensure the training & test sets have similar proportions
# to one another.
round(prop.table(table(sms.train$type))*100)
round(prop.table(table(sms.test$type))*100)

#############################
# Wordcloud to visualise frequency of terms.
# Using wordcloud to look at most common words. We are hoping that the most common words in ham are different
# to the most common words in spam so that our classifier will perform better.
# First we will look at the most common words in general by creating a wordcloud of the corpus.
wordcloud(corpus.train,
          min.freq=40,          # Parameter to set the minimum frequency of words to be included
          random.order = FALSE) # Parameter to ensure most common words will be in the centre.

# Now split raw training data into spam and ham, then visualise the most common words for each.
spam = subset(sms.train, type == "spam")
ham  = subset(sms.train, type == "ham")
wordcloud(spam$text,
          max.words=40,     # Parameter for number of words to show
          scale=c(3, 0, 5)) # Parameter to set max/min font sizes for words

wordcloud(ham$text,
          max.words=40,     
          scale=c(3, 0, 5))


#############################
# Feature engineering
# There are almost 8000 columns in the document-term matrix, we would like to reduce this to a smaller number.
# We will use finFreqTerms to find list of words that appear more than 5 times.
# This list will then be used to subset the document-term matrix so that infrequent words are removed.
freq_terms = findFreqTerms(dtm.train, 5) 
reduced_dtm.train = DocumentTermMatrix(corpus.train, list(dictionary=freq_terms))
reduced_dtm.test =  DocumentTermMatrix(corpus.test, list(dictionary=freq_terms))

# We have reduced the number of features from 7943 to 1231  
ncol(reduced_dtm.train)
ncol(reduced_dtm.test)

# Document-term matrix contains counts of words' appearance in each document, we need to convert these counts
# to factors because Naive Bayes only works on factors.
# Defined function to convert to factors below 
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0) # Returns 1 if x>0 and 0 otherwise
  x = factor(x, levels = c(0, 1), labels=c("No", "Yes")) # Convert numerical values (0,1) to factors and label them
  return (x)
}

# Apply function to each column in document-term matrix, converting counts to yes/no factors
reduced_dtm.train = apply(reduced_dtm.train, MARGIN=2, convert_counts)
reduced_dtm.test  = apply(reduced_dtm.test, MARGIN=2, convert_counts)

#############################
# Training and testing our model
# First need to train and then predict, separate functions
# Use e1071 library's naiveBayes function to create our model
sms_classifier = naiveBayes(reduced_dtm.train, sms.train$type) # Training the model on the training document-term matrix
sms_test.predicted = predict(sms_classifier, reduced_dtm.test) # Use the model to classify the test data

# Now using gmodels library to evaluate the results.
# CrossTable will compare predicted values to actual values.
# CrossTable will display true & false positives, and true & false negatives.
# ie. Messages that are classified correctly as ham, incorrectly as ham, correctly as spam and incorrectly as spam
CrossTable(sms_test.predicted,
             sms.test$type,
             prop.chisq = FALSE, # Parameter to specify inclusion of chi square contributions
             prop.t     = FALSE,  # Parameter to specify inclusion of table proportions
             dnn        = c("predicted", "actual")) # Parameter to set labels for cross table dimensions

#############################
# Laplace Smoothing
# In Naive Bayes, the probability of a message being spam is the product of the probabilities of each word 
# indicating that the message is spam. If a certain word is in the test set but not in the training set, the 
# probability of that word indicating spam will be 0, and hence the probability of that message being spam is 0,
# which is erroneous.
# Laplace smoothing is a solution to this as it adds 1 to the count for each word so that a probability can 
# be calculated.
# We will use the same train-predict process as before:
sms_classifier2 = naiveBayes(reduced_dtm.train,
                             sms.train$type,
                             laplace = 1)
sms_test.predicted2 = predict(sms_classifier2,
                              reduced_dtm.test)

# Again, we will evaluate the results using the CrossTable function from gmodels.
CrossTable(sms_test.predicted2,
           sms.test$type,
           prop.chisq = FALSE, 
           prop.t     = FALSE, 
           dnn        = c("predicted", "actual")) 
