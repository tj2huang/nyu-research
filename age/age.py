#initialize stopWords
import re
#import regex
import csv
import re
import os
import nltk
import pickle

from sklearn import svm

stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

# start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        #end for loop
        values = list(map.values())
        feature_vector.append(values)
        label=2
        if(tweet_opinion == 'underAge'):
            label = 0
        elif(tweet_opinion == 'legalAge'):
            label = 1
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end


def setup():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Read the tweets one by one and process it
    # inpTweets = csv.reader(open('user_tweets - Copys.csv', 'rb'), delimiter=',')
    inpTweets = csv.reader(open(dir_path + '/small_training.csv', 'r',encoding='utf-8'), delimiter=',')

    # st = open('stop_words.txt', 'r')
    stopWords = getStopWordList(dir_path + '/stop_words.txt')
    featureList = []

    # Get tweet words
    tweets = []
    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        processedTweet = processTweet(tweet)
        featureVector = getFeatureVector(processedTweet)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
    #end loop

    # Remove featureList duplicates
    featureList = list(set(featureList))

    # Extract feature vector for all tweets in one shote
    training_set = nltk.classify.util.apply_features(extract_features, tweets)
    # Train the classifier
    result = getSVMFeatureVectorAndLabels(tweets, featureList)

    clf = svm.SVC(kernel='linear')
    clf.fit(result['feature_vector'], result['labels'])
    pickle.dump(clf, open(dir_path + '/age_svm.p', 'wb'))
    pickle.dump(featureList, open(dir_path + '/feature_list.p', 'wb'))
    # problem = svm_problem(result['labels'], result['feature_vector'])
    # '-q' option suppress console output
    # param = svm_parameter('-q')
    # param.kernel_type = LINEAR
    # classifier = svm_train(problem, param)

    return clf, featureList


def predict(classifier, featureList, in_file, out_loc):
    file_name = in_file.split('/')[-1]
    with open(out_loc + '/' + 'under21_' + file_name[:-4] + '.csv', 'w', encoding='utf-8') as under:
        with open(out_loc + '/' + 'over21_' + file_name[:-4]+ '.csv', 'w', encoding='utf-8') as over:
            out_under = csv.writer(under, delimiter=',')
            out_over = csv.writer(over, delimiter=',')
            with open(in_file, 'r', encoding='utf-8') as out:
                testTweets = csv.reader(out, delimiter=',')
                count = 0
                lineno = 0
                text_pos = 0

                for fields in testTweets:
                    lineno += 1
                    if lineno == 1: # Skip the header line.
                        out_under.writerow(fields)
                        out_over.writerow(fields)
                        text_pos = fields.index('text')
                    #fields = line.split(';')
                    if len(fields)== 0:
                        continue
                    try:
                        tweet = fields[text_pos]

                        tTweets = []
                        processedTweet = processTweet(tweet)
                        featureVector = getFeatureVector(processedTweet)
                        #featureList.extend(featureVector)
                        tTweets.append((featureVector, 0))

                        #Test the classifier
                        test_feature_vector = getSVMFeatureVectorAndLabels(tTweets, featureList)
                        #p_labels contains the final labeling result
                        p_labels = classifier.predict(test_feature_vector['feature_vector'])

                        #print (fields[1],fields[2],fields[3],fields[4],p_labels)
                        if p_labels == [0]:
                            out_under.writerow([fields[0],fields[1],fields[2],fields[3],fields[4],p_labels])
                            count += 1
                        else:
                            out_over.writerow([fields[0], fields[1], fields[2], fields[3], fields[4], p_labels])
                    except Exception as e:
                        pass
