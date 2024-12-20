import gensim.models
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os
from collections import defaultdict, Counter
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
import gensim.corpora as corpora
from pprint import pprint

def extract_data(file):
    '''
    Extracts data from file
    :param file: filepath of file
    :return device_text_dict: dictionary with keys that are the three letter device product code and values that are lists of reports for certain device
    '''
    df = pd.read_json(file)
    device_text_dict = defaultdict(list)
    for i, row in df.iterrows():
        if row['device_report_product_code']:
            device_text_dict[row['device_report_product_code']].append(row['TEXT'])
    return device_text_dict

def preprocess_text(text):
    '''
    Preprocesses text by tokenizing, lowercasing, and removing stopwords. Lemmatizing/stemming is optional but not included in final result
    :param text: text of a single report in a string
    :return flattened_words: list of processed text
    '''
    sentences = sent_tokenize(text)
    sentences = [word_tokenize(sentence) for sentence in sentences]

    sentences = [[word.lower() for word in sentence] for sentence in sentences]

    stop_words = set(stopwords.words('english'))
    sentences = [[re.sub(r'[^\w\s]|\d', '', word) for word in sentence if word not in stop_words] for sentence in sentences]
    sentences = [[word for word in sentence if word] for sentence in sentences]

    '''
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in sentences]
    sentences = [[stemmer.stem(word) for word in sentence] for sentence in sentences]
    '''

    flattened_words = [word for sentence in sentences for word in sentence]
    return flattened_words

def phraser(texts):
    '''
    Performs phrase detection
    :param texts: all reports from a certain device
    :return: all reports from certain device but with some phrases ('natural language' to 'natural_language')
    '''
    phrases = Phrases(texts, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    phraser = Phraser(phrases)
    return [phraser[text] for text in texts]

def make_corpus(processed_text):
    '''
    :param processed_text
    :return id2word, corpus
    '''
    id2word = corpora.Dictionary(processed_text)
    texts = processed_text
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word, corpus

def compute_coherence(lda_model, corpus, dictionary):
    '''
    Computes coherence score
    :param lda_model
    :param corpus
    :param dictionary
    :return coherence score
    '''
    coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

def main():
    batch_texts = defaultdict(defaultdict)
    for file in os.listdir('./data/New MDR Dataset'):
        if file.endswith('.json'):
            device_text_dict = extract_data(os.path.join('./data/New MDR Dataset/', file))
            device_processed_dict = defaultdict()
            for device, texts in device_text_dict.items():
                device_processed_dict[device] = phraser([preprocess_text(text) for text in texts])
            batch_texts[file[:6]] = device_processed_dict
    batch_corpi = defaultdict(defaultdict)
    for batch, devices in batch_texts.items():
        for device, text in devices.items():
            batch_corpi[batch][device] = make_corpus(text)

    lda_model = gensim.models.LdaMulticore(corpus=batch_corpi['batch5']['NCJ'][1],
                                           id2word=batch_corpi['batch5']['NCJ'][0], num_topics=2, alpha=0.1, eta=0.01   )
    pprint(lda_model.print_topics())
    coherence_model_lda = CoherenceModel(model=lda_model, texts=batch_texts['batch5']['NCJ'],
                                         dictionary=batch_corpi['batch5']['NCJ'][0], coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    '''for device, corpi in batch_texts['batch5'].items():
        print("Topics for Device " + device)
        lda_model = gensim.models.LdaMulticore(corpus=batch_corpi['batch5'][device][1], id2word=batch_corpi['batch5'][device][0], num_topics=3, alpha=)
        pprint(lda_model.print_topics())
        coherence_model_lda = CoherenceModel(model=lda_model, texts=batch_texts['batch5'][device], dictionary=batch_corpi['batch5'][device][0], coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        print("\n")'''
    '''
    for batch, devices in batch_texts.items():
        for device, corpi in devices.items():
            print("Topics for Device " + device)
            lda_model = gensim.models.LdaMulticore(corpus=batch_corpi[batch][device][1], id2word=batch_corpi[batch][device][0], num_topics=num_topics)
            pprint(lda_model.print_topics())
            coherence_model_lda = CoherenceModel(model=lda_model, texts=batch_texts[batch][device], dictionary=batch_corpi[batch][device][0], coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)
            print("\n")
            '''

if __name__ == "__main__":
    main()