import re, os,string
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk.corpus
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from textblob import TextBlob
from scripts import NLP_plots

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

def read_files(folder):
    print("Reading files...")
    listing = os.listdir(folder)
    texts = []
    for file in listing:
        if file.endswith(".txt"):
            url = folder + "/" + file
            f = open(url, encoding="latin-1");
            raw = f.read()
            f.close()
            texts.append(raw)
    return texts

def first_rude_clean_text(texts):
    for txt in range(len(texts)):
        texts[txt] = texts[txt].lower()
        texts[txt] = re.sub('<.*?>', '', texts[txt])
        texts[txt] = re.sub("\n", "", texts[txt])
        texts[txt] = re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", texts[txt])

    return texts

def translate_texts(texts):
    print("Translating...")
    for txt in range(len(texts)):
        if detect(texts[txt]) == 'es':
            if (len(texts[txt]) > 4000):
                chunks, chunk_size = len(texts[txt]) // 4000, 4000
                chunks_txt = [texts[txt][i:i + chunk_size] for i in range(0, chunks, chunk_size)]
                for j in range(len(chunks_txt)):
                    chunks_txt[j] = GoogleTranslator(source='es', target='en').translate(chunks_txt[j])
                texts[txt] = ''.join(chunks_txt)
            else:
                texts[txt] = GoogleTranslator(source='es', target='en').translate(texts[txt])
    return texts

def tokenize_texts(texts):
    for txt in range(len(texts)):
        texts[txt] = word_tokenize(texts[txt])
    return texts

def clean_stop_words_punct(texts):
    stop = stopwords.words('english')
    punct = list(string.punctuation)
    for txt_tok in range(len(texts)):
        texts[txt_tok] = [w for w in texts[txt_tok] if w not in (stop) and w not in (punct)]
    return texts

def first_processing(texts):
    texts = first_rude_clean_text(texts)
    texts = translate_texts(texts)
    texts = tokenize_texts(texts)
    texts = clean_stop_words_punct(texts)
    return texts

def stemmer_text(texts):
    stemmer = SnowballStemmer("english")
    for txt_tok in range(len(texts)):
        texts[txt_tok] = [stemmer.stem(t) for t in texts[txt_tok]]
    return texts

def lemman_text(texts):
    for txt_tok in range(len(texts)):
        texts[txt_tok] = [WordNetLemmatizer().lemmatize(w) for w in texts[txt_tok]]
    return texts

def get_term_matrix(texts):
    total_w = set()
    for txt in texts:
        for w in txt:
            total_w.add(w)
    total_words = list(total_w)
    table = []
    for text in texts:
        row = np.zeros(len(total_words))
        for word in text:
            index = total_words.index(word)
            row[index] += 1
        table.append(row)
    df = pd.DataFrame(table)
    df.columns = total_words
    return df

def create_df(texts):
    joined = []
    for i in texts:
        joined.append(' '.join(i))
    df = pd.DataFrame(joined)
    df.columns = ['text']
    return df

def ner_analysis(df):
    def get_ner(text):
        doc = nlp(text)
        return [X.label_ for X in doc.ents]

    ent = df['text'].apply(lambda x: get_ner(x))
    ent = [x for sub in ent for x in sub]
    NLP_plots.plot_distinct_ner(ent)

def ner_analysis_most_commun(df):
    types=['ORG', 'DATE', 'PERSON', 'GPE']
    def ner(text, ent="ORG"):
        doc = nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]
    gpe_x=list()
    for tipe in types:
        gpe = df['text'].apply(lambda x: ner(x, tipe))
        gpe_x.append([i for x in gpe for i in x])
    NLP_plots.plot_most_commun_word_by_x(gpe_x, types, 'ER: ')

def pos_analysis(df):
    def pos(text):
        pos = nltk.pos_tag(word_tokenize(text))
        pos = list(map(list, zip(*pos)))[1]
        return pos

    tags = df['text'].apply(lambda x: pos(x))
    tags = [x for l in tags for x in l]
    NLP_plots.plot_distinct_pos(tags)

def pos_analysis_most_commun(df):
    types=['NN','JJ','NNS','VBD']

    def get_adjs(text, tipe):
        adj = []
        pos = nltk.pos_tag(word_tokenize(text))
        for word, tag in pos:
            if tag == tipe:
                adj.append(word)
        return adj

    gpe_x=list()
    for tipe in types:
        gpe = df['text'].apply(lambda x: get_adjs(x, tipe))
        gpe_x.append([i for x in gpe for i in x])
    NLP_plots.plot_most_commun_word_by_x(gpe_x, types, 'POS: ')

def sentiment_analysis_Vader(df):

    def get_vader_score(sent):
        sid = SentimentIntensityAnalyzer()
        # Polarity score returns dictionary
        ss = sid.polarity_scores(sent)
        # return ss
        return np.argmax(list(ss.values())[:-1])

    #Vader Score
    copy_df=df.copy()
    copy_df['pol'] = copy_df['text'].map(lambda x: get_vader_score(x))
    polarity = copy_df['pol'].replace({0: 'neg', 1: 'neu', 2: 'pos'})
    NLP_plots.plot_sentiments_bar(polarity,'Sentiment analysis using Vader')

def sentiment_analysis_TextBlob(df):
    def polarity(text):
        return TextBlob(text).sentiment.polarity

    df['pol_scr'] = df['text'].apply(lambda x: polarity(x))

    def sentiment(x):
        if x < 0:
            return 'neg'
        elif x == 0:
            return 'neu'
        else:
            return 'pos'

    df['pol'] = df['pol_scr'].map(lambda x: sentiment(x))
    NLP_plots.plot_sentiments_bar(df.pol,'Sentiment analysis using Vader')