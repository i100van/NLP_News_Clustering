import warnings
warnings.filterwarnings("ignore")
import NLP_fuctions
import NLP_plots

def news_EDA(term_matrix,df):
    NLP_plots.plot_word_count(term_matrix)
    NLP_plots.most_used_words(term_matrix)
    NLP_fuctions.ner_analysis(df)
    NLP_fuctions.ner_analysis_most_commun(df)
    NLP_fuctions.pos_analysis(df)
    NLP_fuctions.pos_analysis_most_commun(df)
    NLP_fuctions.sentiment_analysis_Vader(df)
    NLP_fuctions.sentiment_analysis_TextBlob(df)

if __name__ == "__main__":
    #Read Data
    folder = "../data/"
    texts = NLP_fuctions.read_files(folder)
    #Clean, Translate, and Tokenize
    texts = NLP_fuctions.first_processing(texts)
    df = NLP_fuctions.create_df(texts)
    term_matrix = NLP_fuctions.get_term_matrix(texts)
    # EDA for deciding what features to use on clustering
    news_EDA(term_matrix,df)
    #Using differents config for clustering
