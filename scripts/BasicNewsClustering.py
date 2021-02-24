import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

# def read_file(file):
#     myfile = open(file,"r")
#     data = ""
#     lines = myfile.readlines()
#     for line in lines:
#         data = data + line
#     myfile.close
#     return data

def cluster_texts(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
    print("Vectors created.")

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                      linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf

if __name__ == "__main__":
    folder = "CorpusNoticiasPractica2021"
    # Empty list to hold text documents.
    texts = []

    listing = os.listdir(folder)
    for file in listing:
        print("File: ",file)
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()
            tokens = nltk.word_tokenize(raw)
            text = nltk.Text(tokens)
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    #distanceFunction = "euclidean"
    test = cluster_texts(texts,5,distanceFunction)
    print("test: ", test)
    # Gold Standard
    # 0 activista Loujain
    # 1 accidente Alonso
    # 2 Muro frontera México
    # 3 Icautación cocaína
    # 4 Rescate cubanos
    # 5 Gobierno de Italia
    # 6 Elecciones Ecuador
    reference =[0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 3, 0, 0, 0, 4, 4, 0, 2, 2]
    print("reference: ", reference)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference,test))

