from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def plot_word_count(data):
    from wordcloud import WordCloud
    data = data.T.sum(axis=1).sort_values(ascending=False)
    wc = WordCloud(width=400, height=330, max_words=150,colormap='rainbow_r',background_color='white').generate_from_frequencies(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def most_used_words(data):
    data.T.sum(axis=1).sort_values(ascending=False).head(20).plot(kind='bar', title='Most used words acros news',colormap='rainbow')
    plt.show()

def plot_distinct_ner(ent):
    counter = Counter(ent)
    count = counter.most_common()
    x, y = map(list, zip(*count))
    sns.barplot(x=y, y=x).set_title('Most used types of NER')
    plt.show()

def plot_most_commun_word_by_x(gpe_x,title,x_t):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 7), constrained_layout=False)
    i,j=0,0
    for k,tipe in enumerate(title):
        counter = Counter(gpe_x[k])
        x, y = map(list, zip(*counter.most_common(10)))
        sns.barplot(y, x, ax=ax[j, i])
        ax[j, i].title.set_text(str(x_t) + tipe)
        j = j + 1
        if j == 2:
            i = 1
            j = 0
    plt.show()

def plot_distinct_pos(tags):
    counter = Counter(tags)
    x, y = list(map(list, zip(*counter.most_common(7))))
    sns.barplot(x=y, y=x).set_title('Most commun POS types')
    plt.show()

def plot_sentiments_bar (data,title):
    plt.bar(data.value_counts().index,data.value_counts())
    plt.title(title)
    plt.show()