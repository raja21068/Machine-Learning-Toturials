from nltk import word_tokenize
import matplotlib as plt


def dispersion_plot(text, words):
    words_token = word_tokenize(text)
    points = [(x,y) for x in range(len(words_token)) for y in
    range(len(words)) if words_token[x] == words[y]]

    if points:
        x,y=zip(*points)
    else:
        x=y=()

    plt.plot(x,y,"rx",scalex=.1)
    plt.yticks(range(len(words)),words,color="b")
    plt.ylim(-1,len(words))
    plt.title("Lexical Dispersion Plot")
    plt.xlabel("Word Offset")
    plt.show()

text = 'statistics skills, and programming skills are equally important for analytics. statistics skills, and domain knowledge are important for analytics'


dispersion_plot(text, ['statistics', 'skills', 'and', 'important'])