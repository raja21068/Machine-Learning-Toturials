from sklearn.metrics.pairwise import cosine_similarity
import os

def CorpusFromDir(dir_path):
    result = dict(docs = [open(os.path.join(dir_path,f)).read() for f in
    os.listdir(dir_path)],
    ColNames = map(lambda x: x, os.listdir(dir_path)))
    return result


df= CorpusFromDir('chapter5/Data/text_files')


print ("Similarity b/w doc 1 & 2: ", cosine_similarity(df['Doc_1.txt'],df['Doc_2.txt']))
print ("Similarity b/w doc 1 & 3: ", cosine_similarity(df['Doc_1.txt'],df['Doc_3.txt']))
print ("Similarity b/w doc 2 & 3: ", cosine_similarity(df['Doc_2.txt'],df['Doc_3.txt']))