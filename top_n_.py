import os
import pickle
import numpy as np
import sys


model_path = './models/'
#loss_model = 'cross_entropy'
#loss_model = 'nce'

loss_model = 'cross_entropy'
if len(sys.argv) > 1:
    if sys.argv[1] == 'nce':
      loss_model = 'nce'
model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

def get_top_n_words(context_word, n):
    cosine_similarities = {}
    context_vector = embeddings[dictionary[context_word]]
    for word in dictionary:
        if word != context_word:
            word_vector = embeddings[dictionary[word]]
            cosine_similarities[word] = get_cosine_similarity(context_vector, word_vector)

    sorted_similarities = sorted(cosine_similarities.items(), key=lambda pair: pair[1], reverse=True)
    return sorted_similarities[:n]


def get_cosine_similarity(v1,v2):
    dot_product = sum([x*y for x,y in zip(v1,v2)])
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    return dot_product/(mod_v1 * mod_v2)


a = get_top_n_words("first",20)
b = get_top_n_words("american", 20)
c = get_top_n_words("would", 20)

with open('top_n.txt', 'w') as output:
    output.write("%s\n\n\n" % a)
    output.write("%s\n\n\n" % b)
    output.write("%s\n\n\n" % c)
