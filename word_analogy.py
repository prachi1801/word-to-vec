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

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word]]



==========================================================================
"""

input_filename = 'word_analogy_test.txt'
if loss_model == 'cross_entropy':
    output_filename = 'word_analogy_test_predictions_cross_entropy.txt'
else:
    output_filename = 'word_analogy_test_predictions_nce.txt'



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


'''a = get_top_n_words("first",20)
b = get_top_n_words("american", 20)
c = get_top_n_words("would", 20)

with open('top_n.txt', 'w') as output:
    output.write("%s\n\n\n" % a)
    output.write("%s\n\n\n" % b)
    output.write("%s\n\n\n" % c)'''

with open(input_filename) as f:
    content = f.readlines()
content = [x.strip() for x in content]

predictions = []
for line in content:

    examples_choices = line.split('||')
    examples_list = examples_choices[0]
    choices_list = examples_choices[1]

    examples = examples_list.split(',')
    choices = choices_list.split(',')

    total_relation = np.zeros(embeddings.shape[1])
    for example in examples :
        stripped_example = example.split(':')
        word_1 = stripped_example[0].replace('"', '')
        word_2 = stripped_example[1].replace('"', '')

        v1 = embeddings[dictionary[word_1]]
        v2 = embeddings[dictionary[word_2]]

        relation = v1 - v2
        total_relation = total_relation + relation

    average_relation = total_relation/len(examples)

    similarities = []
    for choice in choices :
        stripped_choice = choice.split(':')
        word_1 = stripped_choice[0].replace('"', '')
        word_2 = stripped_choice[1].replace('"', '')

        v1 = embeddings[dictionary[word_1]]
        v2 = embeddings[dictionary[word_2]]
        relation = v1 - v2
        similarity = get_cosine_similarity(average_relation,relation)
        similarities.append(similarity)

    max_val = max(similarities)
    index_max = similarities.index(max_val)

    min_val = min(similarities)
    index_min = similarities.index(min_val)

    choices.append(choices[index_min])
    choices.append(choices[index_max])

    predictions.append(choices)

with open(output_filename, 'w') as output:
    for prediction in predictions:
        output.write("%s\n" % (" ".join(prediction)))



