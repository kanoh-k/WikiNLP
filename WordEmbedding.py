# encoding: utf-8

import os

from gensim.models import word2vec
from WikiIterator import WakatiCorpus

class WordEmbedding(object):

    WAKATI_CORPUS_FILE = 'wakati_corpus.txt'
    MODEL_FILE = 'wiki_w2v.model'

    def __init__(self):
        self.model = None
        self.w2i = {} # word --> id mapping
        self.i2w = {} # id --> word mapping

        # Temporal variable
        self.vocabulary = None

    def __contains__(self, word):
        return word in self.vocabulary

    def load(self, filename=MODEL_FILE):
        """ Load word2vec model"""

        if not os.path.exists(filename):
            raise Exception('Model not found: {}'.format(filename))

        self.model = word2vec.Word2Vec.load(filename)

        dict_filename = '{}.dict'.format(filename)
        self.__load_word_ids(dict_filename)

    def prepare_corpus(self, filename=WAKATI_CORPUS_FILE):
        if not os.path.exists(filename):
            wakati = WakatiCorpus()
            wakati.run()
            wakati.save(filename)

    def learn(self, filename=WAKATI_CORPUS_FILE):
        self.prepare_corpus()

        corpus = word2vec.LineSentence(filename)
        self.model = word2vec.Word2Vec(corpus, size=200, min_count=5, workers=6)
        self.__compute_word_ids()

    def save(self, filename=MODEL_FILE):
        self.model.save(filename)

        dict_filename = '{}.dict'.format(filename)
        self.__save_word_ids(filename=dict_filename)

    def get_model(self):
        return self.model

    def get_vocabulary_size(self):
        return len(self.model.wv.vocab)

    def get_most_similar(self, positive=[], negative=[]):
        return self.model.wv.most_similar(positive=positive, negative=negative)

    def save_embedding_projector_files(self, vector_file, metadata_file):
        """ Generate a vector file and a metadata file for Embedding Projector.

        You can upload the generated files to Embedding Projector
        (http://projector.tensorflow.org/), and get vizualization of
        the trained vector space.
        """
        with open(vector_file, 'w', encoding='utf-8') as f, \
             open(metadata_file, 'w', encoding='utf-8') as g:

            # metadata file needs header
            g.write('Word\n')

            for word in self.model.wv.vocab.keys():
                embedding = self.model.wv[word]

                # Save vector TSV file
                f.write('\t'.join([('%f' % x) for x in embedding]) + '\n')

                # Save metadata TSV file
                g.write(word + '\n')

    def __compute_word_ids(self):
        self.vocabulary = sorted(list(self.model.wv.vocab.keys()) +
                                 ['<bos>', '<eos>', '<unk>'])
        self.w2i = {w:i for i,w in enumerate(self.vocabulary)}
        self.i2w = {i:w for i,w in enumerate(self.vocabulary)}

    def __save_word_ids(self, filename):
        if not self.vocabulary:
            self.__compute_word_ids()

        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines([w + '\n' for w in self.vocabulary])

    def __load_word_ids(self, filename):
        if not os.path.exists(filename):
            print('{} not found. Compute word ids from the model instead.'.format(filename))
            self.__compute_word_ids()
            return

        with open(filename, 'r', encoding='utf-8') as f:
            vocabulary = [w.strip() for w in f.readlines()]
            self.w2i = {w:i for i,w in enumerate(vocabulary)}
            self.i2w = {i:w for i,w in enumerate(vocabulary)}

def learn():
    embedding = WordEmbedding()
    embedding.learn()
    embedding.save()

def embedding_projector():
    embedding = WordEmbedding()
    embedding.load()

    print('vocabulary size:', embedding.get_vocabulary_size())
    embedding.save_embedding_projector_files('vector.tsv', 'metadata.tsv')

if __name__ == '__main__':
    # Train the model and save the results to the file
    learn()

    # Load the model from the file, and create two TSV files for Embedding Projector
    embedding_projector()
