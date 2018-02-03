# encoding: utf-8
import MeCab
import glob
import os
import traceback
from bs4 import BeautifulSoup


class Sentence(object):
    def __init__(self, root):
        self.root = root
        self.surfaces = []
        self.features = []

        if self.root:
            node = root
            while node:
                self.surfaces.append(node.surface)
                self.features.append(node.feature)
                node = node.next

    def all_words(self):
        for surface, feature in zip(self.surfaces, self.features):
            yield surface, feature

    def word_count(self):
        return len(self.surfaces)

    def to_wakati(self):
        return ' '.join([w for w in self.surfaces if w])


class WakatiCorpus(object):
    def __init__(self):
        self.wakati_list = []

    def run(self, add_bos=True):
        tagger = MeCab.Tagger('-Ochasen')
        tagger.parseToNode('') # to prevent GC

        for text in wiki_sentences():
            encoded_text = WakatiCorpus.preprocess(text)
            node = tagger.parseToNode(encoded_text)
            sentence = Sentence(node)
            wakati = sentence.to_wakati()
            if wakati:
                self.wakati_list.append('<bos> ' + wakati + ' <eos>\n')

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.wakati_list = [x.strip() for x in lines]

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(self.wakati_list)

    def loaded(self):
        return len(self.wakati_list) > 0
    
    @staticmethod
    def preprocess(text):
        text = text.lower()
        return text


class DocumentIterator(object):

    WIKI_ROOT_DIR = './extracted'

    def __init__(self):
        self._i = 0
        self.document_path = os.path.join(DocumentIterator.WIKI_ROOT_DIR, '*/*')
        self._files = glob.glob(self.document_path)
        self._docs = []

    def __iter__(self):
        return self

    def __next__(self):
        while self._i < len(self._files):
            if len(self._docs) > 0:
                doc = self._docs.pop()
                return doc

            i = self._i
            self._i += 1
            self._docs = self._get_doc_list(self._files[i])

        raise StopIteration()

    def _get_doc_list(self, filename):
        try:
            with open(filename, encoding='utf-8') as f:
                xml = f.read()
                soup = BeautifulSoup(xml, 'html.parser')
                docs = soup.find_all('doc')
                return [doc.string for doc in reversed(docs) if doc.string]
        except:
            print('Failed to read', filename, traceback.format_exc())
            return []


class SentenceIterator(object):
    def __init__(self, document):
        self._i = 0
        self.sentences = self._break(document)

    def _break(self, sentences):
        sentences = sentences.replace('。', '。\n')
        lines = sentences.split('\n')
        return [line for line in lines if len(line) > 0]

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < len(self.sentences):
            i = self._i
            self._i += 1
            return self.sentences[i]
        else:
            raise StopIteration()

def wiki_sentences():
    diter = DocumentIterator()
    for doc in diter:
        siter = SentenceIterator(doc)
        for text in siter:
            yield text

if __name__ == '__main__':
    wakati = WakatiCorpus()
    wakati.run()
    wakati.save('wakati_corpus.txt')
