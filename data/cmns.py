"""
the simplest entity linking by exact match names
input:
    an entity vocabulary
    a text (tokenized)
output:
    the entities and the positions they appear in the text
"""

import logging
from traitlets import (
    Unicode,
    Dict,
    Int,
)
from traitlets.config import Configurable
from nltk.stem import WordNetLemmatizer
from knowledge4ir.utils.kg import read_surface_form
from knowledge4ir.utils.nlp import s_stopwords


class CommonEntityLinker(Configurable):
    entity_vocab_in = Unicode("", help="the entity vocabulary input").tag(config=True)
    kg_in = Unicode("", help="knowledge graph triples input, pick one from entity_vocab_in and this"
                    ).tag(config=True)

    h_surface_name = Dict()
    max_surface_len = Int(4, help='maximum surface form len')

    def __init__(self, **kwargs):
        super(CommonEntityLinker, self).__init__(**kwargs)
        self._load_vocabulary()
        self._load_surface_form_from_kg()

    def _load_vocabulary(self):
        if not self.entity_vocab_in:
            return
        l_lines = open(self.entity_vocab_in).read().splitlines()
        l_name_id_score = [line.split('\t') for line in l_lines]
        h_name_score = {}
        for name, mid, score in l_name_id_score:
            score = float(score)
            if score < 10:
                continue
            # if name != name.upper():
            #     name = name.lower()
            if name in s_stopwords:
                continue
            if name not in h_name_score:
                h_name_score[name] = (mid, score)
            else:
                if score > h_name_score[name][1]:
                    h_name_score[name] = (mid, score)
        self.h_surface_name = dict([(item[0], item[1][0]) for item in h_name_score.items()])
        del h_name_score
        logging.info('loaded a [%d] entity surface name vocabulary', len(self.h_surface_name))
        return

    def _load_surface_form_from_kg(self):
        """
        load surface form from kg
        all names will be lower cased
        :return:
        """
        if not self.kg_in:
            return
        self.h_surface_name = read_surface_form(self.kg_in)
        return

    def link(self, text, stemming=False):
        """
        link the text
        text shall be tokenized
        do exact match, ignoring cases
        :param text: text to annotate
        :param stemming: whether do stemming or not (if True, will try both raw and stemmed)
        :return:
        """
        l_term = text.split()
        offset = 0
        l_annotation = []
        st = 0
        while st < len(l_term):
            matched = False
            for ed in xrange(self.max_surface_len):
                phrase = ' '.join(l_term[st: st + self.max_surface_len - ed])
                l_candidate_phrase = [phrase, phrase.title(), phrase.upper()]
                for this_phrase in l_candidate_phrase:
                    if this_phrase in self.h_surface_name:
                        l_annotation.append([self.h_surface_name[this_phrase],
                                             offset, offset + len(this_phrase),
                                             this_phrase])
                        st += self.max_surface_len - ed
                        offset += len(this_phrase) + 1
                        matched = True
                        break
                    if stemming:
                        stemmed_phrase = self._phrase_stem(this_phrase)
                        if stemmed_phrase in self.h_surface_name:
                            l_annotation.append([self.h_surface_name[stemmed_phrase],
                                                 offset, offset + len(this_phrase),
                                                 stemmed_phrase])
                            st += self.max_surface_len - ed
                            offset += len(this_phrase) + 1
                            matched = True
                            break
                if matched:
                    break

            if not matched:
                offset += len(l_term[st]) + 1
                st += 1
        return l_annotation

    @classmethod
    def _phrase_stem(cls, phrase):
        wnl = WordNetLemmatizer()
        l_term = phrase.split()
        l_term = [wnl.lemmatize(term, 'n') for term in l_term]
        return ' '.join(l_term)

