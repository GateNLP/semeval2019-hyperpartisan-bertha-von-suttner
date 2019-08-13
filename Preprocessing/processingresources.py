"""
Classes representing "processing resources"
"""

import xml.etree.ElementTree

import json
from numbers import Number
import sys

from . import features
from . import htmlparser
from . import nlp
from . import preprocessing


class PrArticle2Line:

    def __init__(self, stream, featureslist, addtargets=True):
        self.stream = stream
        self.features = features.features2use(featureslist)
        self.mp_able = False
        self.addtargets = addtargets
        self.need_et = False

    def __call__(self, article, **kwargs):
        values = features.doc2features(article, self.features)
        strings = []
        for i in range(len(values)):
            val = values[i]
            if isinstance(val, str):
                strings.append(val)
            elif isinstance(val, Number):
                strings.append(str(val))
            elif isinstance(val, list):
                strings.append(json.dumps(val))
            elif isinstance(val, dict):
                strings.append(json.dumps(val))
            else:
                # raise Exception("Not a known type to convert to string: {} for {}, feature {}, article id {}".
                # format(type(val), val, self.features[i], article['id']))
                print("Not a known type to convert to string: {} for {}, feature {}, article id {}".
                      format(type(val), val, self.features[i], article['id']))
        if self.addtargets:
            print(article['id'], article.get('target'),
                  article.get('bias'), article.get('domain'),
                  "\t".join(strings), file=self.stream, sep="\t")
        else:
            print("\t".join(strings), file=self.stream)


class PrAddTarget:

    def __init__(self, a2target, a2bias, a2url):
        self.a2target = a2target
        self.a2bias = a2bias
        self.a2url = a2url
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        id = article['id']
        target = self.a2target[id]
        bias = self.a2bias[id]
        url = self.a2url[id]
        article['target'] = target
        article['bias'] = bias
        article['url'] = url


class PrAddTitle:

    def __init__(self):
        self.mp_able = True
        self.need_et = True

    def __call__(self, article, **kwargs):
        element = article['et']
        attrs = element.attrib
        title = preprocessing.cleantext(attrs["title"])
        article['title'] = title


class PrAddText:

    def __init__(self):
        self.mp_able = True
        self.need_et = False
        self.parser = None  # initialize later, do not want to pickle for the pipeline

    def __call__(self, article, **kwargs):
        if self.parser is None:
            self.parser = htmlparser.MyHTMLParser()
        self.parser.reset()
        self.parser.feed(article['xml'])
        self.parser.close()
        pars = self.parser.paragraphs()
        article['pars'] = pars
        text = " ".join(pars)
        article['text'] = text


class PrRemovePars:

    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        del article['pars']


class PrFilteredText:
    """
    Calculate the single filtered text field text_all_filtered, must already have nlp
    """

    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        text_tokens = article['text_tokens']
        title_tokens = article['title_tokens']
        tokens = nlp.filter_tokens([t[0] for t in title_tokens])
        tokens.append("<sep_t2d>")
        if article.get('link_domains_all'):
            tokens.extend(["DOMAIN_" + d for d in article['link_domains']])
        tokens.append("<sep_d2a>")
        tokens.extend(nlp.filter_tokens([t[0] for sent in text_tokens for t in sent]))
        token_string = " ".join(tokens)
        article['text_all_filtered'] = token_string


class PrNlpSpacy01:
    """
    Tokenise and POS-tag the title and article.
    The title gets converted into a list of list word, POS, lemma.
    The article gets converted into a list of
    sentences containing a list of lists word, POS, lemma for the sentence.
    :return:
    """

    def __init__(self):
        import spacy
        self.mp_able = True
        self.initialized = False
        self.need_et = False
        self.nlp = None

    def initialize(self):
        if self.initialized:
            return
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.initialized = True

    def __call__(self, article, **kwargs):
        # process each paragraph separately to avoid getting sentences
        # crossing paragraphs
        if not self.initialized:
            self.initialize()
        pars = article['pars']
        # store the raw number of paragraphs
        article['n_p'] = len(pars)
        # print("DEBUG: number of pars", len(pars))
        n_p_filled = 0
        # print("\n\nDEBUG: {} the texts we get from the paragraphs: ".format(article['id']), pars)
        docs = list(self.nlp.pipe(pars))
        allthree = [[[t.text, t.pos_, t.lemma_] for t in s] for doc in docs for s in doc.sents]
        article['n_p_filled'] = n_p_filled
        article['text_tokens'] = allthree
        ents = [ent.text for doc in docs for ent in doc.ents if ent.text[0].isupper()]
        article['text_ents'] = ents
        title = article['title']
        doc = self.nlp(title)
        allthree = [(t.text, t.pos_, t.lemma_) for s in list(doc.sents) for t in s]
        article['title_tokens'] = allthree
        ents = [ent.text for ent in doc.ents if ent.text[0].isupper()]
        article['title_ents'] = ents


class PrSeqSentences:
    """
    Creates fields: title_sent, domain_sent, article_sent, title and article generated from the
    token lists for the title and article text (using the original token string)
    The sentences for the article are enclosed in the special <bos> and <eos> markers.
    """
    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        article_tokens = article['text_tokens']
        title_tokens = article['title_tokens']
        title_sent = " ".join([t[0] for t in title_tokens])
        domain_sent = ""
        if article.get('link_domains_all'):
            domain_sent = " ".join(["DOMAIN_" + d for d in article['link_domains']])
        all = []
        first = True
        for sent in article_tokens:
            if first:
                first = False
            else:
                all.append("<splt>")
            # all.append("<bos>")
            for t in sent:
                all.append(t[0])
            # all.append("<eos>")
        article_sent = " ".join(all)
        article['article_sent'] = article_sent
        article['domain_sent'] = domain_sent
        article['title_sent'] = title_sent
