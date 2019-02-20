'''Functions for preprocessing the original article data'''

import xml.etree.ElementTree

import re

txt_tospace1 = re.compile('&#160;')


def cleantext(text):
    '''Clean the text extracted from XML.'''
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("<p>", " ")
    text = text.replace("</p>", " ")
    text = text.replace(" _", " ")
    text = text.replace("–", "-")
    text = text.replace("”", "\"")
    text = text.replace("“", "\"")
    text = text.replace("’", "'")

    text, _ = txt_tospace1.subn(' ', text)
    return text

