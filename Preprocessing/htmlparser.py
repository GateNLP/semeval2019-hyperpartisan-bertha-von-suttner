"""
How to parse the (X)HTML we get into a sequence of paragraphs and gather
a bit of additional data for this.
"""

from html.parser import HTMLParser
import preprocessing

class MyHTMLParser(HTMLParser):

    def __init__(self):
        kwargs = {}
        HTMLParser.__init__(self, **kwargs)
        self.ignore = False
        self.data = []
        self.p = []

    def finishp(self):
        if len(self.p) > 0:
            self.data.append(self.p)
            self.p = []

    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        if tag in ['script', 'style']:
            self.ignore = True
        elif tag in ['p', 'br']:
            self.finishp()
        # any tags that need to get repalced by space?
        # elif tag in ['???']:
        #     self.p.append(" ")

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        if tag in ['script', 'style']:
            self.ignore = False
        elif tag in ['p', 'br']:
            self.finishp()
        # any tags that need to get repalced by space?
        # elif tag in ['???']:
        #     self.p.append(" ")

    def handle_startendtag(self, tag, attrs):
        # print("Encountered a startend tag:", tag)
        if tag in ['p', 'br']:
            self.finishp()

    def handle_data(self, data):
        # print("Encountered some data  :", data)
        if not self.ignore:
            self.p.append(data)

    def close(self):
        HTMLParser.close(self)
        self.finishp()

    def reset(self):
        HTMLParser.reset(self)
        self.data = []
        self.p = []

    def cleanparagraph(self, text):
        """
        How to do basic cleaning up of the text in each paragraph
        :return:
        """
        text = preprocessing.cleantext(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = ' '.join(text.split()).strip()
        return text

    def paragraphs(self):
        """
        Convert collected data to paragraphs
        """
        pars = []
        for par in self.data:
            if len(par) > 0:
                text = self.cleanparagraph(''.join(par)).strip()
                if text:
                    pars.append(text)
        return pars


if __name__ == '__main__':

    xml = """
    <article><script>SCRIPT</script><style>STYLE</style>
    some <em>text</em><p>more <br> text 1</p>text2 with a
     <a href="asas">link</a><p/>text3 ampersand: &amp; and less &lt; <br/>text4<p>some dangling
      text, no closing article
    """
    parser = MyHTMLParser()
    parser.reset()
    parser.feed(xml)
    parser.close()
    print("Paragraphs:", parser.paragraphs())

