#  comparable-dictionary

Create comparable dictionary for low-resource languages with distributional semantic method using seed topic words

create-dictionary: Aligns documents only by the seed topics

create-dictionary-improved: Aligns documents by actual Wikipedia article alignments. Introduces additional weighting scheme for similarity, to take into account how often words occur in the same document. Currently inefficient. Can be improved by using vectors in term-document representation.

**Example run:

python  create-dictionary.py --topic topic-lists/all.lang --inputdir corpus/ --outputdir dictionary2/ --lang en,ar --counts dictionary2/counts/ --words dictionary2/wordcounts/

python create-dictionary-improved.py --topic topic-lists/all.lang.newest --inputdir ../comparable-data/wikipedia-aligned/en/ --outputdir dictionaryAligned/ --lang ar,en

--topic-lists/all.lang: contain the seed words (dimensions) which correspond to initial translations in the different languages

--inputdir: directory containing concatenated documents for each language (in the case of create-dictionary.py) or aligned articles (in the case of create-dictionary-improved.py)

--outputdir: dictionary directory where outputs will be created

--lang: list of desired languages, comma separated




