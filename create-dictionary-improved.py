# create multilingual dictionary from comparable corpora
import sys
import os
import codecs
import argparse
import sklearn
from collections import defaultdict
import operator
import gensim
import pickle
import json
import math
import string
import nltk
reload(sys)
sys.setdefaultencoding('utf8')


#corpus_files, aligned_docs, aligned_docs_freq = collect_aligned_docs(corpus_dir,langs)
# Read corpus, collect corpus files and aligned docs
def collect_aligned_docs(corpus_dir,langs):
	corpus_files = {}
	aligned_docs = {}
	alignment_file = os.path.join(corpus_dir,'aligned-all.list')
	a = codecs.open(alignment_file,'rb')
	for l in langs:
	    corpus_files[l] = {}
	for alignment in a:
		l1, id1, title1, l2, id2, title2, topic = alignment.strip().split('\t')
		if l1 in langs:
			if not id1 in corpus_files[l1]:
				corpus_files[l1][id1] = {}
				corpus_files[l1][id1]['path'] = os.path.join(corpus_dir,topic,'aligned',id1,l1+'_'+id1+'.txt')
				corpus_files[l1][id1]['topics'] = []
				corpus_files[l1][id1]['title'] = title1
			corpus_files[l1][id1]['topics'].append(topic)
		if l2 in langs and l2 != l1:
			if not id2 in corpus_files[l2]:
                                corpus_files[l2][id2] = {}
                                corpus_files[l2][id2]['path'] = os.path.join(corpus_dir,topic,'aligned',id1,l2+'_'+id2+'.txt')
                                corpus_files[l2][id2]['topics'] = []
                                corpus_files[l2][id2]['title'] = title2
			corpus_files[l2][id2]['topics'].append(topic)
			
		if l1 in langs and l2 in langs and l2 != l1 :
			if not (l1,l2) in aligned_docs:
				aligned_docs[l1,l2] = {}
			if not (id1,id2) in aligned_docs[l1,l2]:
				aligned_docs[l1,l2][id1,id2] = 1
                        # may or may not need
			if not (l2,l1) in aligned_docs:
				aligned_docs[l2,l1] = {}
			if not (id2,id1) in aligned_docs[l2,l1]:
				aligned_docs[l2,l1][id2,id1] = 1				

	return corpus_files,aligned_docs	


# Collect context counts for dimensions
def compute_context_counts(context_counts,word_counts,corpus_file,window,lang,dimensions):
#	context_counts = {}
#	word_counts = {}
	word_counts_sorted = {}
	this_doc_words = {}
#	tfidf = {}
	co =  codecs.open(corpus_file,'r','utf8')
#	g = 5
	processed = 0
	times = 0
	for line in co:
		processed += 1
		if processed > 100000:
			processed = 0
			times += 1
			print (processed + 100000*times)
	        line = line.strip()
	#	line = line.translate(None,string.punctuation).strip()
		words = nltk.word_tokenize(line)
#		words = line.split(' ')
		for i in range(0,len(words)):
			w = words[i].strip()
			if w in string.punctuation or "==" in w:
				continue
			if not w in word_counts:
				word_counts[w] = 1
			else:
				word_counts[w] += 1
			if not w in this_doc_words:
				this_doc_words[w] = 1
			else:
				this_doc_words[w] += 1
                        if w in dimensions[lang].values():
                                for k in range(i - window, i + window + 1):
                                        if k < 0 or k > len(words)-1 or k==i:
                                                continue
                                        v = words[k].strip()
                                        if not v in context_counts:
                                                context_counts[v] = {}
                                        if not w in context_counts[v]:
                                                context_counts[v][w]=1
                                        else:
                                                context_counts[v][w]+=1
#	print "Sorting word counts"
#	word_counts_sorted = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        
#	max_freq = word_counts_sorted[0][1]
#	print "max freq:" , max_freq
#	for word, freq in word_counts_sorted:
#		tfidf[word] = ( math.log(float(max_freq)/freq) + 1 ) * float(freq)
#	tfidf_sorted = sorted(tfidf.items(),key=operator.itemgetter(1),reverse=True)
	return context_counts, word_counts, this_doc_words 
					 
# Positive PMI
def PPMI(pij, pi, pj):
	if pij != 0 and pi != 0 and pj != 0 and pi != 0.0 and pj != 0.0 :
		pmi = math.log(float(pij)/(float(pi) * float(pj)))
	else:
		return 0.0
	if pmi > 0:
		return pmi
	else:
		return 0.0


# Given a language corpus, create a vector for each word
# consisting of PMI values between the word and the dimension  
# TODO This is still not sorted
def build_PMI_matrix(context_counts,word_counts,lang,invert_dim,k):
	pmi_matrix = {}
	total_context = 0
	pi_ = {}
	p_j = {}
	fi_ = {}
	f_j = {}
	threshold = 5
#	threshold = 100
	num_total = 0
	num_rare = 0
	num_not_in_context = 0
	rem = 0
	# Collect counts   # can go in freq word order so for w in word_counts
	for w, freq in word_counts:
	    num_total +=1 
	    if not w in context_counts or freq < threshold:
		if not w in context_counts:
			num_not_in_context +=1
		if freq < threshold:
			num_rare +=1
		continue
	    rem += 1
	    fi_[w] = 0.0
	    for index, c in invert_dim[lang].items():
		if c in context_counts[w]:
			count = context_counts[w][c]
		else:
			count = 0.0
		count += k
	        total_context += count
		fi_[w] += count 
                if c in f_j:
		   f_j[c] += count
		else:
		   f_j[c] = 0.0

	print "-----------------------"
	print "Language:", lang
	print "Total vocab:", num_total
	print "Included in vec:", rem
	print "Rare:", num_rare
	print "No overlap with context:", num_not_in_context
	print "Frequent:", num_total - num_rare
	print "Overlap:", num_total - num_not_in_context
	print "Percentage in context to frequent", float(num_total - num_not_in_context)/(num_total - num_rare)
	print "-----------------------"

	# Update p_j
	for index in invert_dim[lang]:
	 	print index, invert_dim[lang][index]
		if invert_dim[lang][index] not in f_j:
			sys.exit("Context word " + invert_dim[lang][index] + " is not in context count dictionary")
		context_word = invert_dim[lang][index]
		p_j[context_word] = float(f_j[context_word])/total_context
	
	# Create word vector matrix   	        	
	for w,freq in word_counts:
		if not w in context_counts or freq < threshold:
			continue
		pmi_matrix[w] = {}
		pi_[w] = float(fi_[w])/total_context
		for index in invert_dim[lang]:
			c = invert_dim[lang][index]
			if not c in context_counts[w]:
				count = 0.0
			else:	
				count = context_counts[w][c]
			count += k
			p_ij = float(count)/total_context
			#pmi_matrix[w][c] = PPMI(p_ij, pi_[w], p_j[c])   	
			pmi_matrix[w][index] = PPMI(p_ij, pi_[w], p_j[c])
	
	return pmi_matrix


def output_vectors(vector_matrix,word_counts,output_dir,lang,invert_dim,header=0):
	vector_dir = os.path.join(output_dir,'vectors')
	if not os.path.exists(vector_dir):
		os.mkdir(vector_dir)
	vector_file = os.path.join(vector_dir,lang)
	v = codecs.open(vector_file,'w','utf-8')
        voc = len(vector_matrix)
	wo = vector_matrix.keys()[0]
	dim = len(vector_matrix[wo])
	if header:
		voc -=1 
	v.write(str(voc) + ' ' + str(dim) + '\n')
	if header:
		for index in vector_matrix[wo]:
			c = invert_dim[lang][index]
			v.write(c + ' ')
		v.write('\n')
	for w, freq in word_counts:
		if not w in vector_matrix:
			continue
		row = w + " "
		for index in vector_matrix[w]:
			pmi = vector_matrix[w][index]
			row += str(pmi) + " "
		v.write(row + '\n')
			

# Compute cosine similarity between 2 vectors
# If traditional method is too slow I can use the efficient method in the vector semantics paper
def sim(x,y):
	dot_prod = 0.0
	sum_x_sqr = 0.0
	sum_y_sqr = 0.0
	for i in x:
		if x[i] != 0.0:
			sum_x_sqr += math.pow(x[i],2)
			dot_prod += float(x[i]) * y[i]
		if y[i] != 0.0:
			sum_y_sqr += math.pow(y[i],2)
	if sum_x_sqr != 0.0 and sum_y_sqr != 0.0:
		return float(dot_prod)/math.sqrt(float(sum_x_sqr) * sum_y_sqr)
	else:
		return 0.0	

# Given a word vector in l1 and word vectors in l2, find the most similar word in l2 and return
# along with the cosine similarity
def most_similar_thresh(l1_vector,l2_matrix,word_counts_l2,thresh):
	max = 0.0
	w_max = ""
	#print "w_max", w_max
	for w2, freq in word_counts_l2:
		if not w2 in l2_matrix:	
			continue
		if freq < thresh: # can also just filter non-freq from the PMI matrix, or take top k in word_counts					
			break
		cos_sim = sim(l1_vector,l2_matrix[w2])		
		if cos_sim > max:
			max = cos_sim
			w_max = w2
	if max == 0.0:
		print "weird, all vectors have similarity 0"
	return (w_max,max)


# "probability" (W(we,wf)) that  w1 and w2 co-occur in the same document. can turn into probability by using frequency instead
def co_occur(w1,w2,l1,l2,articles_words,aligned_docs):
	co_occur = 0
	# create an actual vector and multiply (numpy). If product is 0 it means they don't co-occur. Will be faster than dict (Even with 6000 entries)
	# for index in aligned_docs: ~6000
		# if doc_words[w1][index] == doc_words[w2][index]: # ==1 
			# co_occur +=1 # slower than looking only at articles in w1 than all aligned
	for de in articles_words[l1][w1]:
		for df in articles_words[l2][w2]:
			if (de,df) in aligned_docs[l1,l2] or (df,de) in aligned_docs[l2,l1]:
				co_occur +=1 
	co_occur *= 1/(float(len(articles_words[l1][w1])) * len(articles_words[l2][w2]))
	return co_occur


def co_occur(w1,w2,l1,l2,articles_words,co_occurring):
	co_occur = 0.0 
	if not (l1,l2) in co_occurring:
		print l1, l2, "not in co-occurring documents!"
        	return co_occur
	if not w1 in co_occurring[l1,l2]:		
		return co_occur
	if not w2 in co_occurring[l1,l2][w1]:
		return co_occur
	co_occur = len(co_occurring[l1,l2][w1][w2])
	co_occur *= 1/(float(len(articles_words[l1][w1])) * len(articles_words[l2][w2]))
	return co_occur		

# Given a word vector in l1 and word vectors in l2, find the most similar word in l2 and return
# along with the cosine similarity
def most_similar(l1_vector,l2_matrix):
        max = 0.0
        w_max = ""
        #print "w_max", w_max
        for w2 in l2_matrix: 
                cos_sim = sim(l1_vector,l2_matrix[w2])
                if cos_sim > max:
                        max = cos_sim
                        w_max = w2
        if max == 0.0:
                print "weird, all vectors have similarity 0"
        return (w_max,max)


# Compute similarity weighted by probability of occurrence in the same aligned document
def most_similar_weighted(w1,word_vectors,l1,l2, articles_words, co_occurring):
	max = 0.0
	w_max = ""
#	print "finding similarity for ", w1
	for w2 in word_vectors[l2]:
		weight = co_occur(w1,w2,l1,l2,articles_words, co_occurring)
		if weight == 0.0:
			continue
#		print w2, weight
#	        weight_sim = sim(word_vectors[l1][w1], word_vectors[l2][w2]) * weight
		weight_sim = weight
		if weight_sim > max:
			max = weight_sim
			w_max = w2
	return (w_max,max)
	


def reverse_index(dim):
	reverse_dim = {}
	for i in dim:
		for lang in dim[i]:
			if lang not in reverse_dim:
				reverse_dim[lang]={}
			translation = dim[i][lang]
			reverse_dim[lang][i] = translation 
	return reverse_dim 

# Create dict with number of documents containing each word
def get_articles_words(doc_words,word_counts):
	articles = {}
	threshold = 5
	for article in doc_words:
		for word in doc_words[article]:
			if word_counts[word] < threshold:
				continue
			if not word in articles:
				articles[word]={}
			if not article in articles[word]:
				articles[word][article] = 1
			else:
				articles[word][article] += 1
	return articles


# Collect co-occurring documents for words in l1 and l2
def co_occurring_docs (co_occurring,l1, l2, doc_words,aligned_docs, word_counts):
	co_occurring[l1,l2] = {}
	co_occurring[l2,l1] = {}
	proc = 0
	threshold = 5
	print "Number of aligned docs:", len(aligned_docs[l1,l2])
	for article1,article2 in aligned_docs[l1,l2]:
		if proc % 100 == 0:
			print proc
		if not article1 in doc_words[l1] or not article2 in doc_words[l2]:
			print "couldn't find articles ", article1, article2
	#		for a1 in doc_words[l1]:
	#			print a1
	#		for a2 in doc_words[l2]:
	#			print a2
			continue
		print "Number of words in ", article1, l1, len(doc_words[l1][article1])
		print "Number of words in ", article2, l2, len(doc_words[l2][article2])			
		# try
		# for article, word in articles[l1]	 # need to change struct?
			# for article, word in articles[l2]		
				# if article1,article2 in aligned_docs[l1,l2]:
					# co_occurring[l1,l2][w1][w2][article1,article2] += 1
		# or for article1 in doc_words[l1]
			# for w1 in doc_words[l1][article1]	
				# for article2 in doc_words[l2] (same as before)
		# or use old co_occur with threshold 

		for w1 in doc_words[l1][article1]:
			if word_counts[l1][w1] < threshold:
				continue
			if not w1 in co_occurring[l1,l2]:
				co_occurring[l1,l2][w1] = {}
			for w2 in doc_words[l2][article2]:
				if word_counts[l2][w2] < threshold:
					continue
				if not w2 in co_occurring[l1,l2][w1]:
					co_occurring[l1,l2][w1][w2]={}
				if not (article1,article2) in co_occurring[l1,l2][w1][w2]:
					co_occurring[l1,l2][w1][w2][article1,article2] = 1
				else:
					co_occurring[l1,l2][w1][w2][article1,article2] += 1
				if not w2 in co_occurring[l2,l1]:
                                	co_occurring[l2,l1][w2] = {}
				if not w1 in co_occurring[l2,l1][w2]:
     					co_occurring[l2,l1][w2][w1] = {}
                                if not (article2,article1) in co_occurring[l2,l1][w2][w1]:
                                        co_occurring[l2,l1][w2][w1][article2,article1] = 1
                                else:
                                        co_occurring[l2,l1][w2][w1][article2,article1] += 1
		proc +=1 
	return co_occurring

def read_init_dim(topic_file):
	init_dim = {}
	with codecs.open(topic_file,'r','utf8') as ts:
		topic_lines = ts.readlines()
	langs = topic_lines[0].strip().split('\t')
	for i in range(1,len(topic_lines)):
		#print "topic ", i
		init_dim[i-1] = {}
		translations = topic_lines[i].strip().split('\t')
		if len(translations) != len(langs):
			sys.exit("number of languages is not equal to number of translations!")
		for j in range(0,len(langs)):
			init_dim[i-1][langs[j]] = translations[j]
			#print langs[j], " : ", translations[j] 
	print "Number of initial dimensions: ", len(init_dim)		
	return init_dim

def update_dimensions(l1,l2,dimensions,candidates):
	pass

def write_counts(word_counts_dict, file_handle):
	for word, freq in word_counts_dict:
		file_handle.write(word + '\t' + str(freq) + '\n')

def main(topic_list,corpus_dir,output_dir,langs,max_dim,window_size,top_b,saved_counts,top_words,saved_word_counts):
	
	languages = langs.split(",") # for now, 2. assumed in same order of dimensions?
	
	if max_dim is None:
		max_dim = 300 # dimensions to add
	if window_size is None:
		window_size =  5  # context 
	if top_b is None:
		top_b = 50
	if top_words is None:
		top_words = 2000
	print "languages to create dictionaries for: "
	for l in languages:
		print l
	init_dim = read_init_dim(topic_list) # topic dimensions by language
	inverted_dim = reverse_index(init_dim)
	print "Created index"
	aligned_docs = {} # store aligned document pairs 
	co_occurring = {} # store co-occurring words
	doc_words = {} # store words per document
	articles_words = {} # store number of documents for each word
	corpus_files = {}
	context_counts = {}
	word_counts = {}
	word_counts_sorted = {}
	tfidf = {}
	word_vectors = {}
	dict_files = {}
	k = 3 # for smoothing PMI matrix
#	thresh = 5 # for cosine similarity. can also keep it to top_ words

	print "Preparing dictionary"
	dir_label = ""
	for l in languages:
		dir_label += l + "_"
	output_dir += dir_label
	print "output_dir:", output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	dict_dir = os.path.join(output_dir,'dict')
	print "dict_dir:", dict_dir
        if not os.path.exists(dict_dir):
        	os.mkdir(dict_dir)	


	print "Collecting aligned documents"
	corpus_files, aligned_docs = collect_aligned_docs(corpus_dir,languages)
#	print "Number of aligned documents:"
 #       for l1, l2 in aligned_docs:
#		print l1, l2, len(aligned_docs[l1,l2])	
#		if [l1,l2] not in co_occurring and [l2,l1] not in co_occurring:
#			co_occurring = co_occurring_docs(l1,l2,doc_words,aligned_docs) 
	print "Reading counts"
       	if saved_counts is not None and saved_word_counts is not None:
		print "Loading"	
		for i in range(0,len(languages)):
			f = languages[i] + ".json"
			if f in os.listdir(saved_counts):
				handle = open(os.path.join(saved_counts,f),'rb')
				context_counts[languages[i]] = json.load(handle)
				handle.close()
			if f in os.listdir(saved_word_counts):
				handle = open(os.path.join(saved_word_counts,f),'rb')
				word_counts[languages[i]] = json.load(handle)
				handle.close()
	counts_file = ""
	for l in languages:
		if not l in context_counts or not l in doc_words:  # TODO either save doc words or remove this condition		 
			print "Creating context counts for ", l
	   		#context_counts[l], word_counts[l], tfidf[l] = compute_context_counts(corpus_files[l],window_size,l,inverted_dim)   
			doc_words[l] = {}
			word_counts[l] = {}
			context_counts[l] = {}
			for article in corpus_files[l]:
                        #	print article
                        	context_counts[l],word_counts[l],this_doc_words =  compute_context_counts(context_counts[l], word_counts[l],corpus_files[l][article]['path'],window_size,l,inverted_dim)	
				doc_words[l][article] = this_doc_words
			print "Number of articles for ", l, ": ", len(corpus_files[l])
			print "Sorting word counts"
        		word_counts_sorted[l] = sorted(word_counts[l].items(), key=operator.itemgetter(1), reverse=True)
			articles_words[l] = get_articles_words(doc_words[l],word_counts[l])

	if saved_counts is None:
		print "Saving all counts"
		count_dir = os.path.join(output_dir,'counts')	
		word_dir = os.path.join(output_dir,'wordcounts')
	#	tfidf_dir = os.path.join(output_dir,'tfidf')
                if not os.path.exists(count_dir):
                        os.mkdir(count_dir)
		if not os.path.exists(word_dir):
			os.mkdir(word_dir)
	#	if not os.path.exists(tfidf_dir):
         #               os.mkdir(tfidf_dir)
		for l in context_counts:
			with open(os.path.join(count_dir,l+'.json'), 'wb') as handle:	
				json.dump(context_counts[l],handle,sort_keys=True,indent=4)
				handle.close()
			with codecs.open(os.path.join(word_dir,l+'.txt'),'wb','utf8') as handle:
                              	 write_counts(word_counts_sorted[l],handle)
				 handle.close()
	#		with codecs.open(os.path.join(tfidf_dir,l+'.txt'),'wb','utf8') as handle:
         #                        write_counts(tfidf[l],handle)
          #                       handle.close()

	print "Number of aligned documents:"
        for l1, l2 in aligned_docs:
                print l1, l2, len(aligned_docs[l1,l2])
		print "Saving co_occurring words"
                if (l1,l2) not in co_occurring and (l2,l1) not in co_occurring:
                        co_occurring = co_occurring_docs(co_occurring,l1,l2,doc_words,aligned_docs,word_counts)
		print "Done"

	print "Building pmi matrix"	
	for l in languages:
		word_vectors[l] = build_PMI_matrix(context_counts[l],word_counts_sorted[l],l,inverted_dim,k)
		output_vectors(word_vectors[l],word_counts_sorted[l],output_dir,l,inverted_dim,1)
	
	dict = {} # bilingual lexicon
        dimensions = inverted_dim 
	translation_candidates = {}
	translation_candidates_sorted = {}
	num_dim = len(init_dim)
	num_new_dim = len(init_dim)
	print "num dim:", num_dim
	print "num new dim:", num_new_dim
	num_iter = 0	
	# Bootstrap until no new dimensions are found or until we reach the max number of dimensions 
        # Max dim constraint can also be removed; for better quality lexicons probably best to set it high
	while (num_new_dim > 0 and num_dim <= max_dim):
		for l1 in languages:
			print l1
			translangs = [lang for lang in languages if lang != l1]
			for l2 in translangs:
				print l2
				# Can do both directions
		#		if (l2,l1) in translation_candidates:
		#			print "Already processed ", l1, l2
		#			continue # TODO should we do it bidirectional since now only taking most freq
				print "building dictionary with ", l2
				translation_candidates[l1,l2] = {}
				translation_candidates_sorted[l1,l2] = {}
				#df = codecs.open(os.path.join(dict_dir,l1+'2'+l2+str(num_iter)),'wb','utf-8')
				remlangs = [lang for lang in languages if (lang != l1 and lang!=l2)]
				if not (l1,l2) in dict:
					dict[l1,l2] = {} 
				if not (l2,l1) in dict:
					dict[l2,l1] = {}
				proc = 0
				for w1, freq in word_counts_sorted[l1]:
					if not w1 in word_vectors[l1]:
						#  excludes rare words and words that don't co-occur in context, for both l1 and l2
						continue
					if proc % 100 == 0:
						print proc
						print "Translation candidates"
						for pair in translation_candidates[l1,l2]:
							print pair[0].decode('utf8'),pair[1].decode('utf8'), translation_candidates[l1,l2][pair]
					if proc == top_words:
						print "Reached top number of words for", l1
						break
				#	print "w1:", w1
				#	(w2,sim1) = most_similar(word_vectors[l1][w1],word_vectors[l2])
					(w2,sim1) = most_similar_weighted(w1,word_vectors,l1,l2, articles_words, co_occurring)
				#	print "w2:", w2, sim1
					if w2 != "" :
						dict[l1,l2][w1] = (w2,sim1) 
						(wtest,sim2) = most_similar_weighted(w2,word_vectors,l2,l1, articles_words, co_occurring)			
				#		(wtest,sim2) = most_similar(word_vectors[l2][w2],word_vectors[l1])
				#		print "wtest:", wtest, sim2
						dict[l2,l1][w2] = (wtest,sim2)
						if wtest == w1: # symmetric
							print "Symmetric. Adding translation candidates " , w1, w2
							translation_candidates[l1,l2][w1,w2] = float(sim1+sim2)/2
						# For the other language pairs we choose the word that has highest cross-lingual similarity between  w1 and w2
						for l3 in remlangs: 
							if (l1,l3) not in dict:
								dict[l1,l3] = {}
					#		(w3,sim3) = most_similar(word_vectors[l1][w1],word_vectors[l3])
							(w3,sim3) = most_similar_weighted(w1,word_vectors,l1,l3, articles_words, co_occurring)
							dict[l1,l3][w1] = (w3,sim3)
							
							if (l2,l3) not in dict:
								dict[l2,l3] = {}		
						#	(w3_2,sim3_2) = most_similar(word_vectors[l2][w2],word_vectors[l3])
							(w3_2,sim3_2) = most_similar_weighted(w2,word_vectors,l2,l3, articles_words, co_occurring)
							dict[l2,l3][w2] = (w3_2,sim3_2)

				# update dimensions here (instead of outer loop) so it stays multilingual
					proc +=1
				print "Sorting translation candidates"
				translation_candidates_sorted[l1,l2] = sorted(translation_candidates[l1,l2].items(), key=operator.itemgetter(1),reverse=True)	
				i = 0
				for (w1,w2), similarity in translation_candidates_sorted[l1,l2]:
					update = 0
					if i == top_b or num_dim >= max_dim:
                                                break
                                	print w1.decode('utf8'), w2.decode('utf8'), similarity
					w_new = {}
					w_new[l1] = w1
					w_new[l2] = w2
					# Find w3
					for lang in remlangs:      
                                        	(w3,sim3) = dict[l1,lang][w1]
                                                (w3_2,sim3_2) = dict[l2,lang][w2]
                                                if sim3_2 > sim3:
                                                	w_new[lang] = w3_2
						else:
							w_new[lang] = w3
					# Resolve collisions
					# update = 0
					collision = 0
					for j in range(0,num_dim):
						if dimensions[l1][j] == w1 and dimensions[l2][j] == w2:
							print "This pair is already in dimensions"
							collision = 1
						elif dimensions[l1][j] == w1:		
							print "w1 ",  w1 , "is already at dimension ", j
							#dimensions[l2][j] == w2
							for other_lang in languages:
								# Only update remaining languages if there is no collision
								if other_lang != l1:
									if w_new[other_lang] not in dimensions[other_lang].values():
										print "dimension ", j , "for " , other_lang , "was ", dimensions[other_lang][j]
										dimensions[other_lang][j] = w_new[other_lang]
										update = 1
										print "Safely updated dimension ", j, "for ", other_lang, "with ", w_new[other_lang]
							collision = 1
						elif dimensions[l2][j] == w2:
							#dimensions[l1][j] = w1
							print "w2 ", w2 , "is already at dimension ", j
							for other_lang in languages:
								if other_lang != l2:      
									if w_new[other_lang] not in dimensions[other_lang].values():
										print "dimension ", j , "for " , other_lang , "was ", dimensions[other_lang][j]
                                                                		dimensions[other_lang][j] = w_new[other_lang]
										update = 1	
										print "Safely updated dimension ", j, "for ", other_lang, "with ", w_new[other_lang]
				      			collision = 1
			
					if collision == 0:
						print "No collision with w1 or w2" 
					# Now we can add the new dimension, but make sure it does not collide for other languages
					# TODO we can find where the collision occurs and remove that dimension instead
					# However we are going through 50 translation candidates, if none of them is updated it means it probably converged
					# We can also increase top_b or leave the collision (ie update even if remlang collides)  and check it only in the next pair iteration (i.e l1,l3 or l3,l4)  					    # but might still get redundancy then 
						update = 1
						for lang in remlangs:
							if w_new[lang] in dimensions[lang].values():
								update = 0

						if update == 1:
							print "Updating dimensions for all languages"
							dimensions[l1][num_dim] = w1
							dimensions[l2][num_dim] = w2
							for lang in remlangs:
								dimensions[lang][num_dim] = w_new[lang]
							num_dim +=1 
							i +=1 
						else:
							print "Could not update, collision with other languages"

				
				
				print "num dim:", num_dim
				print "max dim:", max_dim
				print "Added ", i , " new dimensions when processing languages " , l1 , l2	
				num_new_dim = i			
	
			        #TODO can condition loop on update instead of num_new_dim, or add a max on number of iterations	
				print "Updating context counts and word vectors"
				for lang in languages:
					context_counts[lang]= {}
					word_counts[lang]= {}
					doc_words[lang] = {}
					word_counts_sorted[lang] = {}
				#	articles_words[lang] = {}
					for article in corpus_files[lang]:
                                		context_counts[lang],word_counts[lang],this_doc_words = compute_context_counts(context_counts[lang],word_counts[lang],corpus_files[lang][article]['path'],window_size,lang,inverted_dim)
						doc_words[lang][article] = this_doc_words
                        		print "Sorting word counts"
                        		word_counts_sorted[lang] = sorted(word_counts[lang].items(), key=operator.itemgetter(1), reverse=True)
					word_vectors[lang] = build_PMI_matrix(context_counts[lang],word_counts_sorted[lang],lang,dimensions,k)
					output_vectors(word_vectors[lang],word_counts_sorted[lang],output_dir,lang,dimensions,1)
					print "outputted vectors"
					articles_words[lang] = get_articles_words(doc_words[lang],word_counts[lang])
		num_iter+=1

	if num_dim >= max_dim:
		print "Reached maximum number of dimensions"
	else:
		print "No more new dimensions added, convergence reached"

	print "Number of iterations:", num_iter	
	# Compute and output final dictionaries with similarities
	
	print "printing final vectors"	
	for l1 in languages:
		for l2 in languages:
			if l2 != l1:
				df = codecs.open(os.path.join(dict_dir,l1+'2'+l2+str(num_iter)),'wb','utf-8')
				print "Updating dict with all dimensions"
				for index in dimensions[l1]:
				    dim = dimensions[l1][index]
				    if dim not in dict[l1,l2]: 
					if dim in word_vectors[l1]:
						(translation,sim) = most_similar(word_vectors[l1][dim],word_vectors[l2])
					else: # one of the original dimensions, context counts haven't been collected for it
						(translation,sim) = (dimensions[l2][index],1.0)
					df.write(dim + "\t" + translation + "\t" + str(sim) + "\n")
					#dict[l1,l2][dim] = (translation,sim)
				#for w1, freq in word_counts[l1]: if for all
				print "Updating all dict  entries with new dimensions"
				# Could also add only the symmetric ones
				for w1 in dict[l1,l2]: # if for the words already in the dictionary (top sorted)
					try:
						(w2,sim1) = most_similar(word_vectors[l1][w1],word_vectors[l2])
						#(w2,sim) = dict[l1,l2][w1]
						df.write(w1 + "\t" + w2 + "\t" + str(sim1) + "\n")
					except:
						print w1 , " is in dict but not in word vectors" # because we only update word vectors for those that have context, others will be 0-valued
			        df.close()		
				

	#TODO:
	# Should we decrease (pop extra dimensions)  redundant dimensions instead of not updating collisions
	# threshold for pmi freq
	# Compute and output final dictionaries
	# multigram topics
	# Running with more dimensions or iterations to see if quality is better
	# Treat each language pair as one iteration so the reverse dictionaries are both complete?


if __name__ == '__main__':
	try:
        	parser=argparse.ArgumentParser(description = 'Script to create dictionary from comparable data')
        	parser.add_argument('--topic', '-t', dest = 'topic_list', help = 'list of topics by language', required = True)
       		parser.add_argument('--inputdir', '-c', dest = 'corpus_dir', help = 'directory containing aligned documents by topic', required = True)
        	parser.add_argument('--outputdir', '-o', dest = 'output_dir', help = 'directory containing dictionaries and word vectors', required = True)
		parser.add_argument('--lang', '-l', dest = 'langs', help = 'list of languages, string separated', required = True)
       		parser.add_argument('--dim', '-d', dest = 'max_dim' , help = 'max number of dimensions', required = False, type=int)
		parser.add_argument('--window', '-w', dest = 'window_size' , help = 'context window size', required = False, type=int)
        	parser.add_argument('--rate', '-b', dest = 'top_b' , help = 'number of top dimensions to add (learning rate)', required = False, type=int)
		parser.add_argument('--counts', '-f', dest = 'saved_counts' , help = 'directory with saved context count dicts for each language', required = False)
		parser.add_argument('--top', '-n', dest = 'top_words' , help = 'how many top frequency words to use for creating the dictionary', required = False)
		parser.add_argument('--words', '-r', dest = 'saved_word_counts', help = 'directory with saved word counts', required = False)
		args = vars(parser.parse_args())
    	except:
		print "Please specify required arguments"
		
	main(**args)
