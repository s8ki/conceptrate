"""
 /relatedness?node1=/c/en/tea_kettle&node2=/c/en/coffee_pot
 You can make 3600 requests per hour to the ConceptNet API, with bursts of 120 requests per minute allowed. The /related and /relatedness endpoints count as two requests when you call them.

This means you should design your usage of the API to average less than 1 request per second.

 Pages 17
Starting points

FAQ
Web API
Downloads
Reproducibility

Copying and sharing
Build process
Running your own copy
Details

Edges
Relations
Languages
URI hierarchy
Clone this wiki locally

"""
import requests
import re
import sys
import gensim.downloader as api

USAGE = "%s <term>" % (sys.argv[0])
ENGLISH = "?filter=/c/en"
ENGLISH_FILTER = "&other=/c/en"
END = "&limit=1000&filter/c/en"

wv = None
"""
def get_en_related(term):
	obj = requests.get('http://api.conceptnet.io/related/c/en/' + term).json()
	return [re.findall("/(\w*$)", edge['@id'])[0] for edge in obj['related'] if "/en/" in edge['@id']]
"""


def get_en_related(term):
	#obj = requests.get('http://api.conceptnet.io/related/c/en/' + term + ENGLISH).json()
	obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + term + '&rel=/r/RelatedTo' + ENGLISH_FILTER).json()
	# filter relatedto, only get desired word that is not term
	result = [[re.findall("/en/((?!%s/)\w*)/[,\]]" % (term), edge['@id']), edge['weight']] for edge in obj['edges']]
	return map(lambda x: [x[0][0], x[1]], filter(lambda x: len(x[0])>0, result))

def get_en_capable(term):
	obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + term + '&rel=/r/CapableOf' + END).json()
	# filter relatedto, only get desired word that is not term
	result = [[re.findall("/en/((?!%s/)\w*)/[,\]]" % (term), edge['@id']), edge['weight']] for edge in obj['edges']]
	return map(lambda x: [x[0][0], x[1]], filter(lambda x: len(x[0])>0, result))

def get_en_category(term, category):
	obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + term + '&rel=/r/' + category + END).json()
	# filter relatedto, only get desired word that is not term
	#result = [[re.findall("/en/((?!%s/)\w*)/[,\]]" % (term), edge['@id']), edge['weight']] for edge in obj['edges']]
	result = [[re.findall(r"/en/(\w*)/\]", edge['@id']), edge['weight']] for edge in obj['edges']]
	filtered = list(filter(lambda x: len(x[0])>0 and x[0][0] != term, result))
	result2 = list(map(lambda x: [x[0][0], x[1]], filtered))
	return result2

def ul2s(term):
	return "term".replace("_", " ")

def s2ul(term):
	return "term".replace(" ", "_")

def wrapped_similarity(a, b, wv):
	result = 0.0
	try:
		result = wv.similarity(a, b)
	finally:
		return result
	
def get_word2vec_scores(li, term):
	# see other models here: https://www.diycode.cc/projects/RaRe-Technologies/gensim-data
	result = [element + [wrapped_similarity(term, element[0], wv)] for element in li]
	return result
	
def tst_related(path):
	with open(path, "rb") as f:
		terms = f.readlines()

	base_path = r"C:\Users\soki\Documents\HebrewU\Project dafna\bert-babble\\"
	for term in terms:
		term = term.decode("utf-8").strip()
		print(term)
		result = get_en_related(term)
		result2 = get_word2vec_scores(result, term)

		with open(base_path + term + "1.txt", "wb") as f:
			for r in result2:
				f.write(("%-20s %-10s %-10s\r\n" % (r[0], r[1], r[2])).encode())

		#reorder by word2vec
		result2.sort(key=lambda x: x[-1], reverse=True)
		with open(base_path + term + "2.txt", "wb") as f:
			for r in result2:
				f.write(("%-20s %-10s %-10s\r\n" % (r[0], r[1], r[2])).encode())

def tst_categories(terms, categories):

	# with open(path, "rb") as f:
	# 	terms = f.readlines()

	base_path = r"C:\Users\soki\Documents\HebrewU\Project dafna\bert-babble\out\\"
	for term in terms:
		print("====================")
		print(term)
		print("====================")
		with open(base_path + term + "_conceptnet.txt", "wb") as f:
			f.write("====================\r\n".encode())
			f.write(("%s\r\n" % (term)).encode())
			f.write("====================\r\n".encode())
			for cat in categories:
				#term = term.decode("utf-8").strip()
				print("################")
				print(cat)
				print("################")
				f.write("################\r\n".encode())
				f.write(("%s\r\n" % (cat)).encode())
				f.write("################\r\n".encode())

				result = get_en_category(term, cat)

				for r in result:
					f.write(("%-20s %-10s\r\n" % (r[0], r[1])).encode())
					print("%-20s %-10s" % (r[0], r[1]))

categories_cn = ["CapableOf", "IsA", "HasProperty", "AtLocation", "UsedFor"]
categories = ["is capable of", "is a type of", "is", "can be found in", "is used for"]
words = ["screen", "king", "queen", "dog", "wind", "water", "person", "man", "heart", "brain", "hand", "lion", "wheel", "chair", "table", "country", "tree", "glass", "light", "murder", "bird", "computer"]
words = ["person"]
# categories_cn = ["AtLocation"]

def main():
	#
	# if len(sys.argv) != 2:
	# 	print(USAGE)
	# 	sys.exit(0)

	#wv = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
	#tst_capable(words)
	tst_categories(words, categories_cn)


#TODO: debug http://api.conceptnet.io/query?node=/c/en/screen&rel=/r/IsA&limit=1000
#"@id": "/a/[/r/IsA/,/c/en/screen/n/wn/artifact/,/c/en/display/n/wn/artifact/]",
if __name__ == "__main__":
	main()
