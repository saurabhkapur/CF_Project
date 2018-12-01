import urllib2
from bs4 import BeautifulSoup
import csv
import json


header = []
with open('Dataset/Last_fm_dataset/lastfm.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	header = next(csv_reader)

# print len(header)
header = header[1:]
art_info = {}
for i in range(len(header)):
	wiki = "https://www.last.fm/music/" + header[i] + "/+wiki"
	try:
		page = urllib2.urlopen(wiki)
	except:
		print"error"
		art_info[i] = ""
		continue
	soup = BeautifulSoup(page, 'lxml')
	data = soup.findAll('div',attrs={"class":"wiki-content"})

	# print data
	t = data[0].findAll('p')
	t = t[:3]
	f_text = ""
	for x in range(len(t)):
	    f_text += t[x].text
 	if (f_text == ""):
 		print "yes"
 	else:
 		print i
	art_info[i] = f_text

with open('Dataset/Last_fm_dataset/art_info.json', 'w') as outfile:
    json.dump(art_info, outfile)
# print data
