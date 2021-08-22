import sys

# Name of the directory and files
dirname = 'europarl'
german_dir = 'Europarl.de-es.de'
spanish_dir = 'Europarl.de-es.es'

def length_doc(filename):
	"""Get the lines of text in a file"""
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	line_count = 0
	for line in file:
		if line != "\n":
			line_count += 1
	# close the file
	file.close()
	return line_count

german_filename = dirname + '/' + german_dir
spanish_filename = dirname + '/' + spanish_dir

# Open the files
german_file = open(german_filename, mode='rt', encoding='utf-8')
spanish_file = open(spanish_filename, mode='rt', encoding='utf-8')
new_file = open('de-es.txt',mode='w',encoding='utf-8')
max_lines = 1000 #length_doc(german_filename) #1887879
print("Length of the Document: ",max_lines)
#sys.exit()
# Fusion the files in only one document
# Sentence pairs separated by tab
for i in range(max_lines):
	german_line = german_file.readline().rstrip('\n')
	spanish_line = spanish_file.readline()
	new_line = german_line + '\t' + spanish_line
	new_file.write(new_line)

german_file.close()
spanish_file.close()
new_file.close()
