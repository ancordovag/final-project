dirname = 'europarl'
german_dir = 'Europarl.de-es.de'
spanish_dir = 'Europarl.de-es.es'

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

german_filename = dirname + '/' + german_dir
spanish_filename = dirname + '/' + spanish_dir

german_file = open(german_filename, mode='rt', encoding='utf-8')
spanish_file = open(spanish_filename, mode='rt', encoding='utf-8')
new_file = open('de-es.txt',mode='w',encoding='utf-8')
max_lines = 20
for i in range(max_lines):
	german_line = german_file.readline().rstrip('\n')
	spanish_line = spanish_file.readline()
	new_line = german_line + '\t' + spanish_line
	new_file.write(new_line)

german_file.close()
spanish_file.close()
new_file.close()
