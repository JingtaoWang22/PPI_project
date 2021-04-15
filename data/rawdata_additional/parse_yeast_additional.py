import numpy as np
import sys
import string

filename_ppi = "4932.protein.links.v11.0.txt"
filename_seq = "4932.protein.sequences.v11.0.fa"
outfile_dic = "yeast_STRINGDB_dic.txt"
outfile_ppi = "yeast_STRINGDB_ppi.txt"
confidence_thresh = 999    #confidence * 1000
negative_factor = 1

foutdic = open(outfile_dic, "w")
foutppi = open(outfile_ppi, "w")

np.random.seed(1234)
##########Get sequences info##########
f = open(filename_seq, "r")
seq_dic = {}
lines = f.readlines()
i = 0
curr_id = ""
curr_seq = ""
while i < len(lines):
	if lines[i][0] == ">":
		seq_dic[curr_id] = curr_seq
		curr_id = lines[i].strip(">\n")
		curr_seq = ""
	else:
		curr_seq += lines[i].strip("\n")
	i += 1

seq_dic[curr_id] = curr_seq
#print(seq_dic)

#########Get interaction pairs and write output######
f = open(filename_ppi, "r")
peptide_sofar = set([])
peptide_sofar_all = set([])
peptide_positive = {}
peptide_positive_all = {}
positive_count = 0
lines = f.readlines()[1:]
for l in lines:
	content = l.split(" ")
	p1, p2 = content[0], content[1]
	if p1 in peptide_sofar_all:
		peptide_positive_all[p1].append(p2)
	else:
		peptide_positive_all[p1] = [p2]
		peptide_sofar_all.add(p1)
	if p2 in peptide_sofar_all:
		peptide_positive_all[p2].append(p1)
		peptide_sofar_all.add(p2)
	if int(content[2]) >= confidence_thresh:
		foutppi.write(content[0] + "	" + content[1] + "	" + "1\n")
		positive_count += 1
		if p1 in peptide_sofar:
			peptide_positive[p1].append(p2)
		else:
			peptide_sofar.add(p1)
			peptide_positive[p1] = [p2]
			foutdic.write(p1 + "	" + seq_dic[p1] + "\n")
		if p2 in peptide_sofar:
			peptide_positive[p2].append(p1)
		else:
			peptide_sofar.add(p2)
			peptide_positive[p2] = [p1]
			foutdic.write(p2 + "	" + seq_dic[p2] + "\n")

		if p1 == "4932.YER087C-B" or p2 == "4932.YER087C-B":
			print(p1, p2, content[2])

print(positive_count, len(peptide_sofar))
print(peptide_positive["4932.YER087C-B"])
print(peptide_positive_all["4932.YER087C-B"])
###########Create negative interactions###############
total_peptides = len(peptide_sofar)
peptide_sofar = list(peptide_sofar)
peptide_negative = {}
negative_count = 0
while negative_count < positive_count * negative_factor:
	p1, p2 = peptide_sofar[np.random.choice(total_peptides, 1)[0]], peptide_sofar[np.random.choice(total_peptides, 1)[0]]
	if (not p2 in peptide_positive[p1]) and (not p2 in peptide_positive_all[p1]) and p1 != p2:
		if not p1 in peptide_negative.keys():
			peptide_negative[p1] = []
		if not p2 in peptide_negative.keys():
			peptide_negative[p2] = []
		if (not p2 in peptide_negative[p1]) and (not p1 in peptide_negative[p2]):
			peptide_negative[p1].append(p2)
			peptide_negative[p2].append(p1)
			negative_count += 1
			foutppi.write(p1 + "	" + p2 + "	" + "0\n")
		if p1 == "4932.YER087C-B" or p2 == "4932.YER087C-B":
			print(p1, p2)


