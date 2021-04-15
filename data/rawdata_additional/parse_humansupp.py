import numpy as np
import sys
import string

filename_p = "humansuppA.txt"
filename_n = "humansuppB.txt"
outfile_dic = "human_dic.txt"
outfile_ppi = "human_ppi.txt"

letters = string.ascii_uppercase
integers = "0123456789"

foutdic = open(outfile_dic, "w")
foutppi = open(outfile_ppi, "w")

set_protein = set([])

##########Positives##########
f = open(filename_p, "r")
lines = f.readlines()

start = False
line_ind = 0
while not start:
	l = lines[line_ind + 1]
	start = (l[0] == "1")
	line_ind += 1

while line_ind < len(lines):
	ls = lines[line_ind: line_ind + 5]
	#print(ls)

	curr_ps = ls[0].split(" ")
	curr_p1, curr_p2 = curr_ps[1], curr_ps[3].strip()
	seq1, seq2 = ls[2], ls[4]
	foutppi.write(curr_p1 + "	" + curr_p2 + "	" + "1\n")
	if not curr_p1 in set_protein:
		set_protein.add(curr_p1)
		foutdic.write(curr_p1 + "	" + seq1)
	if not curr_p2 in set_protein:
		set_protein.add(curr_p2)
		foutdic.write(curr_p2 + "	" + seq2)

	line_ind += 5

##########Negatives##########
f = open(filename_n, "r")
lines = f.readlines()

start = False
line_ind = 0
while not start:
	l = lines[line_ind + 1]
	start = (l[0] == "1")
	line_ind += 1

while line_ind < len(lines):
	ls = lines[line_ind: line_ind + 5]
	#print(ls)

	curr_ps = ls[0].split(" ")
	curr_p1, curr_p2 = curr_ps[1], curr_ps[3].strip()
	seq1, seq2 = ls[2], ls[4]
	foutppi.write(curr_p1 + "	" + curr_p2 + "	" + "0\n")
	if not curr_p1 in set_protein:
		set_protein.add(curr_p1)
		foutdic.write(curr_p1 + "	" + seq1)
	if not curr_p2 in set_protein:
		set_protein.add(curr_p2)
		foutdic.write(curr_p2 + "	" + seq2)

	line_ind += 5

