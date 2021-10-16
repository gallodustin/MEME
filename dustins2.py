import argparse, os, sys
import numpy as np
from enum import Enum
import os
import math

# CS 776 Homework #1 - Dustin Gallo Feb 2020
def main(args):

	# Parse input arguments
	seq_file_path = args.sequences_filename
	W = args.width
	model_file_path = args.model
	position_file_path = args.positions
	subseq_file_path = args.subseqs

	# Run my code
	MEME(seq_file_path,W,model_file_path,position_file_path,subseq_file_path)

# This function implements MEME
def MEME(seqInFile,width,modelOutFile,posOutFile,subseqOutFile):

	# Read the input sequence file into a np.array
	seqArray = readSeqFile(seqInFile)

	# Set a starting point for p(0)
	p = getStartingPoint(seqArray, width)

	# Main MEME loop
	prob = 999
	probPlusOne = 0
	count = 0
	while abs(prob - probPlusOne) > 1e-3:

		# Re-estimate Z(t) from p(t-1)	<- E step
		Z = EStep(seqArray, p, width)

		# Calculate initial probability
		prob = logProbXGivenP(seqArray, Z, p, width)

		# Re-estimate p(t) from Z(t)	<- M step
		p = MStep(seqArray, Z, width)

		# Calculate new probability
		probPlusOne = logProbXGivenP(seqArray, Z, p, width)

	# Print model to file
	if (os.path.exists(modelOutFile)):
		os.remove(modelOutFile)
	modelOutFile = open(modelOutFile, "w")
	for i in range(0, np.size(p, axis = 0)):
		if i==0:
			print("A", end = "	", file = modelOutFile)
		if i==1:
			print("C", end = "	", file = modelOutFile)
		if i==2:
			print("G", end = "	", file = modelOutFile)
		if i==3:
			print("T", end = "	", file = modelOutFile)
		for j in range (0, np.size(p, axis = 1)):
			print("%.3f" % p[i,j], end = "	", file = modelOutFile)
		print(file = modelOutFile)
	modelOutFile.close()

	# Print positions to file
	maxVal = 0
	startPosArr = np.empty((1, np.size(seqArray, axis = 0)), int)
	if (os.path.exists(posOutFile)):
		os.remove(posOutFile)
	posOutFile = open(posOutFile, "w")
	for i in range (0, np.size(Z, axis = 0)):
		for j in range (0, np.size(Z, axis = 1)):
			if (j == 0):
				maxVal = Z[i,j]
				maxValCol = j
			elif (Z[i,j] > maxVal):
				maxVal = Z[i,j]
				maxValCol = j
		startPosArr[0, i] = maxValCol
		print(maxValCol, file = posOutFile)
	posOutFile.close()

	# Print subsequences to file
	if (os.path.exists(subseqOutFile)):
		os.remove(subseqOutFile)
	subseqOutFile = open(subseqOutFile, "w")
	for i in range (0, np.size(startPosArr, axis = 1)):
		startPos = startPosArr[0, i]
		subseqStr = ""
		for j in range (startPos, startPos + width):
			base = seqArray[i, j]
			if (base == 1):
				subseqStr += "A"
			if (base == 2):
				subseqStr += "C"
			if (base == 3):
				subseqStr += "G"
			if (base == 4):
				subseqStr += "T"
		print(subseqStr, file = subseqOutFile)
	subseqOutFile.close()

# Get a starting point for the MEME algorithm
def getStartingPoint(seqArray, width):

	if (os.path.exists("out.txt")):
		os.remove("out.txt")
	out = open("out.txt", "w")

	probList = []

	# Collect all distinct subsequences of length W
	subseqArray = np.empty((0, width), int)
	for row in seqArray:
		subseq = np.empty((1, width), int)
		for i in range (0, len(row) - width + 1):
			index = 0
			for j in range (i, i + width):
				subseq[0, index] = row[j]
				index = index + 1
			subseqArray = np.append(subseqArray, subseq, axis = 0)

	# Remove duplicates
	subseqArray = np.unique(subseqArray, axis = 0)

	# For every distinct subsequence
	for subseqRow in subseqArray:

		print(subseqRow, file=out)
		# Intialize an empty p and add background values
		p =  np.empty((4, width + 1), np.longdouble)
		base = 0
		p[0, 0] = 0.25
		p[1, 0] = 0.25
		p[2, 0] = 0.25
		p[3, 0] = 0.25

		# Derive an initial p matrix from this subsequence
		for i in range(0, np.size(p, axis = 0)):
			base = base + 1
			for j in range (1, np.size(p, axis = 1)):
				if (subseqRow[j - 1] == base):
					p[i, j] = 0.7
				else:
					p[i, j] = 0.1

		# Run EM for one iteration
		Z = EStep(seqArray, p, width)
		pTPlusOne = MStep(seqArray, Z, width)
		print(pTPlusOne, file=out)
		prob = logProbXGivenP(seqArray, Z, pTPlusOne, width)
		print(prob, file=out)

		# Store the likelihoods in a list
		probList.append((pTPlusOne, prob))

	# Loop over that array and find the most likely one
	maxProb = -9999999999999999
	for tup in probList:
		if (tup[1] > maxProb):
			maxProb = tup[1]
			bestP = tup[0]
	out.close()
	return bestP

# 1. Re-estimate Z(t) from p(t-1). The E-Step
def EStep(seqArray, p, width):

	# Initialize a Z matrix of 1s
	numRows = np.size(seqArray, axis = 0)
	numCols = np.size(seqArray, axis = 1) - width + 1
	Z = np.empty((numRows, numCols), np.longdouble)
	for i in range (0, numRows):
		for j in range (0, numCols):
			Z[i, j] = 1

	# Populate each entry in Z
	for i in range (0, numRows):
		for j in range (0, numCols):
			for k in range (0, np.size(seqArray, axis = 1)):
				base = seqArray[i, k]
				if k < j:
					Z[i, j] *= p[base - 1, 0]
				elif k > j + width - 1:
					Z[i, j] *= p[base - 1, 0]
				else:
					Z[i, j] *= p[base - 1, k - j + 1]

	# Normalize so rows sum to 1
	for i in range (0, np.size(Z, axis = 0)):
		rowTotal = 0
		for j in range (0, np.size(Z, axis = 1)):
			rowTotal += Z[i, j]
		for j in range (0, np.size(Z, axis = 1)):
			Z[i, j] = Z[i, j] / rowTotal

	return Z

# 2. Re-estimate p(t) from Z(t). The M-Step
# Assume a pseudo-count of 1
def MStep(seqArray, Z, width):

	# Initialize a p matrix of zeroes
	p = np.empty((4, width + 1), np.longdouble)
	for i in range (0, np.size(p, axis = 0)):
		for j in range (0, np.size(p, axis = 1)):
			p[i, j] = 0

	# Calculate the denominator since it's the same for every entry in p
	denom = 0
	for i in range (0, np.size(Z, axis = 0)):
		for j in range (0, np.size(Z, axis = 1)):
			denom += Z[i, j]
	denom += 4 # Pseudocount of 1 per base = 4

	# Get the total count of each character
	numA = 0
	numC = 0
	numG = 0
	numT = 0
	for seqRow in range (0, np.size(seqArray, axis = 0)):
		for seqCol in range (0, np.size(seqArray, axis = 1)):
			if seqArray[seqRow, seqCol] == 1: #A
				numA +=1
			if seqArray[seqRow, seqCol] == 2: #C
				numC +=1
			if seqArray[seqRow, seqCol] == 3: #G
				numG +=1
			if seqArray[seqRow, seqCol] == 4: #T
				numT +=1

	# Set each entry in p to the numerator value (no pseudocount yet)
	N = np.size(seqArray, axis = 0)
	L = np.size(seqArray, axis = 1)
	for c in range (0, np.size(p, axis = 0)):
		for k in range(1, np.size(p, axis = 1)):
			for i in range(0, N):
				for j in range(0, L - width + 1):
					if (seqArray[i, j + k - 1] == c + 1):
						p[c, k] += Z[i, j]

		# Calculate the background values
		rowSum = 0
		for j in range (1, np.size(p, axis = 1)):
			rowSum += p[c, j]
		if c == 0: #A
			p[c ,0] = numA - rowSum
		if c == 1: #C
			p[c, 0] = numC - rowSum
		if c == 2: #G
			p[c, 0] = numG - rowSum
		if c == 3: #T
			p[c, 0] = numT - rowSum

	# Calculate final p matrix by adding the pseudocount dividing by the denominator
	for j in range (0, np.size(p, axis = 1)):
		colSum = 0
		for c in range (0, np.size(p, axis = 0)):
			colSum += p[c, j]
		for c in range (0, np.size(p, axis = 0)):
			p[c, j] = (p[c, j] + 1)/(colSum + 4)

	return p

# Calculate the probablity of the provided sequence given the current model
def logProbXGivenP(seqArray, Z, p, width):

	# Initialize probability to 1 since we will be multiplying
	totalProb = 1

	# Do the calculation from slide 23
	N = np.size(seqArray, axis = 0) # Number of sequences
	L = np.size(seqArray, axis = 1) # Sequence length
	for i in range (0, N):
		rowProb = np.longdouble(1)
		for j in range (1, L - width + 1):
			for k in range (1, j):
				rowProb *= p[seqArray[i, j] - 1, 0]
			for k in range (j, j + width):
				rowProb *= p[seqArray[i, j] - 1, k - j + 1]
			for k in range (j + width, L + 1):
				rowProb *= p[seqArray[i, j] - 1, 0]
		rowProb = np.log2(rowProb)
		totalProb += rowProb

	return totalProb

# Read the input sequence file into a np.array
def readSeqFile(path):
	with open(path, "r") as file:
		lines = []
		for line in file.readlines():
			linevals = []
			i = 0
			for letter in line:
				if letter == "A":
					linevals.append(1)
				if letter == "C":
					linevals.append(2)
				if letter == "G":
					linevals.append(3)
				if letter == "T":
					linevals.append(4)
				i = i + 1
			lines.append(linevals)
		pass
	return np.array(lines)

# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
	# Note: this example shows named command line arguments.  See the argparse
	# documentation for positional arguments and other examples.
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('sequences_filename',
		help='sequences file path.',
		type=str)
	parser.add_argument('--width',
		help='width of the motif.',
		type=int, default=6)
	parser.add_argument('--model',
		help='model output file path.',
		type=str,
		default='model.txt')
	parser.add_argument('--positions',
		help='position output file path.',
		type=str,
		default='positions.txt')
	parser.add_argument('--subseqs',
		help='subsequence output file path.',
		type=str,
		default='subseqs.txt')

	args = parser.parse_args()
	# Note: this simply calls the main function above, which we could have
	# given any name
	main(args)
