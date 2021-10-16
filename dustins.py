import argparse, os, sys
import numpy as np
from enum import Enum
import os
import math

debug = True
debugFile = "testout2.txt"

# CS 776 Homework #1 - Dustin Gallo Feb 2020
def main(args):

	global debug
	global debugFile

	# Open output file
	if (debug):
		if (os.path.exists(debugFile)):
			os.remove(debugFile)
		debugFile = open(debugFile, "w")

	# Parse input arguments
	seq_file_path = args.sequences_filename
	W = args.width
	model_file_path = args.model
	position_file_path = args.positions
	subseq_file_path = args.subseqs

	# Run my code
	MEME(seq_file_path,W,model_file_path,position_file_path,subseq_file_path)

	# Close output file
	if (debug):
		debugFile.close()

# Basic EM approach from lecture:

# Given: length parameter W, and a training set of sequences
	# t = 0
	# Set initial values for p(0)
	# Do:
		# ++t
		# Re-estimate Z(t) from p(t-1)	<- E step
		# Re-estimate p(t) from Z(t)	<- M step
	# Until change in p(t) < 1e-3
# Return: p(t), Z(t)

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
	while abs(prob - probPlusOne) > 1e-3 and count < 100:

		# Re-estimate Z(t) from p(t-1)	<- E step
		Z = EStep(seqArray, p, width)
		printMatrix("Z matrix", Z, debugFile)

		# Calculate initial probability
		prob = logProbXGivenP(seqArray, Z, p, width)
		print("prob:", file = debugFile)
		print(prob, file = debugFile)

		# Re-estimate p(t) from Z(t)	<- M step
		p = MStep(seqArray, Z, width)
		printMatrix("p matrix", p, debugFile)

		# Calculate new probability
		probPlusOne = logProbXGivenP(seqArray, Z, p, width)
		print("probPlusOne:", file = debugFile)
		print(probPlusOne, file = debugFile)

		# Temporary failsafe to prevent infinite loop
		count += 1

	# Return p(t) and Z(t)
	if (count >= 100):
		print("failsafe was hit", file = debugFile)
	printMatrix("final p matrix:", p, debugFile)
	printMatrix("final Z matrix:", Z, debugFile)

# Get a starting point for the MEME algorithm
def getStartingPoint(seqArray, width):

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
		prob = logProbXGivenP(seqArray, Z, pTPlusOne, width)

		# Store the likelihoods in a list
		probList.append((p, prob))

	# Loop over that array and find the most likely one
	maxProb = -9999999999999999
	for tup in probList:
		if (tup[1] > maxProb):
			maxProb = tup[1]
			bestP = tup[0]

	return bestP

# 1. Re-estimate Z(t) from p(t-1). The E-Step
def EStep(seqArray, p, width):

	# Initialize an empty Z matrix
	numRows = np.size(seqArray, axis = 0)
	numCols = np.size(seqArray, axis = 1) - width + 1
	Z = np.empty((numRows, numCols), np.longdouble)

	# Populate each entry in Z
	for i in range (0, numRows):
		for j in range (0, numCols):
			Z[i, j] = (0.25)**(numCols - 1)
			for motifPos in range (0, width):
				baseNum = seqArray[i, j + motifPos]
				Z[i, j] *= p[baseNum - 1, motifPos + 1]

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
	print("ACGT")
	print(numA)
	print(numC)
	print(numG)
	print(numT)

	# Set each entry in p to the numerator value (no pseudocount yet)
	print("Z entries")
	for pRow in range (0, np.size(p, axis = 0)):
		print("pRow")
		print(pRow)
		for seqRow in range (0, np.size(seqArray, axis = 0)):
			for seqCol in range (0, np.size(seqArray, axis = 1)):
				if (seqArray[seqRow, seqCol] == pRow + 1):
					for motifPos in range (0, width):
						if (seqCol >= motifPos) and ((np.size(seqArray, axis = 1) - seqCol) >= (width - motifPos)):
							p[pRow, motifPos + 1] += Z[seqRow, seqCol - motifPos - 1]
							print(motifPos)
							print(Z[seqRow, seqCol - motifPos - 1])

	# Calculate the background values
	print("ACGT rowsum")
	for pRow in range (0, np.size(p, axis = 0)):
		rowSum = 0
		for pCol in range(1, np.size(p, axis = 1)):
			rowSum += p[pRow, pCol]
			# Add the pseudocount to the numerator
			p[pRow, pCol] += 1
		if pRow == 0: #A
			p[pRow, 0] = numA - rowSum + 1
		if pRow == 1: #C
			p[pRow, 0] = numC - rowSum + 1
		if pRow == 2: #G
			p[pRow, 0] = numG - rowSum + 1
		if pRow == 3: #T
			p[pRow, 0] = numT - rowSum + 1
		print(rowSum)

	# Calculate final p matrix by dividing by the denominator
	for i in range (0, np.size(p, axis = 0)):
		for j in range (0, np.size(p, axis = 1)):
			p[i, j] /= denom
	print("denom")
	print(denom)
	print(p)
	return p

# Calculate the probablity of the provided sequence given the current model
def logProbXGivenP(seqArray, Z, p, width):

	# Initialize probability to 1 since we will be multiplying
	totalProb = 1

	# Do the calculation from slide 23
	N = np.size(seqArray, axis = 0) # Number of sequences
	L = np.size(seqArray, axis = 1) # Sequence length
	for i in range (0, N):
		rowProb = 1
		for j in range (1, L - width + 1):
			for k in range (1, j):
				rowProb *= 0.25
			for k in range (j, j + width):
				rowProb *= p[seqArray[i, j] - 1, k - j + 1]
			for k in range (j + width, L + 1):
				rowProb *= 0.25
		rowProb = math.log(rowProb,2)
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

# Print np.array in human-readable format, for debugging only
def printMatrix(title, arr, out):
	print(title, file = out)
	for row in arr:
		rowstr = ""
		for i in range(0, len(row)):
			rowstr += (str)(row[i]) + " "
		print(rowstr, file = out)
	print("", file = out)
	return

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
