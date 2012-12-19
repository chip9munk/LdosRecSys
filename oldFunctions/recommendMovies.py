'''
Created on Oct 16, 2012

@author: Ante Odic
'''

import numpy
  
def topNIids(N, matrixIidsRatings):
    sortedMatrixIidsRatings=matrixIidsRatings[matrixIidsRatings[:,1].argsort()]
    top = sortedMatrixIidsRatings[:,0]
    top = list(top)
    top.reverse()
    # reverse in numpy's array
    #top = top[::-1]
    return top[:N]

def bottomNIids(N, matrixIidsRatings):
    sortedMatrixIidsRatings=matrixIidsRatings[matrixIidsRatings[:,1].argsort()]
    top = sortedMatrixIidsRatings[:,0]
    return top[:N]
	
def mockupTopN(uID,N):
	d = [34,56,78,93,42,56,43,12,34,67,78,90,1,2,4,3,5,7,6,9,8,] 
	rez = d[:N]
	return rez
	
