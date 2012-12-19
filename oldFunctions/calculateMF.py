'''
Created on Oct 15, 2012

@author: Ante Odic
'''

import numpy

# fix predicted ratings higher or lower then max and min ratings respectively
def fixPredictedR ( predictedR, minRating, maxRating ):
    if predictedR > maxRating:
        predictedR = maxRating
    elif predictedR < minRating:
        predictedR = minRating
    else:
        predictedR = predictedR
    return predictedR   
    
# calculate predicted rating by the multiple context MF model
def multipleCntxtModel ( u, bu, bi, p, q, cntxtBiases ):
    
    pq = numpy.matrix(p)*numpy.matrix(q).T
    cntxtBiasSum = numpy.sum(cntxtBiases)
    return u + bu + bi + pq + cntxtBiasSum

# calculate predicted rating by the plain MF model
def plainModel ( u, bu, bi, p, q ):
    pq = numpy.matrix(p)*numpy.matrix(q).T
    return u + bu + bi + pq

# MOCKUP calculate predicted rating
def mockupPredict (uID, iID):
	return 3.467
