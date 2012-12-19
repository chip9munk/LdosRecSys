import sumica
import calculateMF
import recommendMovies
from bottle import route, run, post, request
from bottle import default_app

@route('/myplus')
@route('/myplus/<number>')
def mysum_print(number='10'):
	output = int(number)+1
	b=str(output)
	return (b)

@route('/myminus')
@route('/myminus/<x>/<y>')
def mysum_print(x='100',y='1'):
    
	c=int(x)-int(y)
	
	return str(c)	

#@route('/mycall')
@post('/mycall')
def mysum_callSum():
	uID = request.forms.get('userID')
	iID = request.forms.get('itemID')
	
	d=int(uID)+int(iID)
	
	return str(d)
	
@post('/ratingPred')
def predictRating():
	uID = request.forms.get('userID')
	iID = request.forms.get('itemID')
	
	d = calculateMF.mockupPredict(uID, iID)
	
	return str(d)
	
@post('/topN')
def getListTopN():
	uID = request.forms.get('userID')
	N = request.forms.get('N')
	
	d = recommendMovies.mockupTopN(uID,int(N))
	
	return str(d)	
