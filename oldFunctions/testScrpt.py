import calculateMF
import recommendMovies
from bottle import route, run, post, request
from bottle import default_app

	
@post('/recsys')
def getRequestForRecSys():
		
	functionName = request.forms.get('fName')
	
	rez = request.forms.get('clientID')
	rez = str(rez) + 'blabla'
	
	if functionName == 'getTopNItems': rez = recommendMovies.mockupTopN(request.forms.get('userID'),int(request.forms.get('N')))
	elif functionName == 'getRatingPrediction': rez = calculateMF.mockupPredict(request.forms.get('userID'), request.forms.get('itemID'))
	else: rez = 'No function!!'
	
	return str(rez)	
