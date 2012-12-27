'''
Created on Nov 6, 2012

@author: Ante Odic
'''
import numpy
import logging
logging.basicConfig(filename='./logs/example.log',level=logging.DEBUG)
import math
import configparser
from bottle import route, run, post, request
from bottle import default_app
############################## GLOBAL VARIABLES WITH DEFAULT VALUES ##############################

# mf construction parameters  
#contextType = "multiple"
#contextVariable = 1

# data paths
mainDataPath = './clientInformation'
mainResultPath = './trainedData'
#dataSource = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/data/LDOScontextTRAINnoLAST.txt' #make "empty" default, this one is just for testing
#validationDataSource = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/data/MovieATcontextTEST2.txt'


#userBiasesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userBias.txt'
#itemBiasesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemBias.txt'
#globalBiasResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/globalBias.txt'
#userFeaturesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userFeatures.txt'
#itemFeaturesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemFeatures.txt'
#userFeaturesResultMultiple = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userFeaturesMultiple.txt'
#itemFeaturesResultMultiple = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemFeaturesMultiple.txt'

# global matrices
userBiasesMatrix = 0
itemBiasesMatrix = 0
globalBiasMatrix = 0
userFeaturesMatrix = 0 
itemFeaturesMatrix = 0 
multipleContextUserBiasesMatrix = 0
contextValPerVar = 0

# mf parameters
biasLearning = 0
pLearningRate = 0.01
qLearningRate = 0.01
bLearningRate = 0.01
regularization = 0.005
numOfFeatures = 10
numOfIterations = 100
initFeatureValue = 0.03
####################################################################################################

@post('/recsys')
def getRequestForRecSys():
		
	functionName = 	request.forms.get('fName')
	
	if str(request.forms.get('context')) == 'None':
		context = (0,0)
	
	if functionName == 'getTopN_MF': rez = getTopN_MF(request.forms.get('clientID'),int(request.forms.get('userID')),int(request.forms.get('N')), context)
	elif functionName == 'getRating_MF': rez = getRating_MF(request.forms.get('clientID'),int(request.forms.get('userID')), int(request.forms.get('itemID')),context)
	elif functionName == 'getBottomN_MF': rez = getBottomN_MF(request.forms.get('clientID'),int(request.forms.get('userID')),int(request.forms.get('N')), context)
	elif functionName == 'getRandomItems': rez = getRandomItems(request.forms.get('clientID'),int(request.forms.get('N')))
	elif functionName == 'getDiverseN': 
		initialSetIds = request.forms.get('initialSetIds')
		initialSetIds=initialSetIds.split(',')
		initialSetIds=[int(i) for i in initialSetIds]
		rez = getDiverseN(request.forms.get('clientID'),initialSetIds,int(request.forms.get('N')))
	elif functionName == 'getSimilarN': rez = getSimilarN(request.forms.get('clientID'),int(request.forms.get('initialSetId')),int(request.forms.get('N')))
	
	
	else: rez = 'No function!!'
	
	return str(rez)	

@post('/operate')
def trainRecSys():
		
	functionName = 	request.forms.get('fName')
	
		
	if functionName == 'train': 
		train_MF(request.forms.get('clientID')) 
		rez = 'Training completed!'
	elif functionName == 'validate': 
		validate_MF(request.forms.get('clientID'))
		rez = 'Validation completed!'
	elif functionName == 'initialize': 
		initializeClientConfFile (request.forms.get('clientID'), request.forms.get('contextType'))
		rez = 'Initialization completed!'
	else: rez = 'No function!!'
	
	
	
	return rez


######################################## CLIENT FUNCTIONS ##########################################
def getRating_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=itemID, [3]=context
    confFileName = mainDataPath + '/' + arg[0] + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return float(getRating(arg[0],arg[1],arg[2]))
    elif cType == 'multiple':
        return float(getRating_multiContext(arg[0],arg[1],arg[2],arg[3]))
        
    
def getTopN_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=itemID, [3]=context
    confFileName = mainDataPath + '/' + arg[0] + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return getTopN(arg[0],arg[1],arg[2])
    elif cType == 'multiple':
        return getTopN_multiContext(arg[0],arg[1],arg[2],arg[3])

def getBottomN_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=itemID, [3]=context
    confFileName = mainDataPath + '/' + arg[0] + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return getBottomN(arg[0],arg[1],arg[2])
    elif cType == 'multiple':
        return getBottomN_multiContext(arg[0],arg[1],arg[2],arg[3])
####################################################################################################


######################################## Provider Functions ##################################################

def train_MF(clientName):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        trainPlainModel(clientName)
    elif cType == 'multiple':
        trainMultipleCntxtModel(clientName)
    
def validate_MF(clientName):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return validatePlainModel(clientName)
    elif cType == 'multiple':
        return validateMultipleCntxtModel(clientName)


####################################################################################################


######################################## CLIENT CONFIGURATION ##################################################

def initializeClientConfFile (clientName, contextType):
    
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    datasetName = config.get('dataPath', 'datasetName')
    #return confFileName
	
    dataSet=numpy.loadtxt(datasetName, delimiter=';')
    # get dataset information
    numOfItems = len(numpy.unique(dataSet[:,1]))
    numOfUsers = len(numpy.unique(dataSet[:,0]))
    minRating = min(numpy.unique(dataSet[:,2]))
    maxRating = max(numpy.unique(dataSet[:,2]))
    itemIDs = numpy.unique(dataSet[:,1])
    itemIDs = [int(i) for i in itemIDs]
    userIDs = numpy.unique(dataSet[:,0])
    userIDs = [int(i) for i in userIDs]
    getContextInformation(dataSet)
    contextVarsAndVals = contextValPerVar
    contextVarsAndVals = [int(i) for i in contextVarsAndVals]
    
    relevantContextMask = [1]*len(contextVarsAndVals)
    
    
    config.add_section('dataInfo')
    config.set('dataInfo', 'numOfItems', numOfItems)
    config.set('dataInfo', 'numOfUsers', numOfUsers)
    config.set('dataInfo', 'minRating', minRating)
    config.set('dataInfo', 'maxRating', maxRating)
    config.set('dataInfo', 'itemids', itemIDs)
    config.set('dataInfo', 'userids', userIDs)
    config.set('dataInfo', 'contextVarsAndVals', contextVarsAndVals)
    with open(confFileName, 'w') as configfile:
        config.write(configfile)
    
        config.add_section('modelInfo')
    config.set('modelInfo', 'contextType', contextType)
    config.set('modelInfo', 'relevantContextMask', relevantContextMask)
    with open(confFileName, 'w') as configfile:
        config.write(configfile)
    
    config.add_section('mfParameters')
    config.set('mfParameters', 'biasLearning', biasLearning)
    config.set('mfParameters', 'pLearningRate', pLearningRate)
    config.set('mfParameters', 'qLearningRate', qLearningRate)
    config.set('mfParameters', 'bLearningRate', bLearningRate)
    config.set('mfParameters', 'regularization', regularization)
    config.set('mfParameters', 'numOfFeatures', numOfFeatures)
    config.set('mfParameters', 'numOfIterations', numOfIterations)
    config.set('mfParameters', 'initFeatureValue', initFeatureValue)

# Writing our configuration file to 'example.cfg'

    with open(confFileName, 'w') as configfile:
        config.write(configfile)
    
####################################################################################################

######################################## SET FUNCTIONS ##################################################
#def setContextType(cType):
#    global contextType
#    contextType = cType
    
def setContextVariable(variable):
    global contextVariable
    contextVariable = variable 
    
#def setDataSource(filename):
#   global dataSource
#   dataSource = filename
    
def setValidationDataSource(filename):
    global validationDataSource
    validationDataSource = filename

def setBiasLearning(boolVar):
    global biasLearning
    biasLearning = boolVar

def setPLearningRate(r):
    global pLearningRate
    pLearningRate = r
    
def setQLearningRate(r):
    global qLearningRate
    qLearningRate = r
    
def setBLearningRate(r):
    global bLearningRate
    bLearningRate = r
    
def setRegularization(regLambda):
    global regularization
    regularization = regLambda
    
def setNumOfFeatures(f):
    global numOfFeatures
    numOfFeatures = f
    
def setNumOfIterations(i):
    global numOfIterations
    numOfIterations = i
    
#def printSettingStatus():
#    print("Context type: ", contextType)
#    print("Current context variable: ", contextVariable)
#    print("Data source: ", dataSource)
#    print("Bias learning scheme: ", biasLearning)
#    print("User features learning rate: ", pLearningRate)
#    print("Item features learning rate: ", qLearningRate)
#    print("Bias learning rate: ", bLearningRate)
#    print("Regularization parameter: ", regularization)
#    print("Number of features: ", numOfFeatures)
#    print("Number of learning iterations: ", numOfIterations)
####################################################################################################
     
     
     
############################## GET RECOMMENDATIONS FUNCTIONS ########################################    
def getRating(clientName,uID,iID):
    #setContextType("no")
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt(clientName)
    return calculateRating(clientName,uID,iID)

def getRating_multiContext(clientName,uID,iID,contextValues):
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt_MultiContext(clientName)
    getMultipleContextBiasesFromTxt(clientName)
    return calculateRating_multiContext(clientName,uID,iID,contextValues)
    
def calculateRating(clientName,uID,iID):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    minRating = config.get('dataInfo', 'minrating')
    maxRating = config.get('dataInfo', 'maxrating')
    u= getGlobalBias()
    bi = getItemBias(iID)
    bu = getUserBias(uID)
    p = getUserFeatureVector(uID)
    q = getItemFeatureVector(iID)
    r=plainModel(u,bu,bi,p,q)
    return fixPredictedRating(r, float(minRating), float(maxRating))

def calculateRating_multiContext(clientName,uID,iID,contextValues):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    minRating = config.get('dataInfo', 'minrating')
    maxRating = config.get('dataInfo', 'maxrating')
    u= getGlobalBias()
    bi = getItemBias(iID)
    bu = getUserBias(uID)
    p = getUserFeatureVector(uID)
    q = getItemFeatureVector(iID)
    cntxtBiases= getCntxtBiases(uID,contextValues)
    r=multipleCntxtModel ( u, bu, bi, p, q, cntxtBiases)
    return fixPredictedRating(r, float(minRating), float(maxRating))

def getTopN(clientName,uID,n):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
      
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt(clientName)
   
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating(clientName,uID,iID)
    sortedRezMatItemRating=rezMatItemRating[rezMatItemRating[:,1].argsort()]
    top = sortedRezMatItemRating[:,0]
    top = list(top)
    top.reverse()
    # reverse in numpy's array
    #top = top[::-1]
    return top[:n]    
    
def getBottomN(clientName,uID,n):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
      
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt(clientName)
    
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating(clientName,uID,iID)
    sortedRezMatItemRating=rezMatItemRating[rezMatItemRating[:,1].argsort()]
    bottom = sortedRezMatItemRating[:,0]
    bottom = list(bottom)
    return bottom[:n]    
    
def getTopN_multiContext(clientName,uID,n,contextValues):
    global contextValPerVar
    
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
    
    contextValPerVar = config.get('dataInfo', 'contextvarsandvals')
    contextValPerVar=contextValPerVar[1:len(contextValPerVar)-1]
    contextValPerVar=contextValPerVar.split(', ')
    contextValPerVar=[int(i) for i in contextValPerVar]
    
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt_MultiContext(clientName)
    
    getMultipleContextBiasesFromTxt(clientName)
    
    
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating_multiContext(clientName,uID,iID,contextValues)
    sortedRezMatItemRating=rezMatItemRating[rezMatItemRating[:,1].argsort()]
    top = sortedRezMatItemRating[:,0]
    top = list(top)
    top.reverse()
    # reverse in numpy's array
    #top = top[::-1]
    return top[:n]    
    
def getBottomN_multiContext(clientName,uID,n,contextValues):
    global contextValPerVar
    
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
    
    contextValPerVar = config.get('dataInfo', 'contextvarsandvals')
    contextValPerVar=contextValPerVar[1:len(contextValPerVar)-1]
    contextValPerVar=contextValPerVar.split(', ')
    contextValPerVar=[int(i) for i in contextValPerVar]
    
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt_MultiContext(clientName)
    getMultipleContextBiasesFromTxt(clientName)
        
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating_multiContext(clientName,uID,iID,contextValues)
    sortedRezMatItemRating=rezMatItemRating[rezMatItemRating[:,1].argsort()]
    bottom = sortedRezMatItemRating[:,0]
    bottom = list(bottom)
    return bottom[:n]    

def getRandomItems (clientName, n):
    # initialize result list
    resultList = numpy.zeros(n)
    
    # get items' ids from conf File
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    
    # turn itemIDs string into list of integers
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
    
    # get random permutation
    rand=numpy.random.permutation(len(itemIDs))
    # select itemIDs based on random permutation    
    for i in range(n):
        resultList[i] = itemIDs[rand[i]]
    
    return resultList


def getDiverseN(clientName, initialSetIds, n):
    
    resultList = numpy.zeros(n)
    getFeaturesFromTxt(clientName)
      
    # read from configuration file  
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    
    # turn itemIDs string into list of integers
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
    
    # prepare the set of items other than initial  
    sourceSet = numpy.zeros([len(itemIDs),3])
    for i in range(len(itemIDs)):
        sourceSet[i,0]= itemIDs[i]
        sourceSet[i,1]= itemFeaturesMatrix[itemIDs[i],0]
        sourceSet[i,2]= itemFeaturesMatrix[itemIDs[i],1]
        
    for i in range(len(initialSetIds)):     
        sourceSet=numpy.delete(sourceSet, numpy.where(sourceSet[:,0]==initialSetIds[i])[0], 0)
        
    numOfSelected=0
    group = initialSetIds
    
    while numOfSelected < n:
        distances = numpy.zeros([len(group)])
        finalDistances = numpy.zeros([numpy.shape(sourceSet)[0]]) 
        for i in range(numpy.shape(sourceSet)[0]):
            for j in range(len(group)):
                x=numpy.array((itemFeaturesMatrix[group[j], 0:2]))
                y=numpy.array((sourceSet[i,1:3]))
                distances[j] = numpy.linalg.norm(x-y)
            
            finalDistances[i]= math.sqrt(sum(numpy.power(distances,2)))
        indexOfMax = numpy.argmax(finalDistances)
        
        selectedItemId = sourceSet[indexOfMax,0]
        
        group = numpy.append(group, selectedItemId)
        resultList[numOfSelected] = selectedItemId
        numOfSelected = numOfSelected + 1
        sourceSet=numpy.delete(sourceSet, indexOfMax,0)
    return resultList
    
    
def getSimilarN(clientName, initialItemId, n):
    # get features
    getFeaturesFromTxt(clientName)
      
    # read from configuration file  
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    
    # turn itemIDs string into list of integers
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
    
    # prepare the set of items other than initial  
    sourceSet = numpy.zeros([len(itemIDs),3])
    for i in range(len(itemIDs)):
        sourceSet[i,0]= itemIDs[i]
        sourceSet[i,1]= itemFeaturesMatrix[itemIDs[i],0]
        sourceSet[i,2]= itemFeaturesMatrix[itemIDs[i],1]
    sourceSet=numpy.delete(sourceSet, numpy.where(sourceSet[:,0]==initialItemId)[0], 0)
        
    # prepare the matrix for distances
    distances = numpy.zeros([numpy.shape(sourceSet)[0],2])
    
    # get coordinates of input item
    initialItemCoords = itemFeaturesMatrix[initialItemId, 0:2]
    
    # calculate all distances
    for i in range(len(sourceSet)):
        x=numpy.array((sourceSet[i,1:3]))
        y=numpy.array((initialItemCoords))
        distances[i,1] = numpy.linalg.norm(x-y)
        distances[i,0] = sourceSet[i,0]
            
    #sort distances and take n smallest    
    distances=distances[distances[:,1].argsort()]
    resultList = distances[0:n,0]
    
    return resultList
####################################################################################################



######################################## CONTROL FUNCTIONS #########################################
#def updateLearnedDB():
#def learn():
####################################################################################################



############################## FETCH FROM DATABASE FUNCTIONS ########################################
def getBiasesFromTxt(clientName):
    global globalBiasMatrix
    global userBiasesMatrix
    global itemBiasesMatrix
    globalBiasFilename = mainResultPath + '/' + clientName + '_' + 'globalBias.txt'
    userBiasFilename = mainResultPath + '/' + clientName + '_' + 'userBias.txt'
    itemBiasFilename = mainResultPath + '/' + clientName + '_' + 'itemBias.txt'
    globalBiasMatrix = numpy.loadtxt(globalBiasFilename, delimiter=';')
    userBiasesMatrix = numpy.loadtxt(userBiasFilename, delimiter=';')
    itemBiasesMatrix = numpy.loadtxt(itemBiasFilename, delimiter=';')
        
def getFeaturesFromTxt(clientName):
    global userFeaturesMatrix
    global itemFeaturesMatrix
    userFeaturesFilename = mainResultPath + '/' + clientName + '_' + 'userFeatures.txt'
    itemFeaturesFilename = mainResultPath + '/' + clientName + '_' + 'itemFeatures.txt'
    userFeaturesMatrix = numpy.loadtxt(userFeaturesFilename, delimiter=';')
    itemFeaturesMatrix = numpy.loadtxt(itemFeaturesFilename, delimiter=';')
    
def getFeaturesFromTxt_MultiContext(clientName):
    global userFeaturesMatrix
    global itemFeaturesMatrix
    
    userFeaturesMultipleFilename = mainResultPath + '/' + clientName + '_' + 'userFeaturesMultiple.txt'
    itemFeaturesMultipleFilename = mainResultPath + '/' + clientName + '_' + 'itemFeaturesMultiple.txt'
    userFeaturesMatrix = numpy.loadtxt(userFeaturesMultipleFilename, delimiter=';')
    itemFeaturesMatrix = numpy.loadtxt(itemFeaturesMultipleFilename, delimiter=';')
    
def getMultipleContextBiasesFromTxt(clientName):
    global multipleContextUserBiasesMatrix
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    
    contextValPerVar = config.get('dataInfo', 'contextvarsandvals')
    contextValPerVar=contextValPerVar[1:len(contextValPerVar)-1]
    contextValPerVar=contextValPerVar.split(', ')
    contextValPerVar=[int(i) for i in contextValPerVar]
    
    
    userBiasesMultipleFilename = mainResultPath + '/' + clientName + '_' + 'userBias.txt'
    userBiases = numpy.loadtxt(userBiasesMultipleFilename, delimiter=';')
    multipleContextUserBiasesMatrix=numpy.zeros([len(contextValPerVar),numpy.max(userBiases)+1,max(contextValPerVar)])
    
    for i in range(len(contextValPerVar)):
        filename = mainResultPath + '/' + clientName + '_' + 'userBiasPerContext' + str(i) + '.txt'
        multipleContextUserBiasesMatrix[i,:,:] = numpy.loadtxt(filename, delimiter=';')
    
def getGlobalBias():
    data=globalBiasMatrix
    return data

def getUserBias(userID):
    data=userBiasesMatrix
    usrIndexList = list(data[:,0])
    index = usrIndexList.index (userID)
    return data[index,1]

def getItemBias(itemID):
    data=itemBiasesMatrix
    itmIndexList = list(data[:,0])
    index = itmIndexList.index (itemID)
    return data[index,1]

def getUserFeatureVector(userID):
    data=userFeaturesMatrix
    return data[userID,1:]

def getItemFeatureVector(itemID):
    data=itemFeaturesMatrix
    return data[itemID,1:]

def getContextInformation(data):
    global contextValPerVar
    contextValPerVar = numpy.zeros(data.shape[1]-3)
    #for loop for every context variable in the dataset
    for i in range(data.shape[1]-3):
        values=(numpy.unique(data[:,3+i]))
        if values[0] == 0:
            values = numpy.delete(values,0)
        contextValPerVar[i]=len(values)
        
def getCntxtBiases(userID,contextValues):
    cntxtBiases = numpy.zeros(len(contextValues))
    
    for i in range(len(contextValues)):
        if contextValues[i]==0:
            cntxtBiases[i] = 0
            
        else:
            cntxtBiases[i]=multipleContextUserBiasesMatrix[i,userID,int(contextValues[i])-1]
    
    return cntxtBiases    
####################################################################################################



######################################## MODELING FUNCTIONS ########################################
def trainPlainModel(clientName):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    datasetName = config.get('dataPath', 'datasetName')
    
    dataSet=numpy.loadtxt(datasetName, delimiter=';')
           
    maxUsrID = numpy.amax(dataSet[:,0])
    maxItmID = numpy.amax(dataSet[:,1])
     
    pMatrix = numpy.zeros([maxUsrID+1,numOfFeatures+1])
    qMatrix = numpy.zeros([maxItmID+1,numOfFeatures+1])
    
    print("calculating static biases")
    calculateStaticBiases(dataSet,clientName)
    getBiasesFromTxt(clientName)
    
    print ("started training...")    
    for f in range(numOfFeatures):
        print("Training feature: ", f+1, "/", numOfFeatures )
        pMatrix[:,f]=initFeatureValue
        qMatrix[:,f]=initFeatureValue
        for _ in range(numOfIterations):
            for i in range(dataSet.shape[0]):
                userID = dataSet[i,0]
                itemID = dataSet[i,1]
                trueRating = dataSet[i,2]
                estimatedRating = plainModel (getGlobalBias(), getUserBias(userID), getItemBias(itemID), pMatrix[userID,:], qMatrix[itemID,:])
                error = trueRating - estimatedRating
                            
                tempUF = pMatrix[userID,f];
                tempIF = qMatrix[itemID,f];
                        
                pMatrix[userID,f] = tempUF + (error * tempIF - regularization * tempUF) * pLearningRate
                qMatrix[itemID,f] = tempIF + (error * tempUF - regularization * tempIF) * qLearningRate
    
    
    userFeaturesResultFilename = mainResultPath + '/' + clientName + '_' + 'userFeatures.txt'
    itemFeaturesResultFilename = mainResultPath + '/' + clientName + '_' + 'itemFeatures.txt'       
    numpy.savetxt(userFeaturesResultFilename,pMatrix,delimiter=';')
    numpy.savetxt(itemFeaturesResultFilename,qMatrix,delimiter=';')
     

def trainMultipleCntxtModel(clientName):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    datasetName = config.get('dataPath', 'datasetName')
    dataSet=numpy.loadtxt(datasetName, delimiter=';')
    
    getContextInformation(dataSet)
    maxUsrID = numpy.amax(dataSet[:,0])
    maxItmID = numpy.amax(dataSet[:,1])
    
    pMatrix = numpy.zeros([maxUsrID+1,numOfFeatures+1])
    qMatrix = numpy.zeros([maxItmID+1,numOfFeatures+1])
    contextValues = numpy.zeros([len(contextValPerVar)])
    
    print("calculating static biases")
    calculateStaticBiases(dataSet,clientName)
    getBiasesFromTxt(clientName)
    calculateMultipleContextStaticBiases(dataSet,clientName)
    getMultipleContextBiasesFromTxt(clientName)
      
    print ("started training...")    
    for f in range(numOfFeatures):
        print("Training feature: ", f+1, "/", numOfFeatures )
        pMatrix[:,f]=initFeatureValue
        qMatrix[:,f]=initFeatureValue
        for _ in range(numOfIterations):
            for i in range(dataSet.shape[0]):
                userID = dataSet[i,0]
                itemID = dataSet[i,1]
                trueRating = dataSet[i,2]
                for c in range(len(contextValPerVar)):
                    contextValues[c] =  dataSet[i,3+c]
                    
                cntxtBiases= getCntxtBiases(userID,contextValues)
                
                estimatedRating = multipleCntxtModel (getGlobalBias(), getUserBias(userID), getItemBias(itemID), pMatrix[userID,:], qMatrix[itemID,:],cntxtBiases)
                error = trueRating - estimatedRating
                            
                tempUF = pMatrix[userID,f];
                tempIF = qMatrix[itemID,f];
                        
                pMatrix[userID,f] = tempUF + (error * tempIF - regularization * tempUF) * pLearningRate
                qMatrix[itemID,f] = tempIF + (error * tempUF - regularization * tempIF) * qLearningRate
    
    userFeaturesResultFilename = mainResultPath + '/' + clientName + '_' + 'userFeaturesMultiple.txt'
    itemFeaturesResultFilename = mainResultPath + '/' + clientName + '_' + 'itemFeaturesMultiple.txt'               
    numpy.savetxt(userFeaturesResultFilename,pMatrix,delimiter=';')
    numpy.savetxt(itemFeaturesResultFilename,qMatrix,delimiter=';')         
        
def validatePlainModel(clientName):
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt(clientName)
    
    testSet=numpy.loadtxt(validationDataSource, delimiter=';')
    ratingsDifferences = numpy.zeros([testSet.shape[0],1]) 
    for i in range(testSet.shape[0]):   
        userID = testSet[i,0]
        itemID = testSet[i,1]
        trueRating = testSet[i,2]
        estimatedRating=calculateRating(clientName,userID,itemID)
        ratingsDifferences[i] = trueRating - estimatedRating
    
    RMSE = numpy.sum(numpy.square(ratingsDifferences))
    RMSE = numpy.sqrt(RMSE/(testSet.shape[0]))
    return RMSE    

def validateMultipleCntxtModel(clientName):
    getBiasesFromTxt(clientName)
    getFeaturesFromTxt_MultiContext(clientName)
    testSet=numpy.loadtxt(validationDataSource, delimiter=';')
    getContextInformation(testSet)
    getMultipleContextBiasesFromTxt(clientName)
    
    contextValues = numpy.zeros([len(contextValPerVar)])
           
    ratingsDifferences = numpy.zeros([testSet.shape[0],1]) 
    for i in range(testSet.shape[0]):   
        userID = testSet[i,0]
        itemID = testSet[i,1]
        trueRating = testSet[i,2]
        
        for c in range(len(contextValPerVar)):
            contextValues[c] =  testSet[i,3+c]
                               
        estimatedRating=calculateRating_multiContext(clientName,userID,itemID,contextValues)
        ratingsDifferences[i] = trueRating - estimatedRating
    
    RMSE = numpy.sum(numpy.square(ratingsDifferences))
    RMSE = numpy.sqrt(RMSE/(testSet.shape[0]))
    return RMSE    

def calculateStaticBiases (data,clientName):
    ## calculate global bias
    # calculate mean of all the ratings
    globalBias = numpy.zeros([1,1])
    globalBias[0,0] = numpy.mean(data[:,2])
    
    globalBiasFileName = mainResultPath + '/' + clientName + '_' + 'globalBias.txt'
    numpy.savetxt(globalBiasFileName,globalBias,delimiter=';')
    
    # prepare the users' biases matrix    
    numberOfUsers = len(numpy.unique(data[:,0]))
    userMatIdBias = numpy.zeros([numberOfUsers, 2])
    
    # prepare the items' biases matrix
    numberOfItems = len(numpy.unique(data[:,1]))
    itemMatIdBias = numpy.zeros([numberOfItems, 2])
        
    ## calculate user bias    
    # calculate mean rating for the user in the condition
    for i, usrIndex in enumerate(numpy.unique(data[:,0])):
        condition = data[:,0]==usrIndex
        f=numpy.mean(numpy.extract(condition, data[:,2]))
        f=f-globalBias[0,0]
        userMatIdBias[i,:] = [usrIndex,f]
        
        userBiasesFilename = mainResultPath + '/' + clientName + '_' + 'userBias.txt'
        numpy.savetxt(userBiasesFilename,userMatIdBias,delimiter=';')
       
    ## calculate item bias 
    # calculate mean rating for the item in the condition
    for i, itmIndex in enumerate(numpy.unique(data[:,1])):
        condition = data[:,1]==itmIndex
        f=numpy.mean(numpy.extract(condition, data[:,2]))
        f = f-globalBias[0,0]
        itemMatIdBias[i,:] = [itmIndex,f]
        
        itemBiasesFilename = mainResultPath + '/' + clientName + '_' + 'itemBias.txt'
        numpy.savetxt(itemBiasesFilename,itemMatIdBias,delimiter=';')

def calculateMultipleContextStaticBiases(data, clientName):
    global contextValPerVar
    userIDs = numpy.unique(data[:,0])
    userIDs = [int(i) for i in userIDs]
    userBiasesMatrix=numpy.zeros([len(contextValPerVar),max(userIDs)+1,max(contextValPerVar)])
    for i in range(len(contextValPerVar)):
        for j in range(int(contextValPerVar[i])):
            for k in userIDs:
                condition1 = data[:,0]==k
                condition2 = data[:,3+i]==j
                condition3 = condition1&condition2
                
                if sum(condition3)==0:
                    userBiasesMatrix[i,k,j]=getUserBias(k)
                else:
                    mean = numpy.mean(numpy.extract(condition3, data[:,2]))
                    userBiasesMatrix[i,k,j]=mean - getUserBias(k)- getGlobalBias()
                
    for i in range(len(contextValPerVar)):
        filename = mainResultPath + '/' + clientName + '_' + 'userBiasPerContext' + str(i) + '.txt'
        numpy.savetxt(filename,userBiasesMatrix[i,:,:],delimiter=';')   
                  
def fixPredictedRating (predictedR, minRating, maxRating ):
    if predictedR > maxRating:
        predictedR = maxRating
    elif predictedR < minRating:
        predictedR = minRating
    else:
        predictedR = predictedR
    return predictedR   
    
def multipleCntxtModel ( u, bu, bi, p, q, cntxtBiases):
    pq = numpy.matrix(p)*numpy.matrix(q).T
    cntxtBiasSum = numpy.sum(cntxtBiases)
    return u + bu + bi + pq + cntxtBiasSum

def plainModel ( u, bu, bi, p, q ):
    pq = numpy.matrix(p)*numpy.matrix(q).T
    return u + bu + bi + pq
####################################################################################################
#validate_MF('movieat')
#initializeClientConfFile('comoda', 'no')