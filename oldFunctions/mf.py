'''
Created on Nov 6, 2012

@author: Ante Odic
'''
import numpy
import math
import configparser
from bottle import route, run, post, request
from bottle import default_app
############################## GLOBAL VARIABLES WITH DEFAULT VALUES ##############################

# mf construction parameters  
contextType = "multiple"
contextVariable = 1

# data paths
mainDataPath = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/data'
mainResultPath = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results'
dataSource = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/data/LDOScontextTRAINnoLAST.txt' #make "empty" default, this one is just for testing
validationDataSource = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/data/LDOScontextTEST.txt'
userBiasesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userBias.txt'
itemBiasesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemBias.txt'
globalBiasResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/globalBias.txt'
userFeaturesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userFeatures.txt'
itemFeaturesResult = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemFeatures.txt'
userFeaturesResultMultiple = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/userFeaturesMultiple.txt'
itemFeaturesResultMultiple = 'D:/00xBeds/17-TheRecommenderSystem/workspace/The RecommenderProject/results/itemFeaturesMultiple.txt'

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



######################################## CLIENT FUNCTIONS ##########################################
@post('/recsys')
def getRequestForRecSys():
		
	functionName = 	request.forms.get('fName')
	
	if str(request.forms.get('context')) == 'None':
		context = (0,0)
	
	if functionName == 'getTopN_MF': rez = getTopN_MF(request.forms.get('clientID'),int(request.forms.get('userID')),int(request.forms.get('N')), context)
	elif functionName == 'getRating_MF': rez = getRating_MF(request.forms.get('clientID'),int(request.forms.get('userID')), int(request.forms.get('itemID')),context)
	elif functionName == 'getBottomN_MF': rez = getBottomN_MF(request.forms.get('clientID'),int(request.forms.get('userID')),int(request.forms.get('N')), context)
	
	else: rez = 'No function!!'
	
	return str(rez)	


def getRating_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=itemID, [3]=context
    confFileName = mainDataPath + '/' + arg[0] + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return getRating(arg[1],arg[2])
    elif cType == 'multiple':
        return getRating_multiContext(arg[1],arg[2],arg[3])
        
 
def getTopN_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=N, [3]=context
    confFileName = mainDataPath + '/' + arg[0] + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    cType = config.get('modelInfo', 'contexttype')
    if cType == 'no':
        return getTopN(arg[0],arg[1],arg[2])
    elif cType == 'multiple':
        return getTopN_multiContext(arg[0],arg[1],arg[2],arg[3])
		

def getBottomN_MF(*arg):#agr:[0] = clientID, [1]=userID, [2]=N, [3]=context
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
    


####################################################################################################


######################################## CLIENT CONFIGURATION ##################################################

def initializeClientConfFile (clientName, contextType):
    
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    datasetName = config.get('dataPath', 'datasetName')
    
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
def setContextType(cType):
    global contextType
    contextType = cType
    
def setContextVariable(variable):
    global contextVariable
    contextVariable = variable 
    
def setDataSource(filename):
    global dataSource
    dataSource = filename
    
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
    
def printSettingStatus():
    print("Context type: ", contextType)
    print("Current context variable: ", contextVariable)
    print("Data source: ", dataSource)
    print("Bias learning scheme: ", biasLearning)
    print("User features learning rate: ", pLearningRate)
    print("Item features learning rate: ", qLearningRate)
    print("Bias learning rate: ", bLearningRate)
    print("Regularization parameter: ", regularization)
    print("Number of features: ", numOfFeatures)
    print("Number of learning iterations: ", numOfIterations)
####################################################################################################
     
     
     
############################## GET RECOMMENDATIONS FUNCTIONS ########################################    
def getRating(uID,iID):
    setContextType("no")
    getBiasesFromTxt()
    getFeaturesFromTxt()
    return calculateRating(uID,iID)

def getRating_multiContext(uID,iID,contextValues):
    data=numpy.loadtxt(dataSource, delimiter=';')
    getBiasesFromTxt()
    getFeaturesFromTxt_MultiContext()
    getContextInformation(data)
    getMultipleContextBiasesFromTxt()
    return calculateRating_multiContext(uID,iID,contextValues)
    
def calculateRating(uID,iID):
    u= getGlobalBias()
    bi = getItemBias(iID)
    bu = getUserBias(uID)
    p = getUserFeatureVector(uID)
    q = getItemFeatureVector(iID)
    r=plainModel(u,bu,bi,p,q)
    return fixPredictedRating(r,1,5)

def calculateRating_multiContext(uID,iID,contextValues):
    u= getGlobalBias()
    bi = getItemBias(iID)
    bu = getUserBias(uID)
    p = getUserFeatureVector(uID)
    q = getItemFeatureVector(iID)
    cntxtBiases= getCntxtBiases(uID,contextValues)
    r=multipleCntxtModel ( u, bu, bi, p, q, cntxtBiases)
    return fixPredictedRating(r,1,5)

def getTopN(clientName,uID,n):
    confFileName = mainDataPath + '/' + clientName + '.cfg'
    config = configparser.RawConfigParser()
    config.read(confFileName)
    itemIDs = config.get('dataInfo', 'itemids')
    itemIDs=itemIDs[1:len(itemIDs)-1]
    itemIDs=itemIDs.split(', ')
    itemIDs=[int(i) for i in itemIDs]
      
    getBiasesFromTxt()
    getFeaturesFromTxt()
   
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating(uID,iID)
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
      
    getBiasesFromTxt()
    getFeaturesFromTxt()
    
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating(uID,iID)
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
    
    getBiasesFromTxt()
    getFeaturesFromTxt_MultiContext()
    
    getMultipleContextBiasesFromTxt()
    
    
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating_multiContext(uID,iID,contextValues)
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
    
    getBiasesFromTxt()
    getFeaturesFromTxt_MultiContext()
    getMultipleContextBiasesFromTxt()
        
    rezMatItemRating = numpy.zeros([len(itemIDs),2])
    for i, iID in enumerate(itemIDs):
        rezMatItemRating[i,0] = iID
        rezMatItemRating[i,1] = calculateRating_multiContext(uID,iID,contextValues)
    sortedRezMatItemRating=rezMatItemRating[rezMatItemRating[:,1].argsort()]
    bottom = sortedRezMatItemRating[:,0]
    bottom = list(bottom)
    return bottom[:n]    


#def getDiverseN(uID,iID,n,radius):
#def getSimilarN(uID,iID,n,radius):
####################################################################################################



######################################## CONTROL FUNCTIONS #########################################
#def updateLearnedDB():
#def learn():
####################################################################################################



############################## FETCH FROM DATABASE FUNCTIONS ########################################
def getBiasesFromTxt():
    global globalBiasMatrix
    global userBiasesMatrix
    global itemBiasesMatrix
    globalBiasMatrix = numpy.loadtxt(globalBiasResult, delimiter=';')
    userBiasesMatrix = numpy.loadtxt(userBiasesResult, delimiter=';')
    itemBiasesMatrix = numpy.loadtxt(itemBiasesResult, delimiter=';')
        
def getFeaturesFromTxt():
    global userFeaturesMatrix
    global itemFeaturesMatrix
    userFeaturesMatrix = numpy.loadtxt(userFeaturesResult, delimiter=';')
    itemFeaturesMatrix = numpy.loadtxt(itemFeaturesResult, delimiter=';')
    
def getFeaturesFromTxt_MultiContext():
    global userFeaturesMatrix
    global itemFeaturesMatrix
    userFeaturesMatrix = numpy.loadtxt(userFeaturesResultMultiple, delimiter=';')
    itemFeaturesMatrix = numpy.loadtxt(itemFeaturesResultMultiple, delimiter=';')
    
def getMultipleContextBiasesFromTxt():
    global multipleContextUserBiasesMatrix
    userBiases = numpy.loadtxt(userBiasesResult, delimiter=';')
    multipleContextUserBiasesMatrix=numpy.zeros([len(contextValPerVar),numpy.max(userBiases)+1,max(contextValPerVar)])
    
    for i in range(len(contextValPerVar)):
        filename = mainResultPath + '/userBiasPerContext' + str(i) + '.txt'
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
    usrIndexList = list(data[:,0])
    index = usrIndexList.index (itemID)
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
    calculateStaticBiases(dataSet)
    getBiasesFromTxt()
    
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
            
    numpy.savetxt(userFeaturesResult,pMatrix,delimiter=';')
    numpy.savetxt(itemFeaturesResult,qMatrix,delimiter=';')
        
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
    calculateStaticBiases(dataSet)
    getBiasesFromTxt()
    calculateMultipleContextStaticBiases(dataSet)
    getMultipleContextBiasesFromTxt()
      
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
            
    numpy.savetxt(userFeaturesResultMultiple,pMatrix,delimiter=';')
    numpy.savetxt(itemFeaturesResultMultiple,qMatrix,delimiter=';')        
        
def validatePlainModel():
    getBiasesFromTxt()
    getFeaturesFromTxt()
    
    testSet=numpy.loadtxt(validationDataSource, delimiter=';')
    ratingsDifferences = numpy.zeros([testSet.shape[0],1]) 
    for i in range(testSet.shape[0]):   
        userID = testSet[i,0]
        itemID = testSet[i,1]
        trueRating = testSet[i,2]
        estimatedRating=calculateRating(userID,itemID)
        ratingsDifferences[i] = trueRating - estimatedRating
    
    RMSE = numpy.sum(numpy.square(ratingsDifferences))
    RMSE = numpy.sqrt(RMSE/(testSet.shape[0]))
    return RMSE    

def validateMultipleCntxtModel():
    getBiasesFromTxt()
    getFeaturesFromTxt_MultiContext()
    testSet=numpy.loadtxt(validationDataSource, delimiter=';')
    getContextInformation(testSet)
    getMultipleContextBiasesFromTxt()
    
    contextValues = numpy.zeros([len(contextValPerVar)])
           
    ratingsDifferences = numpy.zeros([testSet.shape[0],1]) 
    for i in range(testSet.shape[0]):   
        userID = testSet[i,0]
        itemID = testSet[i,1]
        trueRating = testSet[i,2]
        
        for c in range(len(contextValPerVar)):
            contextValues[c] =  testSet[i,3+c]
                               
        estimatedRating=calculateRating_multiContext(userID,itemID,contextValues)
        ratingsDifferences[i] = trueRating - estimatedRating
    
    RMSE = numpy.sum(numpy.square(ratingsDifferences))
    RMSE = numpy.sqrt(RMSE/(testSet.shape[0]))
    return RMSE    

def calculateStaticBiases (data):
    ## calculate global bias
    # calculate mean of all the ratings
    globalBias = numpy.zeros([1,1])
    globalBias[0,0] = numpy.mean(data[:,2])
    numpy.savetxt(globalBiasResult,globalBias,delimiter=';')
    
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
        numpy.savetxt(userBiasesResult,userMatIdBias,delimiter=';')
       
    ## calculate item bias 
    # calculate mean rating for the item in the condition
    for i, itmIndex in enumerate(numpy.unique(data[:,1])):
        condition = data[:,1]==itmIndex
        f=numpy.mean(numpy.extract(condition, data[:,2]))
        f = f-globalBias[0,0]
        itemMatIdBias[i,:] = [itmIndex,f]
        numpy.savetxt(itemBiasesResult,itemMatIdBias,delimiter=';')

def calculateMultipleContextStaticBiases(data):
    global contextValPerVar
    userIDs = numpy.unique(data[:,0])
    userBiasesMatrix=numpy.zeros([len(contextValPerVar),numpy.max(userIDs)+1,max(contextValPerVar)])
    for i in range(len(contextValPerVar)):
        for j in range(int(contextValPerVar[i])):
            for k in userIDs:
                condition1 = data[:,0]==k
                condition2 = data[:,3+i]==j
                condition3 = condition1&condition2
                mean = numpy.mean(numpy.extract(condition3, data[:,2]))
                if math.isnan(mean):
                    userBiasesMatrix[i,k,j]=getUserBias(k)
                else:
                    userBiasesMatrix[i,k,j]=mean - getUserBias(k)- getGlobalBias()
                
    for i in range(len(contextValPerVar)):
        filename = mainResultPath + '/userBiasPerContext' + str(i) + '.txt'
        numpy.savetxt(filename,userBiasesMatrix[i,:,:],delimiter=';')  
                  
def fixPredictedRating ( predictedR, minRating, maxRating ):
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
