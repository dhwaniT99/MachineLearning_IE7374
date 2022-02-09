from shutil import SpecialFileError
class LinearRegression:
  def __init__(self, X, y, lr, tol, maxIter, gd = False) -> None:   #Init is the part where we initialise the parameters ## self means global means these parameters are accessible from outside functions  
      self.X = X
      self.y = y
      self.lr = lr
      self.tol = tol
      self.maxIter = maxIter
      self.gd = gd

  #First step when we get input divide into train and test split
  def trainTestSplit(self):
    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                         test_size = 0.3, random_state= 0)

    return X_train, X_test, y_train, y_test
  
  def add_X0(self, X):
    return np.column_stack([np.ones([X.shape[0], 1]),X]) 

  

  def normalize(self, X):
    mean = np.mean(X, 0)
    sd = np.std(X, 0)
    X_norm = (X - mean)/sd

    X_norm = self.add_X0(X_norm) #it will recieve x norm
    return X_norm, mean, sd

  def normalizeTestData(self, X, trainMean, trainSd):
    X_norm = (X - trainMean)/trainSd
    X_norm = self.add_X0(X_norm)
    return X_norm #We need closed form solution we need invertable matrix, which should be full rank and should not be invertible



  def rank(self, X, eps = 1e-12):
    #Singular value decomposition
    u , s, vh = np.linalg.svd(X)
    return len([x for x in s if abs(x) >eps]) #Returns the number of non-zeros eigen values, 

  def checkMatrix(self, X):
    X_rank = np.linalg.matrix_rank(X)

    if X_rank == min(X.shape[0], X.shape[1]):   #Shape[0] is number of coloumns and shape[1] is number of full ranks
      self.fullRank = True
      print("Data is Full Rank")
    else:
      self.fullRank = False
      print("Data is not full Rank")
  #when m <d then the matrix will be lower rank

  def checkInvertibility(self, X):
    if X.shape[0] < X.shape[1]:
      self.lowRank = True
      print("Data is low rank")
    else: 
      self.lowRank = False
      print("Data is not low rank")

  def closeFormSolution(self, X, y):
    w = np.linalg.inv(X.T.dot(X.dot(X.T).dot(y))) 
    return w 
    
  def gradientDescent(self, X, y):
    errorSequence = []
    last = float('inf')

    for t in tqdm(range(self, maxIter)):
      self.w = self.w - self.lr * self.costDerivative(X,y)
      cuerr = self.sse(X,y)
      errorSequence.append(cuerr)
      diff = last - cuerr
      last = cuerr

      if diff < self.tol:
        print("The model has stopped no further improvement")
        break

    #Write a for loop   #Normal Equation is closed form solution

  def sse(self, X, y):
    y_hat = self.predict(X)
    return ((y_hat - y) ** 2).sum()

  def predict(self, X):
    return X.dot(self.w)

  def costFunction(self, X, y):
    return self.sse(X,y)/2

  def costDerivative(self, X, y):
    y_hat = self.predict(X)
    return((y_hat - y ) **2 ).sum()

  def fit(self):
    self.X_train, self.X_test, self.y_train, self.y_test, self.trainTestSplit()
    self.X_train, self.mean, self.sd = self.normalize(self.X_train)
    self.X_test = self.normalizeTestData(self.X_test)
    self.checkMatrix(self.X_train)
    self.checkInvertibility(self.X_train)
    if self.fullRank and not self.lowRank and not self.gd:
      print("solving the full rank matrix")
      self.w = self.closedFormSolution(self.X_train, self.y_train)
    else:
      print("Solving using Gradient descent")
      self = np.ones(self.X_train.shape[1], dtype = np.float64) * 0 
      self.gradient = gradientDescent(self.X_train, self.y_train)

    print(self.w)
