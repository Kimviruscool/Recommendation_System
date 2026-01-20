import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings[['user_id','movie_id','rating']].astype(int) #timestamp[ 제거


#MF class
class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
            #파라미터 값 클래스로 저장
            self.R = np.array(ratings)
            self.num_users, self.num_items = np.shape(self.R)
            self.K = K
            self.alpha = alpha
            self.beta = beta
            self.iterations = iterations
            self.verbose = verbose

#Root Mean Squared Error (RMSE) 계산
def rmse(self):
    xs, ys = self.R.nonzero()
    self.predictions = []
    self.errors = []
    for x, y in zip(xs, ys):
        prediction = self.predict(x,y)
        self.predictions.append(prediction)
        self.errors.append(self.R[x,y]-prediction)
    self.predictions = np.array(self.predictions)
    self.errors = np.array(self.errors)
    return np.sqrt(np.mean(self.errors**2))