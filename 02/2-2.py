#사용자 집단별 추천

import pandas as pd
import numpy as np

u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.user',sep='|',names=u_cols,encoding='latin-1')

i_cols = ['movie_id','title','release date','video release date','IMDB URL','unknown','Action','Adventure','Animation','Children\s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.item',sep='|',names=i_cols,encoding='latin-1')

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.data',sep='\t',names=r_cols,encoding='latin-1')

# timestamp 제거
ratings = ratings.drop('timestamp', axis=1)

#movie_id , title만 남기기
movies = movies[['movie_id','title']]

#train test set 분리
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y)

#정확도 계산
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

def score(model):
    id_pairs = zip(x_test['user_id'],x_test['movie_id'])
    y_pred = np.array([model(user,movie) for (user, movie) in id_pairs])
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)

rating_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')
# print(rating_matrix) full matrix 변환됨

# 전체 평균으로 예측치를 계산하는 기본모델
def best_seller(user_id,movie_id):
    try :
        rating = train_mean[movie_id]
    except :
        rating = 3.0
    return rating

train_mean = x_train.groupby(['movie_id'])['rating'].mean()
score(best_seller)
print(score(best_seller))

#Full matrix 를 사용자 데이터와 merge
merged_rating = pd.merge(x_train, users)
users = users.set_index('user_id')

#gender별 평점 평균 계산
g_mean = merged_rating[['movie_id','sex','rating']].groupby(['movie_id','sex'])['rating'].mean()

#gender기준 추천
#gender별 평균을 예측치로 돌려주는 함수
def cf_gender(user_id, movie_id):
    if movie_id in rating_matrix:
        gender = users.loc[user_id]['sex']
        if gender in g_mean[movie_id]:
            gender_rating = g_mean[movie_id][gender]
        else :
            gender_rating = 3.0
    else:
        gender_rating = 3.0
    return gender_rating

print(score(cf_gender))