import pandas as pd
import numpy as np

#user data
u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.user',sep='|',names=u_cols,encoding='latin-1')
# print(users)
users = users.set_index('user_id')
# print(users)
# print(users.head())

i_cols = ['movie_id','title','release date','video release date','IMDB URL','unknown','Action','Adventure','Animation','Children\s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.item',sep='|',names=i_cols,encoding='latin-1')
# print(movies)
movies = movies.set_index('movie_id')
# print(movies)
# print(movies.head())

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('C:/Users/0000/Desktop/Recommendation_System/02/u.data',sep='\t',names=r_cols,encoding='latin-1')
ratings = ratings.set_index('user_id')
# print(ratings.head())

#인기제품 방식
def recom_movie1(n_items):
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    print(recommendations)
    return recommendations

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
recom_movie1(5)

def recom_movie2(n_items):
    return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]['title']

# recom_movie2(5)
print(recom_movie2(5))

def RMSE(y_true,t_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(t_pred))**2))

rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user]['rating']
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true,y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))