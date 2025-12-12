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
movies = movies.set_index('movie_id')

#train test set 분리
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y)

#정확도 계산
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

def score(model, neighbor_size = 0):
    id_pairs = zip(x_test['user_id'],x_test['movie_id'])
    y_pred = np.array([model(user,movie,neighbor_size) for (user, movie) in id_pairs])
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)

rating_matrix = x_train.pivot_table(index='user_id', columns='movie_id', values='rating')
# print(rating_matrix) full matrix 변환됨

from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

def CF_simple(user_id, movie_id):
    if movie_id in rating_matrix:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_matrix[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.dropna()
        sim_scores = sim_scores.drop(none_rating_idx)
        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
    else :
        mean_rating = 3.0
    return mean_rating

def cf_knn(user_id, movie_id, neighbor_size=0):
    if movie_id in rating_matrix :
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_matrix[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)

        if neighbor_size == 0:
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        else :
            if len(sim_scores) > 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            else : mean_rating = 3.0
    else :
        mean_rating = 3.0
    return mean_rating

print(score(cf_knn, neighbor_size=30))

def recommender(user, n_items=10, neighbor_size=20):
    predictions = []
    rated_index = rating_matrix.loc[user][rating_matrix.loc[user] > 0].index
    items = rating_matrix.loc[user].drop(rated_index)
    for item in items.index:
        predictions.append(cf_knn(user, item, neighbor_size))
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items]
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

print(recommender(user=2, n_items=5, neighbor_size=30))

rating_matrix = x_train.pivot_table(values='rating', index='user_id', columns='movie_id')
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
for neighbor_size in [10,20,30,40,50,60] :
    print("Neightbor size = %d : RMSE = %.4f" % (neighbor_size, score(cf_knn, neighbor_size)))