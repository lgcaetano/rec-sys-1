import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math

start = datetime.now()

ratings = pd.read_csv('ratings.csv')
targets = pd.read_csv('targets.csv')

def split_user_item_ids(df):
    split_ids = df['UserId:ItemId'].apply(lambda x: x.split(':'))
    user_ids = [id[0] for id in split_ids]
    item_ids = [id[1] for id in split_ids]
    df['UserId'] = user_ids
    df['ItemId'] = item_ids

split_user_item_ids(ratings)
split_user_item_ids(targets)

print()

users = ratings['UserId'].unique()

items = ratings['ItemId'].unique()

n_items = len(items)
n_users = len(users)

k = 10 # latent factors

learning_rate = 0.01

lambda_reg = 0.1

epochs = 1

batch_size = 64

rating_mean = ratings['Rating'].mean()

initial_mean = math.sqrt(rating_mean / 10) 

users_initials = np.full((n_users, k), initial_mean)

items_initials = np.full((n_items, k), initial_mean)


u_latent = pd.DataFrame(users_initials, index=ratings['UserId'].unique())
i_latent = pd.DataFrame(items_initials, index=ratings['ItemId'].unique())


def predict_with_sgd(user_id, item_id):
    return np.dot(u_latent.loc[user_id], i_latent.loc[item_id])

cur_perc = 0

for i in range(epochs):
    for i in range(0, len(ratings), batch_size):
        mini_batch = ratings.iloc[i:i + batch_size]
        print(f'{(i / (len(ratings) * epochs)) * 100}%')        
        # Compute errors and gradients for the entire mini-batch using vectorized operations
        errors = mini_batch.apply(lambda row: row['Rating'] - predict_with_sgd(row['UserId'], row['ItemId']), axis=1)
        user_reg = 2 * lambda_reg * u_latent.loc[mini_batch['UserId'].values]
        user_err = -2 * errors.values[:, None] * i_latent.loc[mini_batch['ItemId'].values]
        user_gradients = pd.DataFrame(user_err.to_numpy() + user_reg.to_numpy())
        item_err = -2 * errors.values[:, None] * u_latent.loc[mini_batch['UserId'].values]
        item_reg = 2 * lambda_reg * i_latent.loc[mini_batch['ItemId'].values]
        item_gradients = pd.DataFrame(item_err.to_numpy() + item_reg.to_numpy())

        # Update user and item matrices using the averaged gradients over the mini-batch
        u_latent.loc[mini_batch['UserId'].values] -= learning_rate * user_gradients.mean(axis=0).values
        i_latent.loc[mini_batch['ItemId'].values] -= learning_rate * item_gradients.mean(axis=0).values


targets['Rating'] = targets.apply(lambda x: predict_with_sgd(x['UserId'], x['ItemId']), axis=1)


targets = targets.drop(columns = ['UserId', 'ItemId'])

targets.to_csv('submissions.csv', index=False)

end = datetime.now()

time_spent = relativedelta(end, start)

print(f'{time_spent.minutes}m{time_spent.seconds}s')
