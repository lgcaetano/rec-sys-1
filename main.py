import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import sys

start = datetime.now()

if len(sys.argv) < 3:
    print('Missing file names!')
    exit()

ratings = pd.read_csv(sys.argv[1])
targets = pd.read_csv(sys.argv[2])

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

rating_mean = ratings['Rating'].mean()

initial_latent = math.sqrt(rating_mean / k) 

initial_random_seed = 42

shufflers = [np.random.default_rng(initial_random_seed + i) for i in range(k)]

learning_rates = 0.01

reg_factor = 0.1

epochs = 4

batch_size = 64



# users_initials = np.array([[(shufflers[i].random()) * 2 - 1 for i in range(k)] for j in range(n_users)])

# items_initials = np.array([[(shufflers[i].random()) * 2 - 1 for i in range(k)] for j in range(n_items)])
users_initials = np.full((n_users, k), initial_latent)
items_initials = np.full((n_items, k), initial_latent)

u_latent = pd.DataFrame(users_initials, index=ratings['UserId'].unique())
i_latent = pd.DataFrame(items_initials, index=ratings['ItemId'].unique())

def predict_with_sgd_centered(user_id, item_id):
    return min(np.dot(u_latent.loc[user_id], i_latent.loc[item_id]), 5)

cur_perc = 0

for j in range(epochs):
    random_state = math.floor(shufflers[j].random() * 1000)
    ratings = ratings.sample(frac=1, random_state=random_state)
    for i in range(0, len(ratings), batch_size):
        mini_batch = ratings.iloc[i:i + batch_size]
        print(f'{((j * len(ratings) + i) / (len(ratings) * epochs)) * 100}%')        
        # Compute errors and gradients for the entire mini-batch using vectorized operations
        errors = mini_batch.apply(lambda row: row['Rating'] - predict_with_sgd_centered(row['UserId'], row['ItemId']), axis=1)
        user_reg = 2 * reg_factor * u_latent.loc[mini_batch['UserId'].values]
        user_err = -2 * errors.values[:, None] * i_latent.loc[mini_batch['ItemId'].values]
        user_gradients = pd.DataFrame(user_err.to_numpy() + user_reg.to_numpy())
        item_err = -2 * errors.values[:, None] * u_latent.loc[mini_batch['UserId'].values]
        item_reg = 2 * reg_factor * i_latent.loc[mini_batch['ItemId'].values]
        item_gradients = pd.DataFrame(item_err.to_numpy() + item_reg.to_numpy())

        # Update user and item matrices using the averaged gradients over the mini-batch
        u_latent.loc[mini_batch['UserId'].values] -= learning_rates * user_gradients.mean(axis=0).values
        i_latent.loc[mini_batch['ItemId'].values] -= learning_rates * item_gradients.mean(axis=0).values


def predict_with_sgd(user_id, item_id):
    return np.dot(u_latent.loc[user_id], i_latent.loc[item_id])

targets['Rating'] = targets.apply(lambda x: min(predict_with_sgd(x['UserId'], x['ItemId']), 5), axis=1)

targets = targets.drop(columns = ['UserId', 'ItemId'])

u_latent.to_csv('p_latent_matrix.csv')

i_latent.to_csv('q_latent_matrix.csv')

print(targets)

targets.to_csv('submissions.csv', index=False)

end = datetime.now()

time_spent = relativedelta(end, start)

print(f'{time_spent.minutes}m{time_spent.seconds}s')
