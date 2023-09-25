import pandas as pd
import numpy as np
import math

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

# print(ratings['ItemId'].value_counts(), ratings['UserId'].value_counts())
# print(ratings['ItemId'].nunique(), ratings['UserId'].nunique())


n_items = len(items)
n_users = len(users)
# for index, row in ratings.iterrows():
#     user_id = row['UserId']
#     r = row['Rating']
#     if user_id in u_means:
#         u_means[user_id].append(r)
#     else:
#         u_means[user_id] = [r]

 

k = 5 # latent factors

learning_rate = 0.01

lambda_reg = 0.1

epochs = 1

batch_size = 32

users_initials = np.ones((n_users, k))

items_initials = np.ones((n_items, k))


u_latent = pd.DataFrame(users_initials, index=ratings['UserId'].unique())
i_latent = pd.DataFrame(items_initials, index=ratings['ItemId'].unique())


def predict_with_sgd(user_id, item_id):
    return np.dot(u_latent.loc[user_id], i_latent.loc[item_id])

cur_perc = 0

for i in range(epochs):
    print(f'{cur_perc}%')
    # for index, row in ratings.iterrows():
    #     print(f'{index}/{len(ratings)}')
    #     if not index % 5 == 0:
    #         continue
    #     user_id = row['UserId']
    #     item_id = row['ItemId']
    #     r = row['Rating']
    #     pred = predict_with_sgd(user_id, item_id)

    #     e = r - pred

    #     u_latent.loc[user_id] += learning_rate * (e * i_latent.loc[item_id] - lambda_reg * u_latent.loc[user_id])
    #     i_latent.loc[item_id] += learning_rate * (e * u_latent.loc[user_id] - lambda_reg * i_latent.loc[item_id])
    for i in range(0, len(ratings), batch_size):
        mini_batch = ratings.iloc[i:i + batch_size]
        print(f'{(i / len(ratings)) * 100}%')        
        # Compute errors and gradients for the entire mini-batch using vectorized operations
        errors = mini_batch.apply(lambda row: row['Rating'] - np.dot(u_latent.loc[row['UserId']], i_latent.loc[row['ItemId']]), axis=1)
        user_reg = 2 * lambda_reg * u_latent.loc[mini_batch['UserId'].values]
        user_err = -2 * errors.values[:, None] * i_latent.loc[mini_batch['ItemId'].values]
        user_gradients = pd.DataFrame(user_err.to_numpy() + user_reg.to_numpy())
        item_err = -2 * errors.values[:, None] * u_latent.loc[mini_batch['UserId'].values]
        item_reg = 2 * lambda_reg * i_latent.loc[mini_batch['ItemId'].values]
        item_gradients = pd.DataFrame(item_err.to_numpy() + item_reg.to_numpy())

        # Update user and item matrices using the averaged gradients over the mini-batch
        u_latent.loc[mini_batch['UserId'].values] -= learning_rate * user_gradients.mean(axis=0).values
        i_latent.loc[mini_batch['ItemId'].values] -= learning_rate * item_gradients.mean(axis=0).values
        # user_gradients = {}
        # item_gradients = {}
        # mini_batch = ratings.iloc[i:i + batch_size]
        
        # # Accumulate gradients over interactions in mini-batch
        # for index, row in mini_batch.iterrows():
        #     user = row['UserId']
        #     item = row['ItemId']
        #     rating = row['Rating']
            
        #     # Predicted rating
        #     pred_rating = np.dot(u_latent.loc[user], i_latent.loc[item])
            
        #     # Error
        #     error = rating - pred_rating
            
        #     # Accumulate gradients
        #     if user not in user_gradients:
        #         user_gradients[user] = np.zeros(k)
        #     user_gradients[user] += error * i_latent.loc[item] - lambda_reg * u_latent.loc[user]
            
        #     if item not in item_gradients:
        #         item_gradients[item] = np.zeros(k)
        #     item_gradients[item] += error * u_latent.loc[user] - lambda_reg * i_latent.loc[item]
        
        # # Update parameters after processing the mini-batch
        # for user, gradient in user_gradients.items():
        #     u_latent.loc[user] += learning_rate * gradient
        # for item, gradient in item_gradients.items():
        #     i_latent.loc[item] += learning_rate * gradient


targets['Rating'] = targets.apply(lambda x: predict_with_sgd(x['UserId'], x['ItemId']), axis=1)


targets = targets.drop(columns = ['UserId', 'ItemId'])

targets.to_csv('submissions.csv', index=False)

# print(u_latent, i_latent)