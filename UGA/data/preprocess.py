import os, sys, traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import random
import time
from datetime import datetime
import logging

logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def preprocess_movielens_1m():
    # source: https://grouplens.org/datasets/movielens/1m/
    if not os.path.exists("./movielens_1m/ml-1m"):
        if not os.path.exists("./movielens_1m/ml-1m.zip"):
            url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            command(f"wget --no-check-certificate {url} -P ./movielens_1m/")
        command(f"unzip ./movielens_1m/ml-1m.zip -d ./movielens_1m/")

    # process movie data
    movies = pd.read_csv("./movielens_1m/ml-1m/movies.dat", sep="\:\:", header=None)
    # movies = pd.read_csv("./movielens_1m/ml-1m/movies.dat", sep="::", header=None, encoding='ISO-8859-1', engine='python')
    movies.columns = ["MovieID", "Title", "Genres"]

    movies["MovieID"] -= 1  # offset ID to start from zero
    movies.set_index("MovieID", inplace=True)
    movies["Year"] = movies["Title"].apply(lambda x: x[-6:])
    movies["Title"] = movies["Title"].apply(lambda x: x[:-7])
    movies = process_sparse_feats(movies, ["Title", "Genres", "Year"])
    movie_ids = movies.index.to_numpy()

    # process user data
    users = pd.read_csv("./movielens_1m/ml-1m/users.dat", sep="\:\:", header=None)
    users.columns = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    users["UserID"] -= 1  # offset ID to start from zero
    users.set_index("UserID", inplace=True)
    users = process_sparse_feats(users, ["Gender", "Age", "Occupation", "Zip-code"])
    user_ids = users.index.to_numpy()

    # rating data
    ratings = pd.read_csv("./movielens_1m/ml-1m/ratings.dat", sep="\:\:", header=None)
    ratings.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    ratings["UserID"] -= 1
    ratings["MovieID"] -= 1
    ratings.drop(["Rating"], axis=1, inplace=True)

    # positive data
    pos_df = ratings.rename(columns={"MovieID": "Positive"})
    train_pos_u2i, valid_pos_u2i, sequence = \
        split_and_groupby_user_with_timestamp(pos_df, "UserID", "Positive", "Timestamp", 2)

    print(train_pos_u2i.head())

    train_neg_u2i = generate_negative_samples(train_pos_u2i, movie_ids)
    valid_neg_u2i = generate_negative_samples(valid_pos_u2i, movie_ids)

    print(train_neg_u2i.head())

    # merge
    train_u2i = pd.merge(train_pos_u2i, train_neg_u2i, how="left", left_on=["UserID"], right_on=["UserID"]).reindex(
        users.index)
    print(train_u2i.head())
    valid_u2i = pd.merge(valid_pos_u2i, valid_neg_u2i, how="left", left_on=["UserID"], right_on=["UserID"]).reindex(
        users.index)
    train_u2i = train_u2i.apply(lambda s: s.fillna({i: [] for i in train_u2i.index}))
    valid_u2i = valid_u2i.apply(lambda s: s.fillna({i: [] for i in valid_u2i.index}))
    train_u2i.sort_index(inplace=True)
    valid_u2i.sort_index(inplace=True)

    # save
    np.save("./movielens_1m/user_ids.npy", user_ids)
    np.save("./movielens_1m/user_feats.npy", users.to_numpy().astype(np.int32))
    movies = movies.reindex(list(range(movie_ids.min(), movie_ids.max() + 1)), fill_value=0)  # padding
    np.save("./movielens_1m/item_feats.npy", movies.to_numpy().astype(np.int32))
    np.save("./movielens_1m/train_u2i.npy", train_u2i.to_numpy())
    np.save("./movielens_1m/valid_u2i.npy", valid_u2i.to_numpy())
    np.save("./movielens_1m/seq.npy", sequence.to_numpy())

    logging.info("Data preprocessing done")


def preprocess_ad_click():
    # source: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
    if not os.path.exists("./ad_click/raw_sample.csv"):
        if not os.path.exists("./ad_click/raw_sample.csv.tar.gz"):
            url = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=56"
            raise RuntimeError(f"Please download the data from {url} and save to ./ad_click")
        command("tar xzf ./ad_click/user_profile.csv.tar.gz -C ./ad_click/")
        command("tar xzf ./ad_click/ad_feature.csv.tar.gz -C ./ad_click/")
        command("tar xzf ./ad_click/raw_sample.csv.tar.gz -C ./ad_click/")

    # process item data
    items = pd.read_csv("./ad_click/ad_feature.csv", sep=",", header=0)
    print(items.head())
    items["adgroup_id"] -= 1  # offset ID to start from zero
    items.set_index("adgroup_id", inplace=True)
    item_ids = items.index.to_numpy()
    item_sparse_feats = process_sparse_feats(items.astype(str), ['cate_id', 'campaign_id', 'customer', 'brand']) + 1
    item_sparse_feats = item_sparse_feats.reindex(list(range(0, item_ids.max() + 1)), fill_value=0)  # padding
    item_sparse_feats.sort_index(inplace=True)
    item_dense_feats = process_dense_feats(items, ['price'])
    item_dense_feats = item_dense_feats.reindex(list(range(0, item_ids.max() + 1)), fill_value=0)  # padding
    item_dense_feats.sort_index(inplace=True)


    # process user data
    users = pd.read_csv("./ad_click/user_profile.csv", sep=",", header=0)
    print(users.head())
    users["userid"] -= 1  # offset ID to start from zero
    users.set_index("userid", inplace=True)
    users = process_sparse_feats(users.astype(str), users.columns) + 1
    user_ids = users.index.to_numpy()
    users = users.reindex(list(range(0, user_ids.max() + 1)), fill_value=0)  # padding
    users.sort_index(inplace=True)

    # process clicks
    samples = pd.read_csv("./ad_click/raw_sample.csv", sep=",", header=0)
    # sampling for quick testing
    # samples = samples.sample(frac=0.1, random_state=42)

    samples["user"] -= 1
    samples["adgroup_id"] -= 1

    # filter
    samples = samples.loc[samples["user"].isin(user_ids)]
    item_cnts = samples["adgroup_id"].value_counts()
    item_ids_to_filter = set(item_cnts[item_cnts < 5].index)
    samples = samples.loc[~samples["adgroup_id"].isin(item_ids_to_filter)]
    user_cnts = samples["user"].value_counts()
    user_ids_to_filter = set(user_cnts[user_cnts < 5].index)
    samples = samples.loc[~samples["user"].isin(user_ids_to_filter)]

    # split train and valid
    sep_ts = int(time.mktime(datetime.strptime("20170512+8", "%Y%m%d+%H").timetuple()))
    train_df = samples[samples['time_stamp'] < sep_ts]
    valid_df = samples[samples['time_stamp'] >= sep_ts]

    pos_df = split_pos_neg(train_df, "clk", 1)[["user", "adgroup_id", "time_stamp"]]
    pos_df = pos_df.sort_values("time_stamp",ascending=True)
    sequence = generate_user_sequences(pos_df, "user", "adgroup_id", "time_stamp").set_index("user")[["timestamps", "sequence"]]

    # group by users
    train_u2i = groupby_user(train_df, "user", "adgroup_id", "time_stamp","clk", True)
    valid_u2i = groupby_user(valid_df, "user", "adgroup_id", "time_stamp","clk", True)

    train_user_ids = np.sort(train_u2i.index.to_numpy())
    valid_user_ids = np.sort(valid_u2i.index.to_numpy())
    train_u2i = train_u2i.reindex(list(range(0, train_user_ids.max() + 1)), fill_value=[])  # padding
    valid_u2i = valid_u2i.reindex(list(range(0, valid_user_ids.max() + 1)), fill_value=[])  # padding
    sequence = sequence.reindex(list(range(0, train_user_ids.max() + 1)), fill_value="")
    train_u2i.sort_index(inplace=True)
    valid_u2i.sort_index(inplace=True)
    sequence.sort_index(inplace=True)
    print(train_u2i.head())
    print(sequence.head())

    if not os.path.exists("./ad_click"):
        os.mkdir("./ad_click")

    # saving
    np.save(f"./ad_click/train_user_ids.npy", train_user_ids.astype(np.int32))
    np.save(f"./ad_click/valid_user_ids.npy", valid_user_ids.astype(np.int32))
    np.save(f"./ad_click/user_sparse_feats.npy", users.to_numpy().astype(np.int32))
    np.save(f"./ad_click/item_sparse_feats.npy", item_sparse_feats.to_numpy().astype(np.int32))
    np.save(f"./ad_click/item_dense_feats.npy", item_dense_feats.to_numpy().astype(np.int32))
    np.save(f"./ad_click/train_u2i.npy", train_u2i.to_numpy())
    np.save(f"./ad_click/valid_u2i.npy", valid_u2i.to_numpy())
    np.save(f"./ad_click/seq.npy",sequence.to_numpy())

    logging.info("Data preprocessing done")


def preprocess_kuairec():
    user_feats = pd.read_csv("./KuaiRec 2.0/data/user_features.csv", sep=",", header=0).set_index('user_id')

    # process user features
    sparse_feats_name = [
        'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
        'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5',
        'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
        'onehot_feat12','onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17'
    ]
    for col in sparse_feats_name:
        user_feats[col] = user_feats[col].astype(str)
    dense_feats_name = ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']
    user_sparse_feats = process_sparse_feats(user_feats, sparse_feats_name)
    user_dense_feats  = process_dense_feats(user_feats, dense_feats_name, True)

    # process item features
    item_feats_1 = pd.read_csv("./KuaiRec 2.0/data/kuairec_caption_category.csv", sep=",", header=0, engine='python')
    item_feats_2 = pd.read_csv("./KuaiRec 2.0/data/item_categories.csv", sep=",", header=0)
    item_feats_1['video_id'] = item_feats_1['video_id'].astype(int)
    item_feats = pd.merge(item_feats_1, item_feats_2, on='video_id', how='left')

    item_feats_3 = pd.read_csv("./KuaiRec 2.0/data/item_daily_features.csv", sep=",", header=0)
    item_feats = pd.merge(item_feats, item_feats_3, on='video_id', how='left')

    # create a new column based on the video_id and the date
    item_feats['video_id'] = item_feats['video_id'].astype(str)
    item_feats['date'] = item_feats['date'].astype(str)
    item_feats['video_id_date'] = item_feats['video_id'] + item_feats['date']
    # item_feats = item_feats.drop(columns=['video_id', 'date'])
    item_feats = item_feats.set_index('video_id_date')
    video_id_date_to_idx = {v: i for i, v in enumerate(item_feats.index.unique())}
    item_feats['video_id_date_idx'] = item_feats.index.map(video_id_date_to_idx)

    # discard the video_id, manual_cover_text, caption, topic_tag, first_level_category_name, second_level_category_name, third_level_category_name
    item_feats = item_feats.drop(columns=['manual_cover_text', 'caption', 'topic_tag', 'first_level_category_name',
                                          'second_level_category_name', 'third_level_category_name', 'video_tag_name'])
    sparse_feats_name = ['video_id', 'first_level_category_id', 'second_level_category_id', 'third_level_category_id',
                         'feat', 'author_id',
                         'video_type', 'upload_dt', 'upload_type', 'visible_status', 'video_duration', 'music_id',
                         'video_tag_id', 'date']
    dense_feats_name = ['video_width', 'video_height', 'show_cnt', 'show_user_num', 'play_cnt', 'play_user_num',
                        'play_duration',
                        'complete_play_cnt', 'complete_play_user_num', 'valid_play_cnt', 'valid_play_user_num',
                        'long_time_play_cnt',
                        'long_time_play_user_num', 'short_time_play_cnt', 'short_time_play_user_num', 'play_progress',
                        'comment_stay_duration',
                        'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt', 'cancel_like_cnt',
                        'cancel_like_user_num', 'comment_cnt',
                        'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt', 'delete_comment_cnt',
                        'delete_comment_user_num', 'comment_like_cnt',
                        'comment_like_user_num', 'follow_cnt', 'follow_user_num', 'cancel_follow_cnt',
                        'cancel_follow_user_num', 'share_cnt', 'share_user_num',
                        'download_cnt', 'download_user_num', 'report_cnt', 'report_user_num', 'reduce_similar_cnt',
                        'reduce_similar_user_num', 'collect_cnt',
                        'collect_user_num', 'cancel_collect_cnt', 'cancel_collect_user_num']

    # record the features names
    item_sparse_feats = process_sparse_feats(item_feats, sparse_feats_name)
    item_dense_feats = process_dense_feats(item_feats, dense_feats_name, True)
    print(item_sparse_feats.head())
    print(item_dense_feats.head())

    samples = pd.read_csv("./KuaiRec 2.0/data/big_matrix.csv", sep=",", header=0)
    samples = samples[['user_id', 'video_id', 'watch_ratio', 'timestamp', 'date']]
    samples['video_id'] = samples['video_id'].astype(str)
    samples['date'] = samples['date'].astype(str)
    samples['video_id_date'] = samples['video_id'] + samples['date']

    samples['watch_ratio'] = (samples['watch_ratio'] > 2).astype(int)
    samples.rename(columns={"watch_ratio": "is_liked"}, inplace=True)
    # samples = samples.sample(frac=0.5, random_state=42)

    samples['video_id_date_idx'] = samples['video_id_date'].map(video_id_date_to_idx)
    samples = samples.dropna(subset=['video_id_date_idx'])

    print(samples.head())

    percentile = samples['timestamp'].quantile(0.9) # Using the RTX 4090 GPU to train the model, please set the percentile to 0.9 or lagrer

    train_df = samples[samples['timestamp'] <= percentile]
    valid_df = samples[samples['timestamp'] > percentile]

    train_df = train_df.drop(columns=['date'])
    valid_df = valid_df.drop(columns=['date'])
    pos_df = split_pos_neg(train_df, "is_liked", 1)[["user_id", "video_id_date_idx", "timestamp"]]
    sequence = generate_user_sequences(pos_df, "user_id", "video_id_date_idx", "timestamp").set_index("user_id")[["timestamps", "sequence"]]
    print(sequence.head())
    # group by users
    train_u2i = groupby_user(train_df, "user_id", "video_id_date_idx", 'timestamp', "is_liked")
    valid_u2i = groupby_user(valid_df, "user_id", "video_id_date_idx", 'timestamp', "is_liked")
    train_user_ids = np.sort(train_u2i.index.to_numpy())
    valid_user_ids = np.sort(valid_u2i.index.to_numpy())
    train_u2i = train_u2i.reindex(list(range(0, train_user_ids.max() + 1)), fill_value=[])  # padding
    valid_u2i = valid_u2i.reindex(list(range(0, valid_user_ids.max() + 1)), fill_value=[])  # padding
    sequence = sequence.reindex(list(range(0, train_user_ids.max() + 1)), fill_value="")
    train_u2i.sort_index(inplace=True)
    valid_u2i.sort_index(inplace=True)
    sequence.sort_index(inplace=True)
    print(train_u2i.head())
    print(sequence.head())

    # saving
    np.save(f"./KuaiRec 2.0/train_user_ids.npy", train_user_ids.astype(np.int32))
    np.save(f"./KuaiRec 2.0/valid_user_ids.npy", valid_user_ids.astype(np.int32))
    np.save(f"./KuaiRec 2.0/user_sparse_feats.npy", user_sparse_feats.to_numpy().astype(np.int32))
    np.save(f"./KuaiRec 2.0/item_sparse_feats.npy", item_sparse_feats.to_numpy().astype(np.int32))
    np.save(f"./KuaiRec 2.0/user_dense_feats.npy", user_dense_feats.to_numpy().astype(np.float32))
    np.save(f"./KuaiRec 2.0/item_dense_feats.npy", item_dense_feats.to_numpy().astype(np.float32))
    np.save(f"./KuaiRec 2.0/train_u2i.npy", train_u2i.to_numpy())
    np.save(f"./KuaiRec 2.0/valid_u2i.npy", valid_u2i.to_numpy())
    np.save(f"./KuaiRec 2.0/seq.npy", sequence.to_numpy())

    logging.info("Data preprocessing done")


def process_dense_feats(data, feats, use_log = False, use_minmax = False, use_standard = False):
    logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna(0.0)
    if use_log:
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    if use_minmax:
        for f in feats:
            d[f] = (d[f] - d[f].min()) / (d[f].max() - d[f].min())
    if use_standard:
        for f in feats:
            d[f] = (d[f] - d[f].mean()) / d[f].std()
    return d

def process_sparse_feats(data, feats):
    logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna("-1")

    for f in feats:
        if d[f].dtype == 'object':
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f].astype(str))
        else:
            d[f] = d[f].fillna(-1)
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])

    feature_cnt = 0
    for f in feats:
        d[f] += feature_cnt
        feature_cnt += d[f].nunique()

    return d

def generate_negative_samples(user_data, all_item_ids, neg_rate=10):
    """
    Generate negative samples for each user in user_data.
    Args:
        user_data: DataFrame or Series, where each row contains a list of (item_id, timestamp) tuples.
        all_item_ids: List of all possible item IDs.
        neg_rate: The maximum number of negative samples to generate per positive sample.
    Returns:
        neg_data: DataFrame or Series with the same format as user_data, but containing negative samples.
    """
    neg_data = user_data.copy()
    num_samples = 0

    def generate_user_negative_samples(row):
        nonlocal num_samples
        negative_samples = []
        historical_items = set()
        historical_times = set()

        # Extract historical items from the user's positive samples
        for row_cell in row[0]:
            item_id, timestamp = row_cell[0], row_cell[1]
            historical_items.add(item_id)
            historical_times.add(timestamp)

            # Generate negative samples
            neg_set = set(all_item_ids) - historical_items
            num_neg = np.random.randint(0, neg_rate + 1)  # Random number of negative samples

            # Shuffle and select negative samples
            neg_array = np.array(list(neg_set))
            np.random.shuffle(neg_array)

            for item in neg_array[:num_neg]:
                timestamp = random.choice(list(historical_times))
                negative_samples.append((item, timestamp))  # Same format as positive samples

            num_samples += num_neg

        return negative_samples

    # Iterate over each user and generate negative samples
    for user_id, row in tqdm(user_data.iterrows(), desc="Generating Negative Samples"):
        neg_samples = generate_user_negative_samples(row)
        neg_data.at[user_id, 0] = neg_samples  # Update with negative samples

    print(f"Generated {num_samples} negative samples")
    return neg_data

def filter_users_by_interactions(df, user_column, min_interactions):
    """
    Filter users with less than min_interactions
    """
    return df.groupby(user_column).filter(lambda x: len(x) >= min_interactions)

def sort_by_user_and_timestamp(df, user_column, timestamp_column, label_column_value = None):
    """
    Sort the DataFrame by user and timestamp
    """
    return df.sort_values(by=[user_column, timestamp_column])

def split_pos_neg(df, label_column, label_value):
    return df[df[label_column] == label_value]

def generate_user_sequences(df, user_column, item_column, timestamp_column):
    """
    Generate sequences for each user
    """
    def generate_sequence(x):
        x[item_column] = x[item_column].astype(str)
        x[timestamp_column] = x[timestamp_column].astype(str)

        timestamps = "".join(x[timestamp_column])
        sequence = "".join(x[item_column])

        return pd.Series({
            "timestamps": timestamps,
            "sequence": sequence
        })

    return df.groupby(user_column).apply(generate_sequence).reset_index()

def split_train_valid(df, user_column, num_valid_items):
    """
    Split the data into training and validation sets
    """
    train_records = []
    valid_records = []

    for user, user_df in df.groupby(user_column):
        if user_df.shape[0] < num_valid_items:
            continue # Skip users with insufficient interactions
        else:
            valid_records.append(user_df.iloc[-num_valid_items:])
            train_records.append(user_df.iloc[:-num_valid_items])

    return pd.concat(train_records), pd.concat(valid_records)

def generate_user_item_timestamp_list(df, user_column, item_column, timestamp_column):
    """
    Generate a list of (item, timestamp) tuples for each user
    """
    return (
        df.groupby(user_column)[[item_column, timestamp_column]]
        .apply(lambda x: list(zip(x[item_column], x[timestamp_column])))
        .reset_index()
        .set_index(user_column)
    )

def split_and_groupby_user_with_timestamp(df, user_column, item_column, timestamp_column, num_valid_items=2, min_interactions=1):
    """
    Split the data by user and group by user with timestamp
    """

    # Filter out users with less than min_interactions
    df = filter_users_by_interactions(df, user_column, min_interactions)

    # Sort by user and timestamp
    df = sort_by_user_and_timestamp(df, user_column, timestamp_column)

    # Generate sequences
    sequence_df = generate_user_sequences(df, user_column, item_column, timestamp_column).set_index(user_column, inplace=True)

    # Split into train and valid sets
    train_df, valid_df = split_train_valid(df, user_column, num_valid_items)

    print(f"Total samples in the training set: {train_df.shape[0]}")
    print(f"Total samples in the valid set: {valid_df.shape[0]}")

    # Generate train_u2i and valid_u2i
    train_u2i = generate_user_item_timestamp_list(train_df, user_column, item_column, timestamp_column)
    valid_u2i = generate_user_item_timestamp_list(valid_df, user_column, item_column, timestamp_column)

    return train_u2i, valid_u2i, sequence_df

def groupby_user(df, user_column, item_column, timestamp_column, label_column, keep_neg = False):
    if keep_neg:
        users_with_pos = df[df[label_column] == 1][user_column].unique()
        df = df[df[user_column].isin(users_with_pos)]
    pos_df = split_pos_neg(df, label_column, 1)
    neg_df = split_pos_neg(df, label_column, 0)

    pos_df = sort_by_user_and_timestamp(pos_df, user_column, timestamp_column)
    neg_df = sort_by_user_and_timestamp(neg_df, user_column, timestamp_column)

    pos_u2i = generate_user_item_timestamp_list(pos_df, user_column, item_column, timestamp_column)
    neg_u2i = generate_user_item_timestamp_list(neg_df, user_column, item_column, timestamp_column)

    u2i = pos_u2i.join(neg_u2i, how="outer", rsuffix="_neg").reset_index().set_index(user_column)

    u2i.rename(columns={'0': 'Positive', '0_neg': 'Negative'}, inplace=True)

    u2i = u2i.apply(lambda s: s.fillna({i: [] for i in u2i.index}))

    u2i = u2i[['Positive', 'Negative']]

    return u2i

def command(cmd):
    logging.info(f">>> Executing command \"{cmd}\"...")
    ret = os.system(cmd)
    assert ret == 0, f"Command failed, return code: {ret}"
    logging.info(">>> Command done")


if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except:
        logging.error("Missing dataset (movielens_1m, book_crossing, ali_ccp, ad_click, kuairec)")
        logging.error("Missing dataset (movielens_1m, book_crossing, ali_ccp, ad_click, kuairec)")
        sys.exit(1)

    logging.info(f"Preprocssing {dataset}")
    if dataset == "movielens_1m":
        preprocess_movielens_1m()
    elif dataset == "ad_click":
        preprocess_ad_click()
    elif dataset == "kuairec":
        preprocess_kuairec()
    else:
        raise ValueError(f"No such dataset: {dataset}")
