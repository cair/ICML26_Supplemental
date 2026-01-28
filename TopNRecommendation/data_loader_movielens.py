import os
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

from elliot.data.movielens_labels import AGE_DICT, OCCUPATION_DICT, GENDER_DICT

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# From https://github.com/WojciechMigda/Tsetlini/blob/main/lib/examples/california-housing/src/produce_dataset_alt.py
# Booleanizer moved from the PyTsetlinMachineCUDA.tools module to avoid dependency issues.

def _as_bits(x, nbits):
	s = '1' * x + '0' * (nbits - x)
	return np.array([int(c) for c in s])

def _unpack_bits(a, nbits):
	if len(a.shape) > 2:
		raise ValueError("_unpack_bits: input array cannot have more than 2 dimensions, got {}".format(len(a.shape)))

	a = np.clip(a, 0, nbits)
	a_ = np.empty_like(a, dtype=np.uint64)
	np.rint(a, out=a_, casting='unsafe')
	F = np.frompyfunc(_as_bits, 2, 1)
	rv = np.stack(F(a_.ravel(), nbits)).reshape(a.shape[0], -1)
	return rv

class Booleanizer:
	def __init__(self,  max_bits_per_feature = 25):
		self.max_bits_per_feature = max_bits_per_feature

		self.kbd = KBinsDiscretizer(n_bins=max_bits_per_feature+1, encode='ordinal', strategy='quantile')

		return

	def fit(self, X):
		self.kbd_fitted = self.kbd.fit(X)
		
		return

	def transform(self, X):
		X_transformed = self.kbd_fitted.transform(X).astype(int)

		pre = FunctionTransformer(_unpack_bits, validate=False, kw_args={'nbits': self.max_bits_per_feature})
		return pre.fit_transform(X_transformed)

DATASET_PATHS = ["../data/movielens_1m_multilabel/0/", "../data/movielens_1m_multilabel/0/"]
DATASET_PATH = None
for path in DATASET_PATHS:
    if os.path.exists(path):
        DATASET_PATH = path
        break
else:
    raise FileNotFoundError("Dataset path not found in any of the specified locations.")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train.tsv")
TEST_DATASET_PATH = os.path.join(DATASET_PATH, "test.tsv")
MAX_CLASSES = -1

def load_dataset(path):
    df = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    df = df.groupby("user_id").agg(list).reset_index()
    return df

def load_movielens_dataset():
    train_df = load_dataset(TRAIN_DATASET_PATH)
    test_df = load_dataset(TEST_DATASET_PATH)

    df = train_df.merge(test_df, on="user_id", how="outer", suffixes=("_x", "_y"))
    df.rename(columns={"item_id_x": "x", "item_id_y": "y"}, inplace=True)

    return df

# Maps ratings to each position in the user-item binary representation
def build_rating_vector(items, ratings, classes):
    item_to_rating = dict(zip(items, ratings))
    return [item_to_rating.get(item, 0) for item in classes]

ADDITIONAL_MOVIELENS_DATA_PATH = "elliot/data/movielens_1m_multilabel/ml-1m"
USER_DATA_MLB_COLUMNS = {
    "gender": {"how": "direct", "dict": GENDER_DICT},
    "age": {"how": "lb", "dict": AGE_DICT},
    "occupation": {"how": "lb", "dict": OCCUPATION_DICT}
}
# Load external movielens data without any processing
def load_external_movielens_data():
    users_df = pd.read_csv(f"{ADDITIONAL_MOVIELENS_DATA_PATH}/users.dat", names=["user_id"] + list(USER_DATA_MLB_COLUMNS.keys()) + ["zip"], sep="::", header=None, engine="python", encoding="latin-1")
    for column, config in USER_DATA_MLB_COLUMNS.items():
        if config["how"] == "direct": # The gender column is already in the correct format
            continue
        users_df[column] = users_df[column].map(config["dict"])
    movies_df = pd.read_csv(f"{ADDITIONAL_MOVIELENS_DATA_PATH}/movies.dat", names=["movie_id", "title", "genres"], sep="::", header=None, engine="python", encoding="latin-1")
    movies_df["genres"] = movies_df["genres"].str.split("|")
    return users_df, movies_df

PREPROCESSED_POSTFIX = "_lb"
def load_user_data(df, include_binary_features=True):
    users_df = pd.read_csv(f"{ADDITIONAL_MOVIELENS_DATA_PATH}/users.dat", names=["user_id"] + list(USER_DATA_MLB_COLUMNS.keys()) + ["zip"], sep="::", header=None, engine="python", encoding="latin-1")
    df = df.merge(users_df, on="user_id")
    # Convert numeric feature representations to string labels.
    for column, config in USER_DATA_MLB_COLUMNS.items():
        if config["how"] == "lb": # The "direct" columns are already in the correct format
            df[column] = df[column].map(config["dict"])

    if not include_binary_features:
        return df
    
    # Convert categorical features to binary representations
    for column, config in USER_DATA_MLB_COLUMNS.items():
        if config["how"] == "lb":
            lb = LabelBinarizer()
            lb.fit(users_df[column].values)
            df[column + PREPROCESSED_POSTFIX] = list(lb.transform(df[column]))
        elif config["how"] == "direct":
            df[column + PREPROCESSED_POSTFIX] = df[column].map(config["dict"])
    return df

def flatten_list(list_of_list):
    flat_list = []
    for sublist in list_of_list:
        flat_list.extend(sublist)
    return flat_list

# Load a dataframe listing all movies by their id with their listed genre and title
def load_movie_data():
    movies_df = pd.read_csv(f"{ADDITIONAL_MOVIELENS_DATA_PATH}/movies.dat", names=["movie_id", "title", "genres"], sep="::", header=None, engine="python", encoding="latin-1")
    movies_df["genres"] = movies_df["genres"].str.split("|")
    return movies_df

MOVIE_COLUMN_NAMES = ["movie_id", "y", "items"]
# Loads the MovieLens dataset with the external data provided by Grouplens' original publication.
# Convert it to a movie_id : movie_data dict format
def load_movielens_movie_data(df):
    movies_df = load_movie_data()
    for column in MOVIE_COLUMN_NAMES:
        if column in df.columns:
            empty_movies_df = pd.DataFrame(df[column].explode().unique(), columns=["movie_id"])
            break   
    else:
        raise ValueError("Could not locate movies column, does the dataframe contain either 'movie_id', 'y' or 'items' as columns?")
    movie_data_df = empty_movies_df.merge(movies_df, on="movie_id", how="left")
    #movie_data_df["genres"] = movie_data_df["genres"].str.split("|")

    mlb = MultiLabelBinarizer()
    mlb.fit(movie_data_df["genres"])

    movie_data_df["genres_mlb"] = list(mlb.transform(movie_data_df["genres"]))

    movie_data_dict = movie_data_df.set_index("movie_id")["genres_mlb"].to_dict()
    return movie_data_dict

# Append all binary representations to the same list
def create_user_binary_representation(row):
    binary_columns = [column for column in row.index if column.endswith(PREPROCESSED_POSTFIX)]
    flat_list = flatten_list([row[col] for col in binary_columns])
    return flat_list

def convert_user_data_to_binary_dict(df):
    df["user_representation"] = df.apply(create_user_binary_representation, axis=1)
    return df.set_index("user_id")["user_representation"].to_dict()

# Loads additional data for each user and item. User data is added to both the dataframe and returned as a dictionary
# The user and movie dictionary maps their ids to a binary representation of their loaded features.
def augment_movielens_dataset(df, include_movie_features=False, include_binary_features=True):
    df = load_user_data(df, include_binary_features=include_binary_features)
    if include_movie_features:
        movie_df = load_movie_data()
        if "movie_id" in df.columns:
            df = df.merge(movie_df, on="movie_id", how="left")
        elif "y" in df.columns and "x" in df.columns:
            movie_data_dict = movie_df.set_index("movie_id")[["title", "genres"]].to_dict(orient="index")
            df["titles_x"] = df.apply(lambda row: [movie_data_dict.get(movie_id, {}).get("title", "") for movie_id in row["x"]], axis=1)
            df["genres_x"] = df.apply(lambda row: [movie_data_dict.get(movie_id, {}).get("genres", []) for movie_id in row["x"]], axis=1)
            df["titles_y"] = df.apply(lambda row: [movie_data_dict.get(movie_id, {}).get("title", "") for movie_id in row["y"]], axis=1)
            df["genres_y"] = df.apply(lambda row: [movie_data_dict.get(movie_id, {}).get("genres", []) for movie_id in row["y"]], axis=1)
        else:
            raise ValueError("Could not parse df, check format")
    if include_binary_features:
        movie_data_dict = load_movielens_movie_data(df)
        user_data_dict = convert_user_data_to_binary_dict(df)
        return df, movie_data_dict, user_data_dict

    return df, None, None

# Load train and test datasets, max classes restricts the total number of target labels
# sequential_data loads both users and items augmented with additional features from the original grouplens dataset.
#   - max_items: pads out each user's interaction history to a fixed length
# svd_reduction represents a dictionary for reducing the dimensionality of the input data
#   - n_components: The number of components to keep after SVD
#   - num_bits: The number of bits to represent each svd component
#   - include_ratings: svd encode explicit rating data rather than converting it to an implicit representation
def load_train_test_datasets(max_classes=MAX_CLASSES, sequential_data=None, svd_reduction=None):
    df = load_movielens_dataset()
    print(f"Loaded {len(df)} user profiles")

    if max_classes > 0:
        def filter_top_items(row, allowed_items):
            return [item for item in row if item in allowed_items]

        most_common_items = df["y"].explode().value_counts().index[:max_classes]
        df["y"] = df["y"].apply(lambda x: filter_top_items(x, most_common_items))
        df = df[df["y"].apply(len) > 0]

    if not sequential_data:
        x_mlb_encoder = MultiLabelBinarizer()
        df["x_mlb"] = x_mlb_encoder.fit_transform(df["x"]).tolist()
    else: 
        df, movie_data_dict, user_data_dict = augment_movielens_dataset(df)
        # Calculate the full representation length
        movie_rep_length = len(next(iter(movie_data_dict.values())))
        representation_length = sequential_data["max_items"] * movie_rep_length + len(next(iter(user_data_dict.values())))
        # Converts the disparate representations into a single binary list for each user.
        df["x_mlb"] = df.apply(lambda x: [user_data_dict[x["user_id"]]] + [movie_data_dict.get(movie_id, []) for movie_id in x["x"]], axis=1)
        df["x_mlb"] = df["x_mlb"].apply(lambda x: [int(item) for sublist in x for item in sublist]) # Flatten the list
        # Pad or truncate each user's representation to a fixed length
        df["x_mlb"] = df["x_mlb"].apply(lambda x: x[:representation_length] + [0]*(representation_length - len(x)) if len(x) < representation_length else x[:representation_length])
        df["movie_data"] = df["x"].apply(lambda x: [movie_data_dict.get(movie_id, []) for movie_id in x])

    y_mlb_encoder = MultiLabelBinarizer() # The encoding of the x data is independent of y
    df["y_mlb"] = y_mlb_encoder.fit_transform(df["y"]).tolist()


    if svd_reduction is not None:
        if svd_reduction["include_ratings"]:
            df["x_ratings"] = df.apply(
                lambda row: build_rating_vector(row["x"], row["rating_train"], x_mlb_encoder.classes_),
                axis=1
            ) # Include user ratings for each item in the svd
            x_values_to_reduce = np.stack(df["x_ratings"].values)
        else:
            x_values_to_reduce = np.stack(df["x_mlb"].values)

        svd = TruncatedSVD(n_components=svd_reduction["n_components"])
        x_reduced = svd.fit_transform(x_values_to_reduce)
        booleanizer = Booleanizer(max_bits_per_feature=svd_reduction["num_bits"])
        booleanizer.fit(x_reduced)
        x_reduced_binary = booleanizer.transform(x_reduced)

        df["x_svd_binary"] = x_reduced_binary.tolist()

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42
    )

    return df_train, df_test, (x_mlb_encoder, y_mlb_encoder)