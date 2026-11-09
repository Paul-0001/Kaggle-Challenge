"""
This script does all the feature engineering for the features we chose to create.
It takes data leakage into account. (No data is used that we have for after the prediction date)
For further explanations pls. review 02_feature_engineering.ipynb under kaggle challenge -> notebooks
For further mathematical explanations pls. review Math.pdf under kaggle challenge -> notebooks
"""

# Import what needed
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Magic numbers
# Name of all app features our function should create
APP_COLUMNS = ['app_total_plays', 'app_total_listening_hours', 'app_session_count', 'app_avg_tracks_per_session', 'app_active_listening_days', 'app_plays_per_active_day', 'app_days_since_last_play', 'app_plays_last_7d', 'app_plays_last_14d', 'app_plays_last_30d', 'app_recent_activity_share', 'app_unique_genres_listened', 'app_unique_artists_listened', 'app_genre_entropy', 'app_artist_entropy', 'app_top_genre_share', 'app_top_artist_share', 'app_overall_completion_rate', 'app_overall_saved_rate', 'app_overall_search_rate', 'app_repeat_track_ratio', 'app_artist_play_count', 'app_artist_listening_minutes', 'app_artist_play_share', 'app_artist_listening_share', 'app_artist_rank', 'app_artist_is_top_1', 'app_artist_is_top_3', 'app_artist_unique_tracks', 'app_artist_last_play_days', 'app_artist_plays_last_7d', 'app_artist_plays_last_30d', 'app_artist_recent_share', 'app_artist_saved_count', 'app_artist_saved_rate', 'app_artist_search_rate', 'app_artist_completion_rate', 'app_artist_completion_minus_overall_completion_rate_difference', 'app_artist_search_minus_overall_search_rate_difference', 'app_artist_saved_minus_overall_saved_rate_difference', 'app_artist_repeat_track_ratio', 'app_genre_play_share', 'app_genre_listening_share', 'app_genre_saved_share', 'app_genre_search_share', 'app_top_genre_match', 'app_stated_vs_actual_genre_match', 'app_genre_similarity_max', 'app_genre_similarity_mean', 'app_genre_similarity_weighted', 'app_user_artist_cosine', 'app_user_artist_euclidean_distance', 'app_user_artist_js_distance']

# All artist features that have to be set to 0 if user has music_app_id bbut never listened to a given artist
ARTIST_ZERO_COLUMNS = ['app_artist_entropy', 'app_artist_play_count', 'app_artist_listening_minutes', 'app_artist_play_share', 'app_artist_listening_share', 'app_artist_is_top_1', 'app_artist_is_top_3', 'app_artist_unique_tracks', 'app_artist_plays_last_7d', 'app_artist_plays_last_30d', 'app_artist_recent_share', 'app_artist_saved_count']

# Function for ratios that divides two values and stops when dividing by 0
def safe_div(a, b):
    return a / pd.Series(b, index=a.index).replace(0, np.nan)

# Combine two values by ombininf them in strings and then combining by value1_values2 
def combine(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return (a.astype("string") + "__" + b.astype("string")).where(a.notna() & b.notna())

# Function that processes the multistring category for genre for an artist in music_app.json
def split_genres(x):
    return set(str(x).lower().split("|")) if pd.notna(x) else set()

# Compute the haversine distance between coordinates (Pls. see mathematical explanation in doc Math.pdf)
def haversine(lat1, lon1, lat2, lon2):

    # Changes the lattitude fo both points into radients
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    # Difference of lattitude
    dp = p2-p1

    # Transforms longitude for both points in radients
    dl = np.radians(lon2 - lon1)

    # Compute component of harvesine formula
    a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2

    # Return the distance between the two points (6371.0088 is the earth's radius)
    return 6371.0088 * 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# Function that computes the entropy of a frequency distribution (Pls. see mathematical expl in doc Math.pdf)
def _entropy(counts):

    # Transform the frequency into a numpy array
    x = np.asarray(counts, dtype=float)

    # Keeps only the positive observations
    x = x[x > 0]

    # If empty no entropy
    if len(x) == 0:
        return np.nan

    # Transforms the frequencies into probabilities
    p = x / x.sum()

    # Computes the classical shannon entropy
    return float(-(p * np.log(p)).sum())

# Function that computes the similarity of the direction of two vectors (Pls. see mathematical explanation in Math.pdf)
def _cosine(a, b):

    # Convert both entries in numpy array
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Product of the euclidian norm of both vectors
    den = np.linalg.norm(a) * np.linalg.norm(b)

    # Returns the cos similarity aTb/norm(a)*norm(b)
    return float(a @ b / den) if den > 0 else np.nan

# Function that computes the Jensen-Shannon distance (Pls. see math explanation in doc Math.pdf)
def _js_distance(a, b):

    # Transform both entries in np array
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Sum of each entry has to be positive else empty
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan

    # Sum over each element in vector a and b (Normalization)
    a = a / a.sum()
    b = b / b.sum()

    # Compute the average of both (Middle distribution)
    m = (a + b) / 2

    # define inner function for Kullback-Leibner-divergence
    def kl(p, q):

        # Marks everything where p positive (No 0*log(0))
        mask = p > 0

        # Return the divergence
        return float((p[mask] *np.log(p[mask] / q[mask])).sum())

    # return the Jensen-Shannon difference for divergence in both directions since non symmetric
    return float(np.sqrt(max(0, (kl(a, m) + kl(b, m)) / 2)))

# Transform all features of music_app.json into numerical values for further computation with features
def _music_features(df, music_app_path):

    # Open and load the doc music_app.json
    with open(music_app_path, encoding="utf-8") as f:
        music = json.load(f)

    # Cutoff all information we cannot use to avoid data leakage
    cutoff= pd.Timestamp(music["history_ends_before"])

    # Get all genres
    genres = list(music["catalog"]["genres"])

    # Get all artists and their id and genres (Dictionnary)
    artists={a["artist_id"]:a for a in music["catalog"]["artists"]}

    # Create empty list 
    rows=[]

    # Create empty dictionnary
    statuses={}

    # Iterate over all accounts in the json doc
    for account in music["accounts"]:

        # Get the user id for the respective iteration
        uid = account["music_app_user_id"]

        # Gets the activity status of the user and puts it in the statuses dict with user id key
        statuses[uid] = account.get("activity_status")

        # Iterate over all sessions of an user
        for session in account.get("sessions", []):

            # Iterate over all sings that were listened to in that session
            for track in session.get("tracks", []):

                # Get the timestamp of when the track was listened to
                t = pd.Timestamp(track["played_at"])

                # Stop if timestamp > prediction date
                if t >= cutoff:
                    continue

                # Appends a row to our list rows which has all infromation for a track
                rows.append((uid, session["session_id"], t, track["artist_id"], track["genre"], track["track_id"],track["played_ms"], track["track_ms"], track["source"], track["saved"]))

    # Define all columns tht are in the json
    cols = ['music_app_user_id', 'session_id', 'played_at', 'artist_id','genre', 'track_id', 'played_ms', 'track_ms', 'source', 'saved']

    # Transform rows list in dataframe called plays
    plays = pd.DataFrame(rows, columns=cols)

    # If empty plays dataframe is empty return empy dataframe and statuses dict
    if plays.empty:
        return pd.DataFrame(index=df.index), statuses

    # Convert playing time for a track into numerical values if error set NaN
    plays['played_ms'] = pd.to_numeric(plays['played_ms'], errors='coerce')

    # Covert track lenght into numerical values if error set NaN
    plays['track_ms'] = pd.to_numeric(plays['track_ms'],errors='coerce')

    # Compute playing ratio (How much did the user listen to that track in relation to how long the track is)
    plays['completion'] = np.minimum(plays['played_ms']/plays['track_ms'].replace(0,np.nan), 1)

    # If user searched for the track 1, else 0 (Indicatorfunction)
    plays['is_search'] = (plays['source']=='search').astype(float)

    # Indicatorfunction if user saved the track. 1 if yes else 0
    plays['is_saved'] = plays['saved'].astype(float)

    # How long before the cutoff was this track last played
    plays['days_ago'] = (cutoff-plays['played_at']).dt.total_seconds() / 86400

    # Get the date at which the track was listened to
    plays['listening_date'] = plays['played_at'].dt.date

    # Indicator if track was played in the last 7days. 1 if yes else 0
    plays['last_7d'] = (plays['days_ago']<=7).astype(int)

    # Same for fourteen days
    plays['last_14d'] = (plays['days_ago']<=14).astype(int)

    # Same for 30 days
    plays['last_30d'] = (plays['days_ago']<=30).astype(int)

    # Computes a matrix for the whole population with users as rows and genres as columns. Each elements represents how often a user listened to that genre.
    population_counts = pd.crosstab(plays['music_app_user_id'],plays['genre'])

    # Makes safe all genres are in matrix even if not one user listened to them
    population_counts = population_counts.reindex(columns=genres,fill_value=0)

    # Convert into np matrix
    population_mat = population_counts.to_numpy(dtype=float)

    # Normalize each row with the whole population data
    population_mat = population_mat / np.maximum(population_mat.sum(axis=1, keepdims=True), 1)


    # Aggregate only accounts present in this input frame 
    needed = set(df['music_app_user_id'].dropna())

    # Applicate that to plays df
    plays = plays[plays['music_app_user_id'].isin(needed)]

    # Empty dict for aggregate user data
    U = {}

    # Empty dict for user--artist combination
    A = {}

    # Empty dict for user genre frequencies vector
    user_profiles = {}

    # Iterate over each user id and its corresponding df with all his track history
    for uid, g in plays.groupby('music_app_user_id',sort=False):

        # How many track did e listen to
        total = len(g)

        # How many plays per genre
        counts = g['genre'].value_counts()

        # How many plays per artist
        ac = g['artist_id'].value_counts()

        # On how many different days was the user active
        active_days = g['listening_date'].nunique()

        # Summ of all plays in the last 7 days
        recent = g['last_7d'].sum()

        # Genre vector for each user. How often did he listen to each genre
        profile = np.array([counts.get(genre, 0) for genre in genres], dtype=float)

        # Save the vector un user_profiles. Key user id
        user_profiles[uid] = profile

        # Fill the empty dict for aggregate user data
        U[uid] = {

            # What is the users total plays
            'app_total_plays':total,

            # Users total listening hours
            'app_total_listening_hours':g['played_ms'].sum() / 3600000,

            # Number of sessions the user had
            'app_session_count':g['session_id'].nunique(),

            # Number of average tracks the user listened to in a session
            'app_avg_tracks_per_session':total / g['session_id'].nunique(),

            # On how many different days was the user active
            'app_active_listening_days':active_days,

            # On average how many tracks did the user listen per active day
            'app_plays_per_active_day':total / active_days,

            # When was the user last active
            'app_days_since_last_play':g['days_ago'].min(),

            # Number of tracks listened to in the last 7 days
            'app_plays_last_7d':recent,

            # Number of tracks listened to in the last 14 days
            'app_plays_last_14d':g['last_14d'].sum(),

            # Number of tracks listened to in the last 30 days
            'app_plays_last_30d':g['last_30d'].sum(),
            
            # Portion of the last 7 days in ratio with rest
            'app_recent_activity_share':recent / total,

            # How many unique genres did the user listen to
            'app_unique_genres_listened':g['genre'].nunique(),

            # How many unique artists did the user listen to
            'app_unique_artists_listened':g['artist_id'].nunique(),

            # Shannon entropy of the genre frequency of the user
            'app_genre_entropy':_entropy(counts.values),

            # Shannon entropy of the artist frequency of the user
            'app_artist_entropy':_entropy(ac.values),

            # Fraction of the most listened to genre by the user
            'app_top_genre_share':counts.iloc[0] / total,

            # Fraction of the most listened to artist by the user
            'app_top_artist_share':ac.iloc[0] / total,

            # Mean of the users completition rate of a track
            'app_overall_completion_rate':g['completion'].mean(),

            # Fraction of the tracks the user saved
            'app_overall_saved_rate':g['is_saved'].mean(),

            # Fraction of the tracks the user searched for
            'app_overall_search_rate':g['is_search'].mean(),

            # Fraction of the tracks the user listened to more than once (1-unique hearings)
            'app_repeat_track_ratio':1 - g['track_id'].nunique()/total,

            # Name of the most heared genre by the user
            '_top_genre':counts.index[0], 
            
            # Number of unique genres the user listens to
            '_genre_counts':counts.to_dict(),

            # Number of minutes the user listened to a genre
            '_genre_minutes':g.groupby('genre')['played_ms'].sum().to_dict(),

            # Number of tracks of a genre that the user saved
            '_genre_saved':g.groupby('genre')['is_saved'].sum().to_dict(),

            # Number of tracks of a genre the user searched for
            '_genre_search':g.groupby('genre')['is_search'].sum().to_dict(),

            # Total minutes per genre
            '_genre_total_minutes':g['played_ms'].sum(),

            # Number of saved tracks
            '_genre_total_saved':g['is_saved'].sum(),

            # Number of searches
            '_genre_total_search':g['is_search'].sum(),
        }

        # Iterate through all the artists a user listened to
        for rank, (artist,_) in enumerate(ac.items(), 1):

            # Get all information for agiven artist
            a = g[g['artist_id'] == artist]

            # Number of artist listen to by an user
            n=len(a)

            # Create dictionnary entry with tuple as key (Tpule=user id and artist we are analyzing)
            A[(uid,artist)]={

                # Number of artists listened to by a user
                'app_artist_play_count':n,

                # Played minutes by an artist
                'app_artist_listening_minutes':a['played_ms'].sum() / 60000,

                # Fraction of played minutes by this artist in relation to all minutes listened to that artist
                'app_artist_play_share':n / total,

                # Fraction of played minutes by this artist in relation to all minutes listened to by user
                'app_artist_listening_share':a['played_ms'].sum() / g['played_ms'].sum() if g['played_ms'].sum() else np.nan,

                # Get userspecific rank for that artist
                'app_artist_rank':rank,
                
                # Indicator function if rank of that artist is 1
                'app_artist_is_top_1':int(rank == 1),

                # Indicator function if rank of that artist is top 3
                'app_artist_is_top_3':int(rank <= 3),

                # Number of unique tracks by that artist listened to by the user
                'app_artist_unique_tracks':a['track_id'].nunique(),

                # Number of days since last track of that artist
                'app_artist_last_play_days':a['days_ago'].min(),

                # Indicator function if that artist was played in the last 7 days
                'app_artist_plays_last_7d':a['last_7d'].sum(),

                # Same for last 30 days
                'app_artist_plays_last_30d':a['last_30d'].sum(),

                # Fraction of recent tracks by that artist in relation to all recent tracks
                'app_artist_recent_share':a['last_7d'].sum() / recent if recent else np.nan,

                # Number of tracks by that artist the user saved
                'app_artist_saved_count':a['is_saved'].sum(),

                # Fraction of how many tracks by that artist are saved in relation to all saved tracks
                'app_artist_saved_rate':a['is_saved'].mean(),

                # Number of tracks by that artist that were searched by the user
                'app_artist_search_rate':a['is_search'].mean(),

                # Completition rate of the user for given artist
                'app_artist_completion_rate':a['completion'].mean(),

                # artist completition rate minus restant average completition rate
                'app_artist_completion_minus_overall_completion_rate_difference':a['completion'].mean() - g['completion'].mean(),

                # artist search rate minus overall search rate
                'app_artist_search_minus_overall_search_rate_difference':a['is_search'].mean() - g['is_search'].mean(),

                # artist saved rate minus overall saved rate
                'app_artist_saved_minus_overall_saved_rate_difference':a['is_saved'].mean() - g['is_saved'].mean(),

                # Fraction of tracks by the artist the user only listened to once in relation to the number of tracks the user listened to by the artist
                'app_artist_repeat_track_ratio':1 - a['track_id'].nunique() / n,
            }

    # Get matrix into variable mat
    mat = population_mat

    # Create identity matrix with # genres x # genres dimensions (Should be 11x11 dim)
    sim = np.eye(len(genres))

    # iterate throug the rows of the matrix
    for i in range(len(genres)):

        # Iterate through the columns of the matrix
        for j in range(len(genres)):

            # Compue similarity of genres using cosine similarity with function defined before
            sim[i, j] = _cosine(mat[:, i],mat[:, j]) if mat[:, i].sum() > 0 and mat[:, j].sum() > 0 else float(i == j)

    # Create dictionnary which gives each genre its index
    gi={g:i for i,g in enumerate(genres)}

    # Empty list creation
    output=[]

    # Each iteration goes through user id, artist id, the users genre preferences and the genres and connects them
    for uid, artist,stated, genres_string in zip(df['music_app_user_id'], df['artist_id'], df['stated_favourite_genre'], df['genres']):

        # Create a dictionnary for app feature name and set all to NaN for start (If user doesnt have app it stays NaN)
        base = {k:np.nan for k in APP_COLUMNS}

        # Get the aggregated userinformations from U for a specific user
        u = U.get(uid)

        # If not empty we continue
        if u is not None:

            # Updates the dicts with all information we need (Defined in APP_COLUMNS)
            base.update({k:v for k,v in u.items() if k in APP_COLUMNS})


            # Get all the artist information for a given user and put it in a
            a = A.get((uid,artist))

            # If a exists and not empty
            if a:

                # Update the base with the informations
                base.update(a)

            # else some features of artist_Zero_columns are set to 0
            else:
                for k in ARTIST_ZERO_COLUMNS:base[k] = 0

            # Number of genres for an artist
            offered = set(split_genres(genres_string))

            # Number of tracks played for a genre
            gcounts = u['_genre_counts']

            # Number if minute listened to a genre
            gminutes = u['_genre_minutes']

            # Number of tracks from a genre that are saved
            gsaved = u['_genre_saved']

            # Number of track for a genre that were searched
            gsearch = u['_genre_search']

            # Fraction of percentages of listened tracks of a genre by a user for the genre of an artist
            base['app_genre_play_share'] = sum(gcounts.get(x, 0) for x in offered) / u['app_total_plays']

            # Fraction of percentages of listened minutes of a genre by a user for the genre of an artist
            base['app_genre_listening_share'] = sum(gminutes.get(x, 0) for x in offered) / u['_genre_total_minutes'] if u['_genre_total_minutes'] else np.nan

            # fraction of total saved tracks for a genre in relation to the artists genres
            base['app_genre_saved_share'] = sum(gsaved.get(x, 0) for x in offered) / u['_genre_total_saved'] if u['_genre_total_saved'] else np.nan

            # fraction of total searched tracks for a genre in relation to the artists genres
            base['app_genre_search_share'] = sum(gsearch.get(x, 0) for x in offered) / u['_genre_total_search'] if u['_genre_total_search'] else np.nan

            # Is most listened to genre of users in the genres of the artist. 1 if yes 0 else
            base['app_top_genre_match'] = int(u['_top_genre'] in offered)

            # Comparison between favorite genre of user and real favorite genre by playing time
            base['app_stated_vs_actual_genre_match'] = int(str(stated).lower() == u['_top_genre']) if pd.notna(stated) else np.nan

            # Creating indexing of all genres
            valid = [gi[x] for x in offered if x in gi]

            # Looks if fav genre exist, if its valid and if min one artist has that genre
            if pd.notna(stated) and stated in gi and valid:

                # Computes the similarity between users fav genre and the genres the artist has
                vals = [sim[gi[stated], j] for j in valid]

                # Saves the max similarity
                base['app_genre_similarity_max'] = max(vals)

                # Saves the mean similarity
                base['app_genre_similarity_mean'] = float(np.mean(vals))

            # Get user  genre frequenc< vector
            profile = user_profiles[uid]

            # Get artist frequency vector. If genre in artists genres 1 else 0
            artist_profile = np.array([int(x in offered) for x in genres], dtype=float)

            # Checks if both user an artist vectors have a positive sum
            if profile.sum() > 0 and artist_profile.sum() > 0:

                # Compue cosine similarity between user genre frequency vector and artist genre vector
                base['app_user_artist_cosine'] = _cosine(profile, artist_profile)

                # normalize user vector
                p = profile /profile.sum()

                # normalize artist vector
                q = artist_profile / artist_profile.sum()

                # Euclidian distance between both normalized vectors
                base['app_user_artist_euclidean_distance'] = float(np.linalg.norm(p - q))

                # Jenson-Shannon distance between both normalized vectors
                base['app_user_artist_js_distance'] = _js_distance(p, q)

                # 
                base['app_genre_similarity_weighted'] =float(sum((profile[i] / profile.sum())*max(sim[i, j] for j in valid) for i in range(len(genres)))) if valid else np.nan

        # Append all new featurs to base
        output.append(base)

    # Return the new df and the statuses
    return pd.DataFrame(output,index=df.index), statuses

# Function that builds the features and preserves the original feautures as well as order. Detail feature by featire analysis and expl. in 02_feature_engineerin.ipynb
def build_features(data: pd.DataFrame, music_app_path: str | Path) -> pd.DataFrame:

    # Copy the data in a dataframe called df
    df=data.copy().reset_index(drop=True)

    # Convert all date values in datetime values python can read as a date
    for date_column in ['concert_at','sale_opened_at','prediction_at','travel_search_arrival']:
        df[date_column]=pd.to_datetime(df[date_column],utc=True,errors='coerce')

    lookup,statuses=_music_features(df,music_app_path)

    # app_activity_status: categorical status of linked music-app account.
    df['app_activity_status'] = df['music_app_user_id'].map(statuses).fillna('unlinked')

    # genre_match: [int(str(f).strip().lower() in split_genres(g)) if pd.notna(f) else np.nan for f,g in zip(df["stated_favourite_genre"],df["genres"])]
    df["genre_match"] = [int(str(f).strip().lower() in split_genres(g)) if pd.notna(f) else np.nan for f,g in zip(df["stated_favourite_genre"],df["genres"])]

    # distance_km: haversine(df["home_lat"],df["home_lon"],df["venue_lat"],df["venue_lon"])
    df["distance_km"] = haversine(df["home_lat"],df["home_lon"],df["venue_lat"],df["venue_lon"])

    # music_app_user_id_missing: df["music_app_user_id"].isna().astype(int)
    df["music_app_user_id_missing"] = df["music_app_user_id"].isna().astype(int)

    # age_squared: df["age"] ** 2
    df["age_squared"] = df["age"] ** 2

    # education_x_spending_capacity_band: combine(df["education"],df["spending_capacity_band"])
    df["education_x_spending_capacity_band"] = combine(df["education"], df["spending_capacity_band"])

    # age_x_stated_favourite_genre: combine(df["age"].floordiv(10).mul(10),df["stated_favourite_genre"])
    df["age_x_stated_favourite_genre"] = combine(df["age"].floordiv(10).mul(10), df["stated_favourite_genre"])

    # age_x_spending_capacity_band: df["age"] * df["spending_capacity_band"]
    df["age_x_spending_capacity_band"] = df["age"] * df["spending_capacity_band"]

    # age_x_education: combine(df["age"].floordiv(10).mul(10),df["education"])
    df["age_x_education"] = combine(df["age"].floordiv(10).mul(10),df["education"])

    # past_purchase_count_squared: df["past_purchase_count"] ** 2
    df["past_purchase_count_squared"] = df["past_purchase_count"] ** 2

    # log_past_purchase_count: np.log1p(df["past_purchase_count"])
    df["log_past_purchase_count"] = np.log1p(df["past_purchase_count"])

    # past_purchase_count_x_typical_ticket_spend: df["past_purchase_count"] * df["typical_ticket_spend"]
    df["past_purchase_count_x_typical_ticket_spend"] = df["past_purchase_count"] * df["typical_ticket_spend"]

    # first_timer_x_wishlisted: df["first_timer"] * df["wishlisted"]
    df["first_timer_x_wishlisted"] = df["first_timer"] * df["wishlisted"]

    # typical_ticket_spend_squared: df["typical_ticket_spend"] ** 2
    df["typical_ticket_spend_squared"] = df["typical_ticket_spend"] ** 2

    # ticket_price_squared: df["ticket_price"] ** 2
    df["ticket_price_squared"] = df["ticket_price"] ** 2

    # discount_amount: df["ticket_price"] * df["discount_pct"] / 100
    df["discount_amount"] = df["ticket_price"] * df["discount_pct"] / 100

    # booking_fee_to_ticket_price_ratio: safe_div(df["booking_fee"],df["ticket_price"])
    df["booking_fee_to_ticket_price_ratio"] = safe_div(df["booking_fee"],df["ticket_price"])

    # ticket_price_to_typical_ticket_spend_ratio: safe_div(df["ticket_price"],df["typical_ticket_spend"])
    df["ticket_price_to_typical_ticket_spend_ratio"] = safe_div(df["ticket_price"],df["typical_ticket_spend"])

    # ticket_price_minus_typical_ticket_spend_difference: df["ticket_price"] - df["typical_ticket_spend"]
    df["ticket_price_minus_typical_ticket_spend_difference"] = df["ticket_price"] - df["typical_ticket_spend"]

    # ticket_price_above_typical_ticket_spend: np.where(df["typical_ticket_spend"].notna(),(df["ticket_price"] > df["typical_ticket_spend"]).astype(int),np.nan)
    df["ticket_price_above_typical_ticket_spend"] = np.where(df["typical_ticket_spend"].notna(),(df["ticket_price"] > df["typical_ticket_spend"]).astype(int),np.nan)

    # ticket_price_x_spending_capacity_band: df["ticket_price"] * df["spending_capacity_band"]
    df["ticket_price_x_spending_capacity_band"] = df["ticket_price"] * df["spending_capacity_band"]

    # ticket_price_x_first_timer: df["ticket_price"] * df["first_timer"]
    df["ticket_price_x_first_timer"] = df["ticket_price"] * df["first_timer"]

    # discount_pct_x_first_timer: df["discount_pct"] * df["first_timer"]
    df["discount_pct_x_first_timer"] = df["discount_pct"] * df["first_timer"]

    # discount_pct_x_spending_capacity_band: df["discount_pct"] * df["spending_capacity_band"]
    df["discount_pct_x_spending_capacity_band"] = df["discount_pct"] * df["spending_capacity_band"]

    # discount_pct_x_ticket_price: df["discount_pct"] * df["ticket_price"]
    df["discount_pct_x_ticket_price"] = df["discount_pct"] * df["ticket_price"]

    # discount_pct_x_wishlisted: df["discount_pct"] * df["wishlisted"]
    df["discount_pct_x_wishlisted"] = df["discount_pct"] * df["wishlisted"]

    # discount_pct_x_event_page_visits_7d: df["discount_pct"] * df["event_page_visits_7d"]
    df["discount_pct_x_event_page_visits_7d"] = df["discount_pct"] * df["event_page_visits_7d"]

    # same_city: (df["home_city"].str.casefold() == df["venue_city"].str.casefold()).astype(int)
    df["same_city"] = (df["home_city"].str.casefold() == df["venue_city"].str.casefold()).astype(int)

    # travel_search_present: df["travel_search_city"].notna().astype(int)
    df["travel_search_present"] = df["travel_search_city"].notna().astype(int)

    # travel_destination_match: np.where(df["travel_search_city"].notna(),(df["travel_search_city"].str.casefold() == df["venue_city"].str.casefold()).astype(int),np.nan)
    df["travel_destination_match"] = np.where(df["travel_search_city"].notna(),(df["travel_search_city"].str.casefold() == df["venue_city"].str.casefold()).astype(int),np.nan)

    # concert_at_minus_travel_search_arrival_difference_days: (df["concert_at"] - df["travel_search_arrival"]).dt.total_seconds() / 86400
    df["concert_at_minus_travel_search_arrival_difference_days"] = (df["concert_at"] - df["travel_search_arrival"]).dt.total_seconds() / 86400

    # days_until_concert: (df["concert_at"] - df["prediction_at"]).dt.total_seconds() / 86400
    df["days_until_concert"] = (df["concert_at"] - df["prediction_at"]).dt.total_seconds() / 86400

    # days_since_sale_opened: (df["prediction_at"] - df["sale_opened_at"]).dt.total_seconds() / 86400
    df["days_since_sale_opened"] = (df["prediction_at"] - df["sale_opened_at"]).dt.total_seconds() / 86400

    # sale_window_days: (df["concert_at"] - df["sale_opened_at"]).dt.total_seconds() / 86400
    df["sale_window_days"] = (df["concert_at"] - df["sale_opened_at"]).dt.total_seconds() / 86400

    # concert_month: df["concert_at"].dt.month
    df["concert_month"] = df["concert_at"].dt.month

    # concert_weekday: df["concert_at"].dt.dayofweek
    df["concert_weekday"] = df["concert_at"].dt.dayofweek

    # concert_hour: df["concert_at"].dt.hour
    df["concert_hour"] = df["concert_at"].dt.hour

    # event_page_visits_7d_squared: df["event_page_visits_7d"] ** 2
    df["event_page_visits_7d_squared"] = df["event_page_visits_7d"] ** 2

    # has_event_page_visits_7d: np.where(df["event_page_visits_7d"].notna(),(df["event_page_visits_7d"] > 0).astype(int),np.nan)
    df["has_event_page_visits_7d"] = np.where(df["event_page_visits_7d"].notna(),(df["event_page_visits_7d"] > 0).astype(int),np.nan)

    # wishlisted_x_event_page_visits_7d: df["wishlisted"] * df["event_page_visits_7d"]
    df["wishlisted_x_event_page_visits_7d"] = df["wishlisted"] * df["event_page_visits_7d"]

    # event_page_visits_7d_missing: df["event_page_visits_7d"].isna().astype(int)
    df["event_page_visits_7d_missing"] = df["event_page_visits_7d"].isna().astype(int)

    # artist_genre_count: df["genres"].map(lambda x:len(split_genres(x)))
    df["artist_genre_count"] = df["genres"].map(lambda x:len(split_genres(x)))

    # app_total_plays: lookup["app_total_plays"].to_numpy()
    df["app_total_plays"] = lookup["app_total_plays"].to_numpy()

    # app_total_listening_hours: lookup["app_total_listening_hours"].to_numpy()
    df["app_total_listening_hours"] = lookup["app_total_listening_hours"].to_numpy()

    # app_session_count: lookup["app_session_count"].to_numpy()
    df["app_session_count"] = lookup["app_session_count"].to_numpy()

    # app_avg_tracks_per_session: lookup["app_avg_tracks_per_session"].to_numpy()
    df["app_avg_tracks_per_session"] = lookup["app_avg_tracks_per_session"].to_numpy()

    # app_active_listening_days: lookup["app_active_listening_days"].to_numpy()
    df["app_active_listening_days"] = lookup["app_active_listening_days"].to_numpy()

    # app_plays_per_active_day: lookup["app_plays_per_active_day"].to_numpy()
    df["app_plays_per_active_day"] = lookup["app_plays_per_active_day"].to_numpy()

    # app_days_since_last_play: lookup["app_days_since_last_play"].to_numpy()
    df["app_days_since_last_play"] = lookup["app_days_since_last_play"].to_numpy()

    # app_plays_last_7d: lookup["app_plays_last_7d"].to_numpy()
    df["app_plays_last_7d"] = lookup["app_plays_last_7d"].to_numpy()

    # app_plays_last_14d: lookup["app_plays_last_14d"].to_numpy()
    df["app_plays_last_14d"] = lookup["app_plays_last_14d"].to_numpy()

    # app_plays_last_30d: lookup["app_plays_last_30d"].to_numpy()
    df["app_plays_last_30d"] = lookup["app_plays_last_30d"].to_numpy()

    # app_recent_activity_share: lookup["app_recent_activity_share"].to_numpy()
    df["app_recent_activity_share"] = lookup["app_recent_activity_share"].to_numpy()

    # app_unique_genres_listened: lookup["app_unique_genres_listened"].to_numpy()
    df["app_unique_genres_listened"] = lookup["app_unique_genres_listened"].to_numpy()

    # app_unique_artists_listened: lookup["app_unique_artists_listened"].to_numpy()
    df["app_unique_artists_listened"] = lookup["app_unique_artists_listened"].to_numpy()

    # app_genre_entropy: lookup["app_genre_entropy"].to_numpy()
    df["app_genre_entropy"] = lookup["app_genre_entropy"].to_numpy()

    # app_artist_entropy: lookup["app_artist_entropy"].to_numpy()
    df["app_artist_entropy"] = lookup["app_artist_entropy"].to_numpy()

    # app_top_genre_share: lookup["app_top_genre_share"].to_numpy()
    df["app_top_genre_share"] = lookup["app_top_genre_share"].to_numpy()

    # app_top_artist_share: lookup["app_top_artist_share"].to_numpy()
    df["app_top_artist_share"] = lookup["app_top_artist_share"].to_numpy()

    # app_overall_completion_rate: lookup["app_overall_completion_rate"].to_numpy()
    df["app_overall_completion_rate"] = lookup["app_overall_completion_rate"].to_numpy()

    # app_overall_saved_rate: lookup["app_overall_saved_rate"].to_numpy()
    df["app_overall_saved_rate"] = lookup["app_overall_saved_rate"].to_numpy()

    # app_overall_search_rate: lookup["app_overall_search_rate"].to_numpy()
    df["app_overall_search_rate"] = lookup["app_overall_search_rate"].to_numpy()

    # app_repeat_track_ratio: lookup["app_repeat_track_ratio"].to_numpy()
    df["app_repeat_track_ratio"] = lookup["app_repeat_track_ratio"].to_numpy()

    # app_artist_play_count: lookup["app_artist_play_count"].to_numpy()
    df["app_artist_play_count"] = lookup["app_artist_play_count"].to_numpy()

    # app_artist_listening_minutes: lookup["app_artist_listening_minutes"].to_numpy()
    df["app_artist_listening_minutes"] = lookup["app_artist_listening_minutes"].to_numpy()

    # app_artist_play_share: lookup["app_artist_play_share"].to_numpy()
    df["app_artist_play_share"] = lookup["app_artist_play_share"].to_numpy()

    # app_artist_listening_share: lookup["app_artist_listening_share"].to_numpy()
    df["app_artist_listening_share"] = lookup["app_artist_listening_share"].to_numpy()

    # app_artist_rank: lookup["app_artist_rank"].to_numpy()
    df["app_artist_rank"] = lookup["app_artist_rank"].to_numpy()

    # app_artist_is_top_1: lookup["app_artist_is_top_1"].to_numpy()
    df["app_artist_is_top_1"] = lookup["app_artist_is_top_1"].to_numpy()

    # app_artist_is_top_3: lookup["app_artist_is_top_3"].to_numpy()
    df["app_artist_is_top_3"] = lookup["app_artist_is_top_3"].to_numpy()

    # app_artist_unique_tracks: lookup["app_artist_unique_tracks"].to_numpy()
    df["app_artist_unique_tracks"] = lookup["app_artist_unique_tracks"].to_numpy()

    # app_artist_last_play_days: lookup["app_artist_last_play_days"].to_numpy()
    df["app_artist_last_play_days"] = lookup["app_artist_last_play_days"].to_numpy()

    # app_artist_plays_last_7d: lookup["app_artist_plays_last_7d"].to_numpy()
    df["app_artist_plays_last_7d"] = lookup["app_artist_plays_last_7d"].to_numpy()

    # app_artist_plays_last_30d: lookup["app_artist_plays_last_30d"].to_numpy()
    df["app_artist_plays_last_30d"] = lookup["app_artist_plays_last_30d"].to_numpy()

    # app_artist_recent_share: lookup["app_artist_recent_share"].to_numpy()
    df["app_artist_recent_share"] = lookup["app_artist_recent_share"].to_numpy()

    # app_artist_saved_count: lookup["app_artist_saved_count"].to_numpy()
    df["app_artist_saved_count"] = lookup["app_artist_saved_count"].to_numpy()

    # app_artist_saved_rate: lookup["app_artist_saved_rate"].to_numpy()
    df["app_artist_saved_rate"] = lookup["app_artist_saved_rate"].to_numpy()

    # app_artist_search_rate: lookup["app_artist_search_rate"].to_numpy()
    df["app_artist_search_rate"] = lookup["app_artist_search_rate"].to_numpy()

    # app_artist_completion_rate: lookup["app_artist_completion_rate"].to_numpy()
    df["app_artist_completion_rate"] = lookup["app_artist_completion_rate"].to_numpy()

    # app_artist_completion_minus_overall_completion_rate_difference: lookup["app_artist_completion_minus_overall_completion_rate_difference"].to_numpy()
    df["app_artist_completion_minus_overall_completion_rate_difference"] = lookup["app_artist_completion_minus_overall_completion_rate_difference"].to_numpy()

    # app_artist_search_minus_overall_search_rate_difference: lookup["app_artist_search_minus_overall_search_rate_difference"].to_numpy()
    df["app_artist_search_minus_overall_search_rate_difference"] = lookup["app_artist_search_minus_overall_search_rate_difference"].to_numpy()

    # app_artist_saved_minus_overall_saved_rate_difference: lookup["app_artist_saved_minus_overall_saved_rate_difference"].to_numpy()
    df["app_artist_saved_minus_overall_saved_rate_difference"] = lookup["app_artist_saved_minus_overall_saved_rate_difference"].to_numpy()

    # app_artist_repeat_track_ratio: lookup["app_artist_repeat_track_ratio"].to_numpy()
    df["app_artist_repeat_track_ratio"] = lookup["app_artist_repeat_track_ratio"].to_numpy()

    # app_genre_play_share: lookup["app_genre_play_share"].to_numpy()
    df["app_genre_play_share"] = lookup["app_genre_play_share"].to_numpy()

    # app_genre_listening_share: lookup["app_genre_listening_share"].to_numpy()
    df["app_genre_listening_share"] = lookup["app_genre_listening_share"].to_numpy()

    # app_genre_saved_share: lookup["app_genre_saved_share"].to_numpy()
    df["app_genre_saved_share"] = lookup["app_genre_saved_share"].to_numpy()

    # app_genre_search_share: lookup["app_genre_search_share"].to_numpy()
    df["app_genre_search_share"] = lookup["app_genre_search_share"].to_numpy()

    # app_top_genre_match: lookup["app_top_genre_match"].to_numpy()
    df["app_top_genre_match"] = lookup["app_top_genre_match"].to_numpy()

    # app_stated_vs_actual_genre_match: lookup["app_stated_vs_actual_genre_match"].to_numpy()
    df["app_stated_vs_actual_genre_match"] = lookup["app_stated_vs_actual_genre_match"].to_numpy()

    # app_genre_similarity_max: lookup["app_genre_similarity_max"].to_numpy()
    df["app_genre_similarity_max"] = lookup["app_genre_similarity_max"].to_numpy()

    # app_genre_similarity_mean: lookup["app_genre_similarity_mean"].to_numpy()
    df["app_genre_similarity_mean"] = lookup["app_genre_similarity_mean"].to_numpy()

    # app_genre_similarity_weighted: lookup["app_genre_similarity_weighted"].to_numpy()
    df["app_genre_similarity_weighted"] = lookup["app_genre_similarity_weighted"].to_numpy()

    # app_user_artist_cosine: lookup["app_user_artist_cosine"].to_numpy()
    df["app_user_artist_cosine"] = lookup["app_user_artist_cosine"].to_numpy()

    # app_user_artist_euclidean_distance: lookup["app_user_artist_euclidean_distance"].to_numpy()
    df["app_user_artist_euclidean_distance"] = lookup["app_user_artist_euclidean_distance"].to_numpy()

    # app_user_artist_js_distance: lookup["app_user_artist_js_distance"].to_numpy()
    df["app_user_artist_js_distance"] = lookup["app_user_artist_js_distance"].to_numpy()

    # ticket_price_rank_within_user: df.groupby("user_id")["ticket_price"].rank(method="average",ascending=True)
    df["ticket_price_rank_within_user"] = df.groupby("user_id")["ticket_price"].rank(method="average",ascending=True)

    # distance_km_rank_within_user: df.groupby("user_id")["distance_km"].rank(method="average",ascending=True)
    df["distance_km_rank_within_user"] = df.groupby("user_id")["distance_km"].rank(method="average",ascending=True)

    # ticket_price_minus_user_offer_mean_difference: df["ticket_price"] - df.groupby("user_id")["ticket_price"].transform("mean")
    df["ticket_price_minus_user_offer_mean_difference"] = df["ticket_price"] - df.groupby("user_id")["ticket_price"].transform("mean")

    # genre_match_rank_within_user: df.groupby("user_id")["genre_match"].rank(method="average",ascending=False)
    df["genre_match_rank_within_user"] = df.groupby("user_id")["genre_match"].rank(method="average",ascending=False)

    # app_artist_play_share_rank_within_user: df.groupby("user_id")["app_artist_play_share"].rank(method="average",ascending=False)
    df["app_artist_play_share_rank_within_user"] = df.groupby("user_id")["app_artist_play_share"].rank(method="average",ascending=False)

    # wishlist_count_within_user: df.groupby("user_id")["wishlisted"].transform("sum")
    df["wishlist_count_within_user"] = df.groupby("user_id")["wishlisted"].transform("sum")

    # event_page_visits_7d_share_within_user: safe_div(df["event_page_visits_7d"],df.groupby("user_id")["event_page_visits_7d"].transform("sum"))
    df["event_page_visits_7d_share_within_user"] = safe_div(df["event_page_visits_7d"],df.groupby("user_id")["event_page_visits_7d"].transform("sum"))

    # concert_id_x_age_group: combine(df["concert_id"],(df["age"]//10)*10)
    df["concert_id_x_age_group"] = combine(df["concert_id"],(df["age"]//10)*10)

    # concert_id_x_spending_capacity_band: combine(df["concert_id"],df["spending_capacity_band"])
    df["concert_id_x_spending_capacity_band"] = combine(df["concert_id"],df["spending_capacity_band"])

    # past_purchase_count_x_genre_match: df["past_purchase_count"] * df["genre_match"]
    df["past_purchase_count_x_genre_match"] = df["past_purchase_count"] * df["genre_match"]

    # first_timer_x_genre_match: df["first_timer"] * df["genre_match"]
    df["first_timer_x_genre_match"] = df["first_timer"] * df["genre_match"]

    # first_timer_x_distance_km: df["first_timer"] * df["distance_km"]
    df["first_timer_x_distance_km"] = df["first_timer"] * df["distance_km"]

    # total_price: df["ticket_price"] + df["booking_fee"] - df["discount_amount"]
    df["total_price"] = df["ticket_price"] + df["booking_fee"] - df["discount_amount"]

    # discount_amount_x_first_timer: df["discount_amount"] * df["first_timer"]
    df["discount_amount_x_first_timer"] = df["discount_amount"] * df["first_timer"]

    # discount_amount_x_spending_capacity_band: df["discount_amount"] * df["spending_capacity_band"]
    df["discount_amount_x_spending_capacity_band"] = df["discount_amount"] * df["spending_capacity_band"]

    # discount_amount_x_ticket_price: df["discount_amount"] * df["ticket_price"]
    df["discount_amount_x_ticket_price"] = df["discount_amount"] * df["ticket_price"]

    # discount_amount_x_wishlisted: df["discount_amount"] * df["wishlisted"]
    df["discount_amount_x_wishlisted"] = df["discount_amount"] * df["wishlisted"]

    # discount_amount_x_event_page_visits_7d: df["discount_amount"] * df["event_page_visits_7d"]
    df["discount_amount_x_event_page_visits_7d"] = df["discount_amount"] * df["event_page_visits_7d"]

    # distance_km_squared: df["distance_km"] ** 2
    df["distance_km_squared"] = df["distance_km"] ** 2

    # log_distance_km: np.log1p(df["distance_km"])
    df["log_distance_km"] = np.log1p(df["distance_km"])

    # distance_km_x_ticket_price: df["distance_km"] * df["ticket_price"]
    df["distance_km_x_ticket_price"] = df["distance_km"] * df["ticket_price"]

    # distance_km_x_spending_capacity_band: df["distance_km"] * df["spending_capacity_band"]
    df["distance_km_x_spending_capacity_band"] = df["distance_km"] * df["spending_capacity_band"]

    # distance_km_x_age: df["distance_km"] * df["age"]
    df["distance_km_x_age"] = df["distance_km"] * df["age"]

    # distance_km_x_genre_match: df["distance_km"] * df["genre_match"]
    df["distance_km_x_genre_match"] = df["distance_km"] * df["genre_match"]

    # distance_km_x_wishlisted: df["distance_km"] * df["wishlisted"]
    df["distance_km_x_wishlisted"] = df["distance_km"] * df["wishlisted"]

    # travel_arrival_before_concert: np.where(df["travel_search_arrival"].notna(),(df["concert_at_minus_travel_search_arrival_difference_days"] >= 0).astype(int),np.nan)
    df["travel_arrival_before_concert"] = np.where(df["travel_search_arrival"].notna(),(df["concert_at_minus_travel_search_arrival_difference_days"] >= 0).astype(int),np.nan)

    # travel_arrival_within_1_days: np.where(df["travel_search_arrival"].notna(),df["concert_at_minus_travel_search_arrival_difference_days"].between(0,1).astype(int),np.nan)
    df["travel_arrival_within_1_days"] = np.where(df["travel_search_arrival"].notna(),df["concert_at_minus_travel_search_arrival_difference_days"].between(0,1).astype(int),np.nan)

    # travel_arrival_within_3_days: np.where(df["travel_search_arrival"].notna(),df["concert_at_minus_travel_search_arrival_difference_days"].between(0,3).astype(int),np.nan)
    df["travel_arrival_within_3_days"] = np.where(df["travel_search_arrival"].notna(),df["concert_at_minus_travel_search_arrival_difference_days"].between(0,3).astype(int),np.nan)

    # travel_search_present_x_distance_km: df["travel_search_present"] * df["distance_km"]
    df["travel_search_present_x_distance_km"] = df["travel_search_present"] * df["distance_km"]

    # days_until_concert_squared: df["days_until_concert"] ** 2
    df["days_until_concert_squared"] = df["days_until_concert"] ** 2

    # days_until_concert_minus_usual_booking_lead_days_difference: df["days_until_concert"] - df["usual_booking_lead_days"]
    df["days_until_concert_minus_usual_booking_lead_days_difference"] = df["days_until_concert"] - df["usual_booking_lead_days"]

    # days_until_concert_to_usual_booking_lead_days_ratio: safe_div(df["days_until_concert"],df["usual_booking_lead_days"])
    df["days_until_concert_to_usual_booking_lead_days_ratio"] = safe_div(df["days_until_concert"],df["usual_booking_lead_days"])

    # is_weekend_concert: df["concert_weekday"].isin([5,6]).astype(int)
    df["is_weekend_concert"] = df["concert_weekday"].isin([5,6]).astype(int)

    # concert_month_sin: np.sin(2*np.pi*df["concert_month"]/12)
    df["concert_month_sin"] = np.sin(2*np.pi*df["concert_month"]/12)

    # concert_month_cos: np.cos(2*np.pi*df["concert_month"]/12)
    df["concert_month_cos"] = np.cos(2*np.pi*df["concert_month"]/12)

    # concert_weekday_sin: np.sin(2*np.pi*df["concert_weekday"]/7)
    df["concert_weekday_sin"] = np.sin(2*np.pi*df["concert_weekday"]/7)

    # concert_weekday_cos: np.cos(2*np.pi*df["concert_weekday"]/7)
    df["concert_weekday_cos"] = np.cos(2*np.pi*df["concert_weekday"]/7)

    # days_until_concert_x_distance_km: df["days_until_concert"] * df["distance_km"]
    df["days_until_concert_x_distance_km"] = df["days_until_concert"] * df["distance_km"]

    # days_until_concert_x_wishlisted: df["days_until_concert"] * df["wishlisted"]
    df["days_until_concert_x_wishlisted"] = df["days_until_concert"] * df["wishlisted"]

    # event_page_visits_7d_x_days_until_concert: df["event_page_visits_7d"] * df["days_until_concert"]
    df["event_page_visits_7d_x_days_until_concert"] = df["event_page_visits_7d"] * df["days_until_concert"]

    # music_app_user_id_missing_x_genre_match: df["music_app_user_id_missing"] * df["genre_match"]
    df["music_app_user_id_missing_x_genre_match"] = df["music_app_user_id_missing"] * df["genre_match"]

    # age_x_genre_match: df["age"] * df["genre_match"]
    df["age_x_genre_match"] = df["age"] * df["genre_match"]

    # genre_match_x_wishlisted: df["genre_match"] * df["wishlisted"]
    df["genre_match_x_wishlisted"] = df["genre_match"] * df["wishlisted"]

    # first_timer_x_app_artist_play_share: df["first_timer"] * df["app_artist_play_share"]
    df["first_timer_x_app_artist_play_share"] = df["first_timer"] * df["app_artist_play_share"]

    # distance_km_minus_user_offer_mean_difference: df["distance_km"] - df.groupby("user_id")["distance_km"].transform("mean")
    df["distance_km_minus_user_offer_mean_difference"] = df["distance_km"] - df.groupby("user_id")["distance_km"].transform("mean")

    # total_price_squared: df["total_price"] ** 2
    df["total_price_squared"] = df["total_price"] ** 2

    # total_price_to_typical_ticket_spend_ratio: safe_div(df["total_price"],df["typical_ticket_spend"])
    df["total_price_to_typical_ticket_spend_ratio"] = safe_div(df["total_price"],df["typical_ticket_spend"])

    # absolute_days_until_concert_minus_usual_booking_lead_days_difference: df["days_until_concert_minus_usual_booking_lead_days_difference"].abs()
    df["absolute_days_until_concert_minus_usual_booking_lead_days_difference"] = df["days_until_concert_minus_usual_booking_lead_days_difference"].abs()

    # wishlisted_x_total_price_to_typical_ticket_spend_ratio: df["wishlisted"] * df["total_price_to_typical_ticket_spend_ratio"]
    df["wishlisted_x_total_price_to_typical_ticket_spend_ratio"] = df["wishlisted"] * df["total_price_to_typical_ticket_spend_ratio"]
    return df

# Function that checks that all our features are complete and valid
def check_feature_integrity(train_features, test_features):

    # Save all feature names in train columns (Buy ommited since not in test)
    train_columns=set(train_features.columns) - {'buy'}

    # Get all column names from test dataset
    test_columns=set(test_features.columns)

    # Error if they are not the same. Raise AssertionError
    if train_columns != test_columns:
        raise AssertionError(f'Train-only: {train_columns-test_columns}; test-only: {test_columns-train_columns}')

    # Check if we have duplicate rows and raise assertionerror
    if train_features['row_id'].duplicated().any() or test_features['row_id'].duplicated().any():
        raise AssertionError('Duplicate row_id found')

    
    return True