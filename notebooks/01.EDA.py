import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# read train data
user_concert_train = pd.read_csv('train.csv')
user_concert_train.columns.to_list()

# read test data
user_concert_test = pd.read_csv('test.csv')
user_concert_test.columns.to_list()


sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.columns.to_list()
sample_submission.head()


# EDA 
print(user_concert_train.shape) # 30k observations, 32 columns
print(user_concert_test.shape)  # 20k observations, 31 columns

#glimpse of data
user_concert_train.head()
user_concert_test.head()


# describe data
user_concert_train.buy.value_counts() 
# class imbalance 2:1 on target data, hat’s a mild imbalance, nothing crazy.. lets hope the eval criteria in kaggle is good for this


#info
user_concert_train.info() 



# find missing values
user_concert_train.isnull().sum()  # 7 columsn with gaps, check why they are missing

#1. music_app_user_id
# count the percentage of rows that buy, grouped by non music app/music app user (True = no music app)
user_concert_train.groupby(user_concert_train["music_app_user_id"].isna())["buy"].mean()
# Missing music app id isn't random, no-account users buy 49% vs 33% with an account, so I keep it as a flag
user_concert_train["has_music_app"] = user_concert_train["music_app_user_id"].notna().astype(int)
user_concert_test["has_music_app"]  = user_concert_test["music_app_user_id"].notna().astype(int)

# 2. stated_favourite_genre
user_concert_train.groupby(user_concert_train["stated_favourite_genre"].isna())["buy"].mean()
# Missing genre is random: buy rate 34.6% when missing vs 33.4% when given, so no flag needed,
# instead of omitting, ill replace with unknown because we dont know its strength as a predictor variable
user_concert_train["stated_favourite_genre"] = user_concert_train["stated_favourite_genre"].fillna("unknown")
user_concert_test["stated_favourite_genre"]  = user_concert_test["stated_favourite_genre"].fillna("unknown")

# 3. event_page_visits_7d
user_concert_train.groupby(user_concert_train["event_page_visits_7d"].isna())["buy"].mean()
# Small gap: 30.8% when missing vs 33.7% -> keep a flag, fill with 0
user_concert_train["visits_missing"] = user_concert_train["event_page_visits_7d"].isna().astype(int)
user_concert_test["visits_missing"]  = user_concert_test["event_page_visits_7d"].isna().astype(int)
user_concert_train["event_page_visits_7d"] = user_concert_train["event_page_visits_7d"].fillna(0)
user_concert_test["event_page_visits_7d"]  = user_concert_test["event_page_visits_7d"].fillna(0)

# 4. travel_search_city / travel_search_arrival
user_concert_train.groupby(user_concert_train["travel_search_city"].isna())["buy"].mean()
# Not random: people who searched travel buy 38.3% vs 32.1% who didn't so keep a flag, fill with "none"
user_concert_train["searched_travel"] = user_concert_train["travel_search_city"].notna().astype(int)
user_concert_test["searched_travel"]  = user_concert_test["travel_search_city"].notna().astype(int)
for c in ["travel_search_city", "travel_search_arrival"]:
    user_concert_train[c] = user_concert_train[c].fillna("none")
    user_concert_test[c]  = user_concert_test[c].fillna("none")

# 5. typical_ticket_spend / usual_booking_lead_days
## filter rows where typical_ticket_spend is missing, then count their first_timer values (only 1s = gaps are all first-timers)
pd.crosstab(user_concert_train["typical_ticket_spend"].isna(), user_concert_train["first_timer"]) #dd
# Not random because missing exactly for first-timers (no past tickets to average),
# first_timer already flags them, so fill with the median (from train, used for test too)
# fill first-timers' gaps with the median (taken from train, applied to both train and test)
for c in ["typical_ticket_spend", "usual_booking_lead_days"]:   # do the same for both columns
    med = user_concert_train[c].median()                         # middle value of that column in train
    user_concert_train[c] = user_concert_train[c].fillna(med)   # replace empty cells in train with it
    user_concert_test[c]  = user_concert_test[c].fillna(med)    # same median for test


#for median above: the gaps are exactly the first-timers, from #dd
#who have no past tickets to average, so the value can't exist for them. 
#Dropping them would lose a quarter of the data, and the test set has first-timers you still need to predict. 
#You use the median and not the mean because spend is right-skewed, and the first timer column already flags those rows.


#moves buy to the end
user_concert_train["buy"] = user_concert_train.pop("buy")

user_concert_train.describe() # typical_ticket_spend, usual_booking_lead_days are suspicious


user_concert_train.nlargest(10, "typical_ticket_spend")[["user_id", "spending_capacity_band", "typical_ticket_spend", "past_purchase_count", "ticket_price", "buy"]] # not errors, but they’re extreme
user_concert_train.nlargest(10, "usual_booking_lead_days")[["user_id", "spending_capacity_band", "typical_ticket_spend", "past_purchase_count","usual_booking_lead_days", "ticket_price", "buy"]] # not errors, but they’re extreme



#sb.pairplot(user_concert_train) # check associations ( too large + I dont understand much)
#leave out the 0/1 flags, the IDs and the lat/lon columns, which only show up as rows of dots.

num_cols = ["age", "spending_capacity_band", "past_purchase_count", "typical_ticket_spend",
            "usual_booking_lead_days", "ticket_price", "booking_fee", "discount_pct",
            "event_page_visits_7d"]

sb.pairplot(
    user_concert_train[num_cols + ["buy"]].sample(5000, random_state=0), #using a random sample to avoid 30k rows
    hue="buy", corner=True, plot_kws={"alpha": 0.3, "s": 10},
)


#a,rm booking fee its an exact copy of ticket price (corr), anything that doesnt show a relationship in the plot wont be ommited. 
#b,The higher someone’s spending capacity band, the more likely they are to buy.
#c,Buyers tend to have a higher typical ticket spend.
#d,Booking fee and ticket price form a perfect line, so drop booking fee.
#e,More event page visits go with more buying.

# booking_fee is an exact copy of ticket_price, ill drop it since correlation is exact
user_concert_train = user_concert_train.drop(columns=["booking_fee"])
user_concert_test = user_concert_test.drop(columns=["booking_fee"])


 # use spearman because some distributions are skewed, using this to support my observations above (a,b,c,d,e)
sb.heatmap(user_concert_train.corr(method="spearman", numeric_only=True), cmap="coolwarm")
plt.show() 

# categorical columns for analysis 
cols = ["home_city", "education", "stated_favourite_genre", "artist_name",
        "venue_city", "travel_search_city", "first_timer", "wishlisted", "discount_pct"]

## 
fig, axes = plt.subplots(3, 3, figsize=(15, 12)) # 3 rows x 3 columns
for col, ax in zip(cols, axes.flat): # loop through columns, and pair each with the box and grid 
     # buyers divided by all rows for that value: ex: filter for bordeaux and count observations with buy =1 over all filtered bordeaux observations
    user_concert_train.groupby(col)["buy"].mean().plot(kind="bar", ax=ax, title=col)
plt.tight_layout()
plt.show()

#Home city: small differences (Rennes about 36%, Strasbourg about 31%), so it's weak on its own.
#- Education: no difference, so it's the weakest column.
#- Favourite genre: pop fans buy most (41%), metal and classical least, so genre match is likely useful.
#- Artist: Taylor Swift highest (39%), Tame Impala lowest (28%).
#- Venue city: the same as artist, since each artist plays one city, so drop it.
#- Travel search: searching for a concert city means about 40% buy, other cities about 30%, the same as not searching.
#- Discount: buying rises with bigger discounts.
#- Wishlisted: 43% versus 32%, the strongest flag.
#- First timers: buy slightly more, 37% versus 32%.



#check missing values ok
user_concert_train.isnull().sum()
user_concert_test.isnull().sum()