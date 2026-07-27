import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("\n=== eDNA Known vs Unknown System (Confidence-Based) ===")

# -------------------------------------------------
# CLEAN DNA
# -------------------------------------------------

def clean_sequence(seq):
    if pd.isna(seq):
        return ""
    seq = seq.upper()
    return re.sub(r"[^ACGTU]", "", seq)

# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------

df = pd.read_csv("output_10000.csv", header=None)
df.columns = ["sequence", "class", "genus"]

df["sequence"] = df["sequence"].apply(clean_sequence)
df = df[df["sequence"].str.len() > 10]
df["class"] = df["class"].fillna("Unknown")

print("Original dataset:", df.shape)

# -------------------------------------------------
# REMOVE RARE CLASSES
# -------------------------------------------------

counts = df["class"].value_counts()
valid_classes = counts[counts >= 3].index
df = df[df["class"].isin(valid_classes)]

print("After cleaning:", df.shape)

if df["class"].nunique() < 2:
    raise ValueError("Not enough classes to train.")

# -------------------------------------------------
# K-MER FEATURES
# -------------------------------------------------

K = 4

def kmer_counts(seq):
    return Counter(seq[i:i+K] for i in range(len(seq)-K+1))

print("Building k-mer space...")

all_kmers = set()
for seq in df["sequence"]:
    all_kmers.update(kmer_counts(seq))

all_kmers = sorted(all_kmers)

def vectorize(seq):
    counts = kmer_counts(seq)
    return [counts.get(k, 0) for k in all_kmers]

X = np.array([vectorize(seq) for seq in df["sequence"]])

# -------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(df["class"])

# -------------------------------------------------
# SAFE TRAIN TEST SPLIT
# -------------------------------------------------

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Using stratified split.")
except:
    print("Fallback to random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -------------------------------------------------
# XGBOOST TRAINING
# -------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

print("Training XGBoost...")
model.fit(X_train, y_train)

print("Model ready.")

# -------------------------------------------------
# LIVE PREDICTION WITH CONFIDENCE
# -------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7

while True:

    user_seq = input("\nEnter DNA (or exit): ")

    if user_seq.lower() == "exit":
        break

    user_seq = clean_sequence(user_seq)

    if len(user_seq) < 4:
        print("Too short.")
        continue

    vec = np.array([vectorize(user_seq)])

    probs = model.predict_proba(vec)[0]
    confidence = max(probs)

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"UNKNOWN (confidence {confidence:.2f})")
        continue

    pred = model.predict(vec)
    label = encoder.inverse_transform(pred)[0]

    print(f"KNOWN: {label} (confidence {confidence:.2f})")

print("\nSystem finished.")
