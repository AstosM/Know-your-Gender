import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ---------------------------------------------------------
# Load dataset (CSV only)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Names_dataset.csv")
    except FileNotFoundError:
        st.error("❌ 'Names_dataset.csv' not found. Place it in same folder as app.py.")
        st.stop()

    df = df.copy()
    df["gender"] = df["gender"].replace({"f": 0, "m": 1})
    return df


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df["name"].astype(str)
    y = df["gender"]

    cv = CountVectorizer()
    X_vec = cv.fit_transform(X.values.astype("U"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return clf, cv, train_acc, test_acc


# ---------------------------------------------------------
# Predict single name
# ---------------------------------------------------------
def predict_gender(name, clf, cv):
    if not name.strip():
        return "Unknown"
    vec = cv.transform([name]).toarray()
    pred = clf.predict(vec)[0]
    return "Female" if pred == 0 else "Male"


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Gender Prediction from Names", page_icon="🧬")

    st.title("🧬 Gender Classification by Name")
    st.write("Enter a name to predict if it's more likely **Male** or **Female**.")

    # Load dataset
    df = load_data()

    # Train model
    clf, cv, train_acc, test_acc = train_model(df)

    # Show dataset info
    with st.expander("📊 Dataset Overview"):
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Class Counts:", df["gender"].value_counts())

    # Show accuracy
    st.subheader("🎯 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{train_acc*100:.2f}%")
    col2.metric("Test Accuracy", f"{test_acc*100:.2f}%")

    st.markdown("---")

    # Single Name Prediction
    st.subheader("🔮 Predict Gender for a Single Name")
    name = st.text_input("Enter name:", value="Sita")

    if st.button("Predict"):
        result = predict_gender(name, clf, cv)
        if result == "Male":
            st.success(f"**{name}** → **Male** ♂️")
        elif result == "Female":
            st.success(f"**{name}** → **Female** ♀️")
        else:
            st.warning("Please enter a valid name.")

    st.markdown("---")

    # Batch Prediction
    st.subheader("📦 Predict for Multiple Names")
    names_block = st.text_area(
        "Write one name per line:",
        value="Ram\nSita\nAmit\nPriya",
        height=150
    )

    if st.button("Predict Batch"):
        names = [x.strip() for x in names_block.split("\n") if x.strip()]
        if not names:
            st.warning("Enter at least one name.")
        else:
            vecs = cv.transform(names).toarray()
            preds = clf.predict(vecs)
            results = pd.DataFrame({
                "name": names,
                "predicted_gender": np.where(preds == 0, "Female", "Male")
            })
            st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
