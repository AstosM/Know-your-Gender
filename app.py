import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# -----------------------------
# Utility functions
# -----------------------------
def load_default_data():
    """Load default CSV shipped with the app."""
    df = pd.read_csv("Names_dataset.csv")
    df = df.copy()
    df["gender"] = df["gender"].replace({"f": 0, "m": 1})
    return df


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    # Expect columns: name, gender (0/1 or f/m)
    if df["gender"].dtype == object:
        df["gender"] = df["gender"].replace({"f": 0, "m": 1}).astype(int)
    return df[["name", "gender"]].dropna()


def train_model(df: pd.DataFrame):
    """Always use Naive Bayes (no model selection)."""
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


def predict_single(name: str, clf, cv):
    if not name.strip():
        return None, None, None
    vec = cv.transform([name]).toarray()
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vec)[0]
        pred = int(np.argmax(proba))
    else:
        pred = int(clf.predict(vec)[0])
    label = "Female" if pred == 0 else "Male"
    return label, proba, pred


def extract_name_from_sentence(text: str) -> str:
    """(Old behavior) Naive extraction: return last capitalized word, else last word."""
    if not text.strip():
        return ""
    tokens = text.strip().split()
    caps = [t for t in tokens if t[0].isalpha() and t[0].isupper()]
    if caps:
        return caps[-1]
    return tokens[-1]


def extract_names_from_sentence(text: str):
    """
    NEW: Extract multiple possible names from a sentence.

    Heuristics:
    - Words starting with a capital letter (Ram, Sheela)
    - Words that come right after 'named', 'name', 'called' (to catch lowercase like 'ashish')
    """
    import re

    if not text.strip():
        return []

    words = re.findall(r"[A-Za-z]+", text)
    names = []
    lower_words = [w.lower() for w in words]

    for i, w in enumerate(words):
        # Capitalized words are likely names
        if w[0].isupper():
            names.append(w)

        # Words after certain triggers are treated as names even if lowercase
        if i > 0 and lower_words[i - 1] in {"named", "name", "called"}:
            names.append(w)

    # Deduplicate (case-insensitive) while preserving order
    seen = set()
    unique_names = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique_names.append(n)

    return unique_names


def get_nicknames(name: str, gender_label: str):
    """Simple hardcoded nickname suggestions."""
    base = name.strip()
    if not base:
        return []
    suggestions = set()
    if len(base) > 3:
        suggestions.add(base[:3])
        suggestions.add(base[:4])
    if base.endswith("a"):
        suggestions.add(base[:-1])
    if gender_label == "Male":
        suggestions.update([base + " bhai", "Mr. " + base])
    else:
        suggestions.update([base + " di", "Ms. " + base])
    return list(suggestions)


def similar_names(df: pd.DataFrame, name: str, gender_int: int, topn: int = 5):
    """Suggest similar names by prefix."""
    if not name:
        return []
    prefix = name[:2].lower()
    subset = df[df["gender"] == gender_int]
    similar = subset[subset["name"].str.lower().str.startswith(prefix)]["name"].unique()
    similar = [n for n in similar if n.lower() != name.lower()]
    return list(similar[:topn])


def letter_frequency(df: pd.DataFrame):
    """Compute letter frequency per gender."""
    rows = []
    for _, row in df.iterrows():
        name = str(row["name"]).lower()
        g = row["gender"]
        for ch in name:
            if ch.isalpha():
                rows.append((ch, g))
    if not rows:
        return pd.DataFrame(columns=["letter", "gender", "count"])
    lf = pd.DataFrame(rows, columns=["letter", "gender"])
    lf["count"] = 1
    lf = lf.groupby(["letter", "gender"])["count"].sum().reset_index()
    return lf


def make_wordcloud(text_series, title):
    text = " ".join(map(str, text_series))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="Gender Prediction", page_icon="🧬", layout="wide")

    # Theme (light/dark)
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    st.session_state["theme"] = theme

    bg_color = "#0b0c10" if theme == "Dark" else "#f5f7fb"
    text_color = "#f5f7fb" if theme == "Dark" else "#222222"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .main {{
            background-color: {bg_color};
            color: {text_color};
        }}
        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .animated-header {{
            background: linear-gradient(-45deg, #4A90E2, #9013FE, #50E3C2, #B8E986);
            background-size: 400% 400%;
            animation: gradient 12s ease infinite;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="animated-header">
            <h2 style="color:white;text-align:center;margin:0;">
                🧬 Gender Classification by Name (Know Your Gender)
            </h2>
            <p style="color:#f0f0f0;text-align:center;margin:5px 0 0 0;">
                Predict gender, explore name statistics, and play with ML features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Predict Single Name",
            "Batch Prediction",
            "Data Insights",
            "About",
        ],
    )

    # Load base data once
    if "base_df" not in st.session_state:
        st.session_state["base_df"] = prepare_data(load_default_data())

    df = st.session_state["base_df"]

    # Track metrics over time
    if "metrics_history" not in st.session_state:
        st.session_state["metrics_history"] = []

    # Train model with simple progress bar (Naive Bayes only)
    with st.spinner("Training Naive Bayes model..."):
        progress = st.progress(0)
        progress.progress(20)
        clf, cv, train_acc, test_acc = train_model(df)
        progress.progress(100)

    # Save current metrics
    st.session_state["metrics_history"].append(
        {"train_acc": train_acc, "test_acc": test_acc}
    )

    # Shared performance section
    def show_performance():
        st.subheader("🎯 Model Performance (Naive Bayes)")
        c1, c2 = st.columns(2)
        c1.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        c2.metric("Test Accuracy", f"{test_acc*100:.2f}%")

        hist = pd.DataFrame(st.session_state["metrics_history"])
        if len(hist) > 1:
            st.write("Accuracy over this session (re-trains):")
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(hist["train_acc"], marker="o", label="Train")
            ax.plot(hist["test_acc"], marker="o", label="Test")
            ax.set_xlabel("Training run ")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)

    # -----------------------------
    # Page: Predict Single Name
    # -----------------------------
    if page == "Predict Single Name":
        show_performance()
        st.markdown("---")

        st.subheader("🔮 Single Name / Sentence Prediction")

        name_mode = st.radio(
            "How do you want to input?",
            ["Type a name", "Use a sentence with names"],
            horizontal=True,
        )

        # ---- Mode 1: Single name (unchanged) ----
        if name_mode == "Type a name":
            name = st.text_input("Name:")

            if st.button("Predict", key="predict_single"):
                label, proba, pred_int = predict_single(name, clf, cv)
                if label is None:
                    st.warning("Please enter a valid name.")
                else:
                    if label == "Male":
                        card_color = "#d1ecf1"
                        card_text = "#0c5460"
                        accent = "#004085"
                        icon = "♂️"
                    else:
                        card_color = "#f8d7da"
                        card_text = "#721c24"
                        accent = "#721c24"
                        icon = "♀️"

                    st.markdown(
                        f"""
                        <div style="background-color:{card_color};color:{card_text};
                                    padding:18px;border-radius:12px;margin-top:15px;">
                            <h3 style="margin:0;">Prediction: 
                                <span style="color:{accent};">{label} {icon}</span>
                            </h3>
                            <p style="margin:5px 0 0 0;"><strong>Name:</strong> {name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Probabilities
                    if proba is not None and len(proba) == 2:
                        st.write("**Prediction probabilities:**")
                        prob_df = pd.DataFrame(
                            {
                                "Gender": ["Female (0)", "Male (1)"],
                                "Probability": [proba[0], proba[1]],
                            }
                        )
                        st.bar_chart(prob_df.set_index("Gender"))

                        st.info(
                            f"Model confidence: **{max(proba)*100:.2f}%** "
                            f"for **{label}**."
                        )

                    # Nicknames & similar names
                    nicknames = get_nicknames(name, label)
                    if nicknames:
                        st.write("**Nickname suggestions:** ", ", ".join(nicknames))

                    sims = similar_names(df, name, pred_int)
                    if sims:
                        st.write("**Similar names in dataset:** ", ", ".join(sims))

                    # Meaning / origin placeholder
                    with st.expander("📖 Name meaning & origin (placeholder)"):
                        st.write(
                            "To enable this, connect a name dictionary dataset and "
                            "look up meanings/origins here."
                        )

        # ---- Mode 2: Sentence with multiple names (NEW) ----
        else:
            sentence = st.text_input(
                "Sentence:",
                value="Ram is the friend of Sheela's brother named ashish",
            )

            # Show what *single* name the old extractor would pick (optional)
            if sentence:
                single_guess = extract_name_from_sentence(sentence)
                st.caption(f"(Single-name guess from old logic: **{single_guess}**)")

            if st.button("Detect & Predict from Sentence", key="predict_sentence"):
                names = extract_names_from_sentence(sentence)
                if not names:
                    st.warning("No names detected. Try a sentence with clearer names.")
                else:
                    st.info(f"Detected names: **{', '.join(names)}**")

                    vecs = cv.transform(names).toarray()
                    preds = clf.predict(vecs)
                    if hasattr(clf, "predict_proba"):
                        probas = clf.predict_proba(vecs)
                    else:
                        probas = None

                    results_df = pd.DataFrame(
                        {
                            "name": names,
                            "predicted_gender": np.where(preds == 0, "Female", "Male"),
                            "label": preds,
                        }
                    )
                    if probas is not None:
                        results_df["prob_female"] = probas[:, 0]
                        results_df["prob_male"] = probas[:, 1]

                    st.write("### Predictions for detected names")
                    st.dataframe(results_df, use_container_width=True)

    # -----------------------------
    # Page: Batch Prediction
    # -----------------------------
    elif page == "Batch Prediction":
        show_performance()
        st.markdown("---")

        st.subheader("📦 Predict for Multiple Names (Text Input Only)")
        names_block = st.text_area(
            "Write one name per line:",
            value="Ram\nSheela\nashish\nAmit\nPriya",
            height=200,
        )

        if st.button("Predict (Text Area)", key="predict_batch_text"):
            names = [x.strip() for x in names_block.split("\n") if x.strip()]
            if not names:
                st.warning("Enter at least one name.")
            else:
                vecs = cv.transform(names).toarray()
                preds = clf.predict(vecs)
                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba(vecs)
                else:
                    probas = None

                results_df = pd.DataFrame(
                    {
                        "name": names,
                        "predicted_gender": np.where(preds == 0, "Female", "Male"),
                        "label": preds,
                    }
                )
                if probas is not None:
                    results_df["prob_female"] = probas[:, 0]
                    results_df["prob_male"] = probas[:, 1]

                st.dataframe(results_df, use_container_width=True)

    # -----------------------------
    # Page: Data Insights
    # -----------------------------
    elif page == "Data Insights":
        st.subheader("📊 Data Insights & Visualizations")

        st.write(f"Current dataset shape: **{df.shape}**")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class counts:**")
            st.bar_chart(df["gender"].replace({0: "Female", 1: "Male"}).value_counts())
        with col2:
            st.write("**Top 10 most frequent names:**")
            st.write(df["name"].value_counts().head(10))

        # Names that appear as both genders
        st.markdown("### 🔁 Unisex / ambiguous names")
        gender_counts = df.groupby("name")["gender"].nunique()
        unisex_names = gender_counts[gender_counts > 1].index.tolist()
        st.write(f"Found **{len(unisex_names)}** names used for both genders.")
        if unisex_names:
            st.write(unisex_names[:50])

        # Wordclouds
        st.markdown("### ☁️ WordClouds")
        if not WORDCLOUD_AVAILABLE:
            st.warning(
                "Install `wordcloud` package to see wordclouds: `pip install wordcloud`."
            )
        else:
            f_names = df[df["gender"] == 0]["name"].head(5000)
            m_names = df[df["gender"] == 1]["name"].head(5000)
            colw1, colw2 = st.columns(2)
            with colw1:
                st.write("Female names")
                fig_f = make_wordcloud(f_names, "Female Names")
                st.pyplot(fig_f)
            with colw2:
                st.write("Male names")
                fig_m = make_wordcloud(m_names, "Male Names")
                st.pyplot(fig_m)

        # Letter frequency heatmap
        st.markdown("### 🔤 Letter frequency by gender")
        lf = letter_frequency(df)
        if not lf.empty:
            pivot = lf.pivot_table(
                index="letter", columns="gender", values="count", fill_value=0
            )
            pivot = pivot.sort_index()
            st.write("Counts table:")
            st.dataframe(pivot)

            fig, ax = plt.subplots()
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(["Female(0)", "Male(1)"])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Letter frequency by gender")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No letters found to compute frequencies.")

        # Popularity graph placeholder
        st.markdown("### 📈 Popularity over years (placeholder)")
        st.write(
            "To enable this, you would need a dataset with a 'year' column and "
            "name frequencies per year, then plot trends here."
        )

    # -----------------------------
    # Page: About
    # -----------------------------
    elif page == "About":
        st.subheader("ℹ️ About This App")
        st.write(
            """
            This interactive app demonstrates **gender prediction from names** using
            a Naive Bayes classifier.
            """
        )
        st.markdown(
            """
            **Features included:**
            - Light/Dark theme toggle  
            - Animated gradient header  
            - Single-name prediction  
            - NEW: sentence mode that detects **multiple names** and predicts each  
            - Batch predictions (multi-line text)  
            - Probability display & confidence (Naive Bayes)  
            - Nickname & similar-name suggestions (single-name mode)  
            - Data insights, wordclouds, and letter frequency heatmap  
            - Session-based accuracy tracking  
            """
        )

        st.markdown(
            """
            <hr>
            <div style="text-align:center; padding:10px; color:#888;">
                Made with ❤️ using Streamlit by AstosM
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
