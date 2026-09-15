import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="CERN Collision AI Explorer",
    page_icon="⚛️",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("⚛️ CERN Collision AI Explorer")

st.write(
    """
    An interactive Physics + Artificial Intelligence project
    using CERN dielectron collision data.

    This project studies particle energy, momentum and invariant mass
    and demonstrates how Symbolic Regression can rediscover a known
    physics relationship.
    """
)


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload dielectron.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload the CERN dielectron.csv file from the sidebar.")
    st.stop()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    return df


df = load_data(uploaded_file)


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "E1", "E2",
    "px1", "px2",
    "py1", "py2",
    "pz1", "pz2",
    "M"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "These required columns are missing: "
        + ", ".join(missing_columns)
    )

    st.write("Columns found in the CSV:")
    st.write(list(df.columns))

    st.stop()


# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------

data = df[required_columns].copy()

data = data.dropna()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

data["E_total"] = (
    data["E1"]
    + data["E2"]
)

data["px_total"] = (
    data["px1"]
    + data["px2"]
)

data["py_total"] = (
    data["py1"]
    + data["py2"]
)

data["pz_total"] = (
    data["pz1"]
    + data["pz2"]
)


# Dataset invariant mass squared

data["M_squared"] = (
    data["M"] ** 2
)


# ---------------------------------------------------------
# PHYSICS EQUATION
# ---------------------------------------------------------

data["M_squared_physics"] = (

    data["E_total"] ** 2

    - data["px_total"] ** 2

    - data["py_total"] ** 2

    - data["pz_total"] ** 2
)


# Remove tiny negative numerical values before sqrt

data["M_physics"] = np.sqrt(
    np.clip(
        data["M_squared_physics"],
        0,
        None
    )
)


# Difference

data["difference"] = np.abs(

    data["M_squared"]

    - data["M_squared_physics"]
)


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Home",
        "📊 Data Explorer",
        "⚛️ Physics Analysis",
        "🤖 PySR Discovery",
        "🧠 Machine Learning",
        "📈 Final Results"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("Project Overview")

    st.write(
        """
        Particle collisions generate large amounts of data.

        In this project we use CERN dielectron collision events
        to investigate whether Artificial Intelligence can learn
        relationships between particle energy, momentum and mass.
        """
    )

    st.subheader("Main Question")

    st.info(
        """
        Can Symbolic Regression rediscover the invariant-mass
        relationship from CERN collision data?
        """
    )

    st.subheader("Project Pipeline")

    st.code(
        """
CERN Dataset
      ↓
Data Cleaning
      ↓
Physics Features
      ↓
Energy & Momentum Analysis
      ↓
Invariant Mass Calculation
      ↓
Symbolic Regression
      ↓
AI Discovers Equation
      ↓
Machine Learning Comparison
      ↓
Graphs + Results
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Events",
        f"{len(df):,}"
    )

    col2.metric(
        "Usable Events",
        f"{len(data):,}"
    )

    col3.metric(
        "Variables",
        len(df.columns)
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "📊 Data Explorer":

    st.header("CERN Dataset Explorer")

    st.subheader("First Rows")

    st.dataframe(
        data.head(100),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(data):,}"
    )

    col2.metric(
        "Columns",
        len(data.columns)
    )

    col3.metric(
        "Missing Values",
        int(data.isnull().sum().sum())
    )


    st.subheader("Available Variables")

    st.write(list(data.columns))


    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )


# =========================================================
# PHYSICS ANALYSIS
# =========================================================

elif page == "⚛️ Physics Analysis":

    st.header("Physics Analysis")

    st.write(
        """
        Each event contains two electrons.

        Their energies and momenta are combined to calculate
        the invariant mass of the electron pair.
        """
    )


    st.subheader("Invariant Mass Equation")

    st.latex(
        r"""
        M^2 =
        E^2
        -
        p_x^2
        -
        p_y^2
        -
        p_z^2
        """
    )


    # -----------------------------------------------------
    # MASS HISTOGRAM
    # -----------------------------------------------------

    st.subheader("Invariant Mass Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        data["M"],
        bins=80
    )

    ax.set_xlabel("Invariant Mass M")
    ax.set_ylabel("Number of Events")
    ax.set_title("CERN Dielectron Invariant Mass Distribution")

    st.pyplot(fig)


    # -----------------------------------------------------
    # ENERGY DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Total Energy Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        data["E_total"],
        bins=80
    )

    ax.set_xlabel("Total Energy")
    ax.set_ylabel("Number of Events")
    ax.set_title("Total Energy of Electron Pairs")

    st.pyplot(fig)


    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    st.subheader("Momentum Components")

    momentum_choice = st.selectbox(
        "Choose momentum component",
        [
            "px_total",
            "py_total",
            "pz_total"
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        data[momentum_choice],
        bins=80
    )

    ax.set_xlabel(momentum_choice)
    ax.set_ylabel("Events")

    st.pyplot(fig)


    # -----------------------------------------------------
    # PHYSICS ERROR
    # -----------------------------------------------------

    st.subheader("Dataset Mass vs Physics Formula")

    average_difference = (
        data["difference"].mean()
    )

    maximum_difference = (
        data["difference"].max()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Average |M² difference|",
        f"{average_difference:.6f}"
    )

    col2.metric(
        "Maximum difference",
        f"{maximum_difference:.6f}"
    )


# =========================================================
# PYSR
# =========================================================

elif page == "🤖 PySR Discovery":

    st.header("Symbolic Regression — PySR")

    st.write(
        """
        Instead of giving the computer the physics equation,
        Symbolic Regression searches for mathematical expressions
        that explain the data.
        """
    )

    st.subheader("Inputs Given to PySR")

    st.code(
        """
E_total
px_total
py_total
pz_total
        """
    )

    st.write("Target:")

    st.code(
        """
M_squared
        """
    )


    st.subheader("Equation Discovered by PySR")

    st.success(
        """
        PySR successfully recovered the invariant-mass structure.
        """
    )

    st.latex(
        r"""
        M^2 =
        E_{total}^2
        -
        p_{x,total}^2
        -
        p_{y,total}^2
        -
        p_{z,total}^2
        """
    )


    st.subheader("PySR Raw Equation")

    st.code(
        """
(E_total * E_total)
-
(
    (pz_total * pz_total)
    +
    (px_total * px_total)
    +
    (py_total * py_total)
)
        """
    )


    st.subheader("Compare with Known Physics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Known Physics")

        st.latex(
            r"""
            M^2 =
            E^2
            -
            p_x^2
            -
            p_y^2
            -
            p_z^2
            """
        )

    with col2:

        st.markdown("### AI Result")

        st.latex(
            r"""
            M^2 =
            E^2
            -
            p_x^2
            -
            p_y^2
            -
            p_z^2
            """
        )


    st.success(
        "The symbolic regression result has the same mathematical structure as the known invariant-mass equation."
    )


    # -----------------------------------------------------
    # ACTUAL VS PYSR
    # -----------------------------------------------------

    st.subheader("Actual vs PySR Prediction")

    sample_plot = data.sample(
        min(5000, len(data)),
        random_state=42
    )

    actual = sample_plot["M_squared"]

    predicted = sample_plot["M_squared_physics"]


    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        actual,
        predicted,
        alpha=0.3
    )

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    ax.set_xlabel("Actual CERN M²")

    ax.set_ylabel(
        "PySR Equation Predicted M²"
    )

    ax.set_title(
        "Actual vs Symbolic Regression Prediction"
    )

    st.pyplot(fig)


    pysr_mae = mean_absolute_error(
        actual,
        predicted
    )

    pysr_r2 = r2_score(
        actual,
        predicted
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{pysr_mae:.6f}"
    )

    col2.metric(
        "R² Score",
        f"{pysr_r2:.6f}"
    )


# =========================================================
# MACHINE LEARNING
# =========================================================

elif page == "🧠 Machine Learning":

    st.header("Machine Learning Comparison")

    st.write(
        """
        Symbolic Regression produces an understandable equation.

        Now we compare it with a traditional Machine Learning model:
        Random Forest Regression.
        """
    )


    # -----------------------------------------------------
    # SAMPLE SIZE
    # -----------------------------------------------------

    maximum_sample = min(
        20000,
        len(data)
    )

    minimum_sample = min(
        1000,
        maximum_sample
    )

    sample_size = st.slider(

        "Number of events used for ML",

        min_value=minimum_sample,

        max_value=maximum_sample,

        value=min(
            10000,
            maximum_sample
        ),

        step=500
    )


    ml_data = data.sample(
        sample_size,
        random_state=42
    )


    X = ml_data[
        [
            "E_total",
            "px_total",
            "py_total",
            "pz_total"
        ]
    ]

    y = ml_data[
        "M_squared"
    ]


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )


    if st.button("🚀 Train Random Forest"):

        with st.spinner(
            "Training Random Forest..."
        ):

            model = RandomForestRegressor(

                n_estimators=100,

                max_depth=15,

                random_state=42,

                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )


        st.success(
            "Training completed!"
        )


        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        col1, col2 = st.columns(2)

        col1.metric(
            "Random Forest MAE",
            f"{mae:.5f}"
        )

        col2.metric(
            "Random Forest R²",
            f"{r2:.5f}"
        )


        # -------------------------------------------------
        # GRAPH
        # -------------------------------------------------

        st.subheader(
            "Actual vs Random Forest Prediction"
        )

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.scatter(
            y_test,
            predictions,
            alpha=0.3
        )


        minimum = min(
            y_test.min(),
            predictions.min()
        )

        maximum = max(
            y_test.max(),
            predictions.max()
        )


        ax.plot(
            [minimum, maximum],
            [minimum, maximum]
        )


        ax.set_xlabel(
            "Actual M²"
        )

        ax.set_ylabel(
            "Random Forest Predicted M²"
        )

        ax.set_title(
            "Random Forest Prediction"
        )

        st.pyplot(fig)


        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        st.subheader(
            "Feature Importance"
        )

        importance = pd.DataFrame(
            {
                "Feature":
                    X.columns,

                "Importance":
                    model.feature_importances_
            }
        )

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance,
            use_container_width=True
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_ylabel(
            "Importance"
        )

        ax.set_title(
            "Random Forest Feature Importance"
        )

        st.pyplot(fig)


# =========================================================
# FINAL RESULTS
# =========================================================

elif page == "📈 Final Results":

    st.header("Final Project Results")


    physics_mae = mean_absolute_error(

        data["M_squared"],

        data["M_squared_physics"]
    )


    physics_r2 = r2_score(

        data["M_squared"],

        data["M_squared_physics"]
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Events Analysed",
        f"{len(data):,}"
    )

    col2.metric(
        "Equation MAE",
        f"{physics_mae:.6f}"
    )

    col3.metric(
        "Equation R²",
        f"{physics_r2:.6f}"
    )


    st.subheader(
        "Main Discovery"
    )

    st.success(
        """
        Symbolic Regression successfully rediscovered
        the mathematical structure of the invariant-mass
        relationship using CERN collision data.
        """
    )


    st.latex(
        r"""
        M^2 =
        E^2
        -
        p_x^2
        -
        p_y^2
        -
        p_z^2
        """
    )


    st.subheader(
        "Why This Matters"
    )

    st.write(
        """
        A traditional Machine Learning model can learn to predict
        invariant mass, but the internal reasoning of the model
        is difficult to interpret.

        Symbolic Regression is different because it produces an
        explicit mathematical expression.

        This demonstrates how Artificial Intelligence can be used
        not only for prediction, but also for interpretable
        scientific discovery and equation recovery.
        """
    )


    st.subheader(
        "Conclusion"
    )

    st.write(
        """
        CERN collision data was cleaned and analysed using Python.

        Physics features were created from electron energies
        and momentum components.

        The known invariant-mass calculation was verified using
        the dataset.

        Symbolic Regression was then applied to the data and
        recovered the same mathematical structure as the
        invariant-mass relationship.

        Finally, the result can be compared with conventional
        Machine Learning to demonstrate the difference between
        predictive AI and interpretable AI.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "CERN Collision AI Project"
)   