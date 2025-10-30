
---

# 🎬 Movie Recommendation System Using SVD

A **comprehensive Movie Recommendation System** that combines **Singular Value Decomposition (SVD)** for collaborative filtering and **Random Forest models** for rating prediction.
It also includes an **interactive Streamlit web app** that allows users to explore and get movie recommendations by **genre, user preferences, or ratings**.

---

## 🧠 Project Overview

This project implements an **Advanced Recommendation Engine** for movies using the **MovieLens dataset** (`movies.dat` and `ratings.dat`).
It leverages **dimensionality reduction (SVD)** and **machine learning (Random Forest)** to predict user preferences and generate movie recommendations.

---

## ⚙️ Key Features

### 🎯 Core Python Script

* Builds a **user-movie rating matrix** and computes **SVD**.
* Analyzes **movie similarity** using cosine similarity.
* Trains:

  * **Random Forest Regressor** → Predicts movie ratings.
  * **Random Forest Classifier** → Predicts high/low rating categories.
* Produces detailed **visualizations**:

  * Explained variance plots
  * Top singular values
  * Similarity heatmaps
  * Rating distributions
  * Feature importance for Random Forest models

### 🌐 Streamlit Web App

An interactive UI for exploring and generating recommendations.

* **Recommendation Modes:**

  * *Genre-based:* Top-rated movies in a chosen genre.
  * *User-based (SVD):* Finds similar users and recommends what they like.
  * *Rating-based:* Shows highest-rated movies overall.
* Adjustable parameters:

  * Genre filter
  * User ID input
  * Number of recommendations
  * Minimum rating threshold
* Displays recommendations with movie title, genre, and average rating.

---

## 🧩 Technologies Used

| Category              | Libraries / Tools                |
| --------------------- | -------------------------------- |
| **Core**              | Python 3.x                       |
| **Data Handling**     | NumPy, Pandas                    |
| **Visualization**     | Matplotlib, Seaborn              |
| **Machine Learning**  | Scikit-learn (SVD, RandomForest) |
| **Web App**           | Streamlit                        |
| **Model Persistence** | Joblib                           |

---

## 📁 Dataset

The project uses the **MovieLens dataset** containing:

* `movies.dat` → Movie IDs, titles, and genres
* `ratings.dat` → User ratings for movies

Each file uses the `::` delimiter.

Example:

```
movies.dat → movie_id::title::genre  
ratings.dat → user_id::movie_id::rating::timestamp
```



---

## 🧮 How It Works

1. **Data Loading & Cleaning**

   * Reads `ratings.dat` and `movies.dat`.
   * Creates a rating matrix (movies × users).
2. **Dimensionality Reduction**

   * Uses **SVD** to reduce dimensions while preserving similarity patterns.
3. **Similarity Computation**

   * Computes cosine similarity between movies to find similar titles.
4. **Model Training**

   * Trains Random Forest models on SVD-reduced features.
5. **Visualization**

   * Generates analytical charts and saves them as PNGs.
6. **Streamlit App**

   * Provides an easy interface for interactive recommendations.

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/vignesh-r-24/Movie-Recommendation-project.git
cd movie-recommendation-project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit joblib
```

### 3️⃣ Place dataset files

Place the following in the project root:

```
movies.dat
ratings.dat
```

### 4️⃣ Run the analysis script

```bash
python movie_recommendation.py
```

### 5️⃣ Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🖥️ Streamlit App Usage

1. Enter **User ID** and **Genre** filter (e.g., "Action").
2. Choose **Recommendation Method**:

   * Genre-based
   * Rating-based
   * User-based (SVD)
3. Click **Get Recommendations**.
4. View top recommended movies with average ratings.

---

## 📊 Output Visualizations

The script generates:

* `movie_recommendation_analysis.png`
* `movie_recommendation_analysis_with_rf.png`

These include:

* Explained variance curve
* Similar movie recommendations
* Rating distribution
* Confusion matrix (for classification)
* Feature importance (for regression)

---

## 🧾 Example Output

### Console Output

```
[1/5] Loading and parsing datasets...
✓ Ratings shape: (1000209, 4)
✓ Movies shape: (3883, 3)
✓ Rating matrix shape: (3883, 6040)
✓ Using 50 principal components
✓ Finding top 10 recommendations for movie_id: 2
```

### Streamlit Output

| Movie ID | Title                             | Genre  | Avg Rating |
| -------- | --------------------------------- | ------ | ---------- |
| 50       | Die Hard (1988)                   | Action | ⭐4.2       |
| 172      | Terminator 2: Judgment Day (1991) | Action | ⭐4.1       |
| 257      | Speed (1994)                      | Action | ⭐3.9       |

---

