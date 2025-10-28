This project builds an **intelligent movie recommendation system** using  
**Singular Value Decomposition (SVD)** and **collaborative filtering**.  
It uses **Streamlit** for an interactive web interface and **Scikit-learn** for machine learning.

Users can explore movies filtered by genre, ratings, and personalized similarity scores in real time.

---

##  Key Features

- 🔹 **Genre-Based Filtering:** Filter and rank movies by genre and ratings.  
- 🔹 **Rating-Based Ranking:** Recommend globally top-rated movies.  
- 🔹 **SVD Collaborative Filtering:** Discover latent patterns using Truncated SVD.  
- 🔹 **Interactive Web UI:** Built with Streamlit for real-time interaction.  
- 🔹 **Optimized Performance:** Cached computation for sub-second response time.

---

## 🗂️ Dataset


**🎞️ MovieLens 1M Dataset**

| File | Columns | Description |
|------|----------|-------------|
| `ratings.dat` | user_id, movie_id, rating, timestamp | User movie ratings |
| `movies.dat` | movie_id, title, genre | Movie details and genres |

📊 **Stats:**  
- 6K+ Users  
- 4K+ Movies  
- 1M+ Ratings  

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites
- Python 3.8 or higher  
- Libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`

### 🚀 Steps
```bash
# Clone the repository
git clone https://github.com/<your-username>/Advanced-Movie-Recommendation-System.git

# Move into project directory
cd Advanced-Movie-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
