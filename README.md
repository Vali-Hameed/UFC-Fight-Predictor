<div align="center">
  <h1>🥊 UFC Fight Predictor ML</h1>
  <p><strong>A Machine Learning Microservice for Predicting UFC Fight Outcomes.</strong></p>
  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
  ![AWS](https://img.shields.io/badge/AWS_ECS-232F3E?style=flat-square&logo=amazon-aws&logoColor=white)
</div>

<hr />

## 📖 Overview

> **Note**: This repository houses the standalone Machine Learning microservice. For the core full-stack web application, visit the [UFC-Fight-Predictor-Website](https://github.com/Vali-Hameed/UFC-Fight-Predictor-Website) repository.

A machine learning project that predicts the winner of UFC fights using a **Gradient Boosting Classifier** with **66.09%** true symmetric accuracy. The model is served via a **RESTful API** built with **FastAPI**, containerized with **Docker**, and currently deployed on **AWS ECS** for scalable performance.

---

## ✨ Key Features

- **RESTful API**: Exposes the prediction model through an API built with FastAPI.
- **Fight Winner Prediction**: Predicts the winner of a UFC match (Red or Blue corner).
- **Automated Data Updates**: Includes a scraper (`update_dataset.py`) that pulls the latest fight results from `ufcstats.com` and appends them directly to the training dataset.
- **Data-Driven**: Uses a comprehensive dataset of past UFC fights (`ufc-master.csv`) with 7,000+ fight records.
- **Hyperparameter Tuned**: Model parameters optimized for tree depth, learning rate, and estimators to handle heavy target skew and prevent overfitting.
- **Symmetrized Inference**: Ensures prediction impartiality by averaging the model's output across both corner assignments.

---

## 📂 Repository Structure

```text
UFC-Fight-Predictor/
├── app/                      # Main application logic
│   ├── main.py               # FastAPI entrypoint
│   ├── model.py              # Model training script
│   ├── ufc_predictor.py      # Core inference logic
│   ├── update_dataset.py     # Local manual dataset backfill script
│   ├── ufc-master.csv        # Historical UFC dataset
│   └── *.pkl                 # Pickled model files
├── Dockerfile                # Deployment configuration
├── docker-compose.yml        # Local docker composition
└── requirements.txt          # Python dependencies
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3
- pip
- Docker

### 2. Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/Vali-Hameed/UFC-Fight-Predictor.git
cd UFC-Fight-Predictor
```

Set up a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
playwright install chromium  # Needed for dataset updater
```

### 3. Training the Model
Before running the API, you need to train the model and generate the `.pkl` files:
```bash
cd app
python model.py
```

### 4. Updating the Dataset
> **Note:** While this repository contains a local `update_dataset.py` script for manual dataset backfills, the live production scraping is managed by a dedicated microservice. See the [UFC-Scraper](https://github.com/Vali-Hameed/UFC-Scraper) repository for details on the automated cron scraping architecture.

To pull in the latest UFC fight results locally and append them to `ufc-master.csv`:
```bash
cd app
python update_dataset.py
```
After updating the dataset, re-run `python model.py` to retrain with the new data.

### 5. Running the API Server
To serve the model via the FastAPI application, run the following command from the root directory:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
You can then access the interactive API documentation at `http://localhost:8000/docs`. 

The primary endpoint is `POST /predict`, which accepts `red_fighter_name` and `blue_fighter_name` as query parameters.

---

## ☁️ Deployment

### Docker Container
To build the Docker image for the application locally:
```bash
docker build -t ufc-fight-predictor .
```

To run the Docker container locally and expose the API:
```bash
docker run -p 8000:8000 ufc-fight-predictor
```

### AWS ECS Deployment
This project has been successfully deployed as a Docker container on **AWS Elastic Container Service (ECS)** using an image from Docker Hub. Here is a high-level overview of the process:

1. **Push Image to Docker Hub:** The Docker image was first built and then pushed to a public or private repository on Docker Hub.
2. **Set up an ECS Cluster:** An ECS cluster was created in the AWS Management Console to manage and run the containerized application.
3. **Create a Task Definition:** A task definition was configured for the application. This specifies the Docker image, **configures port mappings to expose the API**, and sets required resources.
4. **Run the Task:** The task was then run on the ECS cluster, which pulls the image and runs it as a container, making the API accessible.

*(Note: The codebase also contains `.pkl` files and `Dockerfile` configurations compatible with dynamic `$PORT` PaaS providers, but the active production environment is AWS ECS.)*

---

## 📊 The Data

The model is trained on the `ufc-master.csv` dataset, which contains detailed statistics for each fighter in every match, including:
- **Physical Attributes**: Age, height, reach, and weight.
- **Performance Metrics**: Win streaks, losses, knockouts, submissions, and average significant strikes landed.
- **Betting Odds**: Odds for both the Red and Blue corner fighters.

---

## 🤖 The Model

This project uses a Gradient Boosting Classifier from the scikit-learn library to classify fight outcomes, replacing an earlier Logistic Regression approach.
- **Data Preprocessing**: Computes per-corner raw statistics (striking rates, takedown defense, etc.) rather than simple differentials to capture non-linear relationships.
- **Hyperparameter Tuning**: Tuned to handle skewed real-world target data (`max_depth=2`, `n_estimators=300`, `learning_rate=0.05`).
- **Symmetrized Inference**: To solve the inherent red-corner bias, the API evaluates every fight twice (A vs B, and B vs A) and averages the probabilities.

### Current Results (Gradient Boosting — Symmetrized Evaluation)
```text
=== Model C: Per-corner features ===
  Raw accuracy:  0.7433
  Sym accuracy:  0.6609
  Sym Brier:     0.2203

============================================================
SUMMARY
============================================================
  Model                                         Raw      Sym    Brier
  Baseline (always Red)                      0.7900      N/A      N/A
  A: GBM, diffs, no augmentation             0.7014   0.5216   0.2556
  B: GBM, diffs, WITH augmentation           0.5381   0.5347   0.2510
  C: GBM, per-corner features                0.7433   0.6609   0.2203
```

---

## 📈 Model Performance

The model was recently evaluated against the real-world results of **UFC 321**, achieving **100% accuracy** on all determinable matchups.

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Tom Aspinall vs. Ciryl Gane          | Tom Aspinall          | **No Contest (Draw)** | N/A         |
| Mackenzie Dern vs. Virna Jandiroba   | Mackenzie Dern        | **Mackenzie Dern**    | ✅ Correct   |
| Umar Nurmagomedov vs. Mario Bautista | Umar Nurmagomedov     | **Umar Nurmagomedov** | ✅ Correct   |
| Jailton Almeida vs. Alexander Volkov | Alexander Volkov      | **Alexander Volkov**  | ✅ Correct   |
| Azamat Murzakanov vs. Aleksandar Rakić| Azamat Murzakanov    | **Azamat Murzakanov** | ✅ Correct   |

> **Note**: Missing fighters from the CSV could not previously be evaluated. By running the new automated dataset updater before generating predictions, the model now has access to the most recent fighter records, significantly reducing missing data issues!

---

## 🛠️ Tech Stack

### Backend & ML
- **Language**: Python 3
- **Framework**: FastAPI
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Server**: Uvicorn

### Infrastructure
- **Deployment**: AWS ECS
- **Containerization**: Docker

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <i>Developed by <a href="https://github.com/Vali-Hameed">Vali Hameed</a></i>
</div>
