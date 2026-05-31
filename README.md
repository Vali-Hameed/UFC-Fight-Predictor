![UFC Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png)

# 🥊 UFC Fight Predictor
A machine learning project that predicts the winner of UFC fights using a **Logistic Regression model**. The model is served via a **RESTful API** built with **FastAPI**, containerized with **Docker**, and deployed on **AWS ECS** for scalable performance.

## 📑 Table of Contents
- [✨ Features](#-features)
- [🚀 Getting Started](#-getting-started)
- [☁️ Deployment](#️-deployment)
- [📊 The Data](#-the-data)
- [🤖 The Model](#-the-model)
- [📈 Model Performance](#-model-performance)
- [🤝 Contributing](#-contributing)

## ✨ Features
* **RESTful API:** Exposes the prediction model through an API built with FastAPI.
* **Fight Winner Prediction:** Predicts the winner of a UFC match (Red or Blue corner).
* **Automated Data Updates:** Includes a scraper (`update_dataset.py`) that pulls the latest fight results from [ufcstats.com](http://ufcstats.com) and appends them directly to the training dataset.
* **Data-Driven:** Uses a comprehensive dataset of past UFC fights (`ufc-master.csv`) with 7,000+ fight records.
* **Hyperparameter Tuned:** Model parameters optimised via GridSearchCV for best accuracy.
* **Model Evaluation:** Includes model accuracy, a classification report, and a confusion matrix to evaluate performance.
* **Containerized Deployment:** Packaged as a Docker container for easy portability and deployment.
* **Cloud Deployment:** Successfully deployed and running on AWS ECS.

## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You'll need to have Python 3, pip and Docker installed.

### Installation
Clone the repository to your local machine:
```
git clone https://github.com/Vali-Hameed/UFC-Fight-Predictor.git
```
Navigate into the project directory and set up a virtual environment:
```
cd UFC-Fight-Predictor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```
Install Playwright (needed for the dataset updater):
```
playwright install chromium
```

### Training the Model
Before running the API, you need to train the model and generate the `.pkl` files:
```
cd app
python model.py
```

### Updating the Dataset
To pull in the latest UFC fight results and append them to `ufc-master.csv`:
```
cd app
python update_dataset.py
```
This script will:
1. Read the latest date in `ufc-master.csv`
2. Scrape all completed UFC events after that date from [ufcstats.com](http://ufcstats.com)
3. For each fight, scrape both fighters' career stats (height, reach, wins, losses, streaks, etc.)
4. Calculate all derived features (e.g. `HeightDif`, `WinStreakDif`, `TotalTitleBoutDif`)
5. Append the new rows to the CSV

After updating the dataset, re-run `python model.py` to retrain with the new data.

### Usage
#### Running the API Server
To serve the model via the FastAPI application, run the following command from the root directory:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
You can then access the interactive API documentation at http://localhost:8000/docs.

## ☁️ Deployment

### Docker Container
To build the Docker image for the application:
```
docker build -t ufc-fight-predictor .
```
To run the Docker container locally and expose the API:
```
docker run -p 8000:8000 ufc-fight-predictor
```
To push the image to Docker Hub (replace your-dockerhub-username with your actual username):
```
docker tag ufc-fight-predictor your-dockerhub-username/ufc-fight-predictor
docker push your-dockerhub-username/ufc-fight-predictor
```
### AWS ECS Deployment
This project has been successfully deployed as a Docker container on AWS Elastic Container Service (ECS) using an image from Docker Hub. Here is a high-level overview of the process:

1. **Push Image to Docker Hub:** The Docker image was first built and then pushed to a public or private repository on Docker Hub.

2. **Set up an ECS Cluster:** An ECS cluster was created in the AWS Management Console to manage and run the containerized application.

3. **Create a Task Definition:** A task definition was configured for the application. This specifies the Docker image, **configures port mappings to expose the API**, and sets required resources.

4. **Run the Task:** The task was then run on the ECS cluster. which pulls the image and runs it as a container, making the API accessible.
This deployment method leverages AWS-managed infrastructure, making it a scalable and robust way to run containerized applications
## 📊 The Data
The model is trained on the `ufc-master.csv` dataset, which contains detailed statistics for each fighter in every match, including:

* **Physical Attributes:** Age, height, reach, and weight.

* **Performance Metrics:** Win streaks, losses, knockouts, submissions, and average significant strikes landed.

* **Betting Odds:** Odds for both the Red and Blue corner fighters.

The dataset is kept up-to-date using the `update_dataset.py` scraper, which pulls the latest results from [ufcstats.com](http://ufcstats.com).

## 🤖 The Model
This project uses a Logistic Regression model from the scikit-learn library to classify fight outcomes.

* **Data Preprocessing:** The model handles categorical features with one-hot encoding and scales numerical features using StandardScaler to ensure they are appropriately weighted.

* **Hyperparameter Tuning:** GridSearchCV was used to find the optimal parameters. The best configuration uses **Elastic Net regularisation** (`C=0.1`, `l1_ratio=0.5`) with the SAGA solver.

* **Training:** The dataset is split into training (50%) and testing (50%) sets to train and then evaluate the model on unseen data.

* **Evaluation:** The model's performance is measured using its accuracy score, a detailed classification report, and a confusion matrix.

### Current Results (Tuned Model — 50/50 split)
```
--- Model Evaluation ---
Model Accuracy: 0.6966

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.62      0.48      0.54      1343
    Red Wins       0.73      0.83      0.77      2292

    accuracy                           0.70      3635
   macro avg       0.67      0.65      0.66      3635
weighted avg       0.69      0.70      0.69      3635


Confusion Matrix:
[[ 640  703]
 [ 400 1892]]

Confusion Matrix Interpretation:
Correctly predicted 'Blue Wins': 640
Incorrectly predicted 'Red Wins' (False Positive): 703
Incorrectly predicted 'Blue Wins' (False Negative): 400
Correctly predicted 'Red Wins': 1892
```

### Previous Results (Before Tuning)
<details>
<summary>Click to expand previous results</summary>

#### Result with 50/50 train/test split
```
Model Accuracy: 0.6590

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.60      0.51      0.55      1340
    Red Wins       0.69      0.76      0.72      1924

    accuracy                           0.66      3264
```

#### Result with 80/20 train/test split
```
Model Accuracy: 0.6478

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.57      0.51      0.54       527
    Red Wins       0.69      0.74      0.71       779

    accuracy                           0.65      1306
```

#### Result with 40/60 train/test split
```
Model Accuracy: 0.6602

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.60      0.52      0.55      1602
    Red Wins       0.69      0.76      0.73      2315

    accuracy                           0.66      3917
```
</details>

## 📈 Model Performance

The model was evaluated against the real-world results of **UFC 321**. It correctly predicted 4 out of the 4 fights it had sufficient data for, achieving **100% accuracy** on this event (excluding the no contest of Tom Aspinall vs Ciryl Gane).

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Tom Aspinall vs. Ciryl Gane          | Tom Aspinall     | **No Contest (Draw)** | Tom will beat him next time   |
| Mackenzie Dern vs. Virna Jandiroba    | Mackenzie Dern   | **Mackenzie Dern** | ✅ Correct   |
| Umar Nurmagomedov vs. Mario Bautista      | Umar Nurmagomedov   | **Umar Nurmagomedov** | ✅ Correct   |
| Jailton Almeida vs. Alexander Volkov           | Alexander Volkov      | **Alexander Volkov** | ✅ Correct |
| Azamat Murzakanov vs. Aleksandar Rakić | Azamat Murzakanov | **Azamat Murzakanov**| ✅ Correct |

> **Note**: Previously, fighters missing from the CSV could not be evaluated. By running the new automated dataset updater before generating predictions, the model now has access to the most recent fighter records, significantly reducing missing data issues for active veterans!

### Previous Events
<details>
<summary>Click to expand older event predictions</summary>

The model was evaluated against the real-world results of **Noche UFC: Lopes vs. Silva**. It correctly predicted 3 out of the 4 fights it had sufficient data for, achieving **75% accuracy** on this event.

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Alexander Hernandez vs. Diego Ferreira | Alexander Hernandez   | **Alexander Hernandez** | ✅ Correct   |
| Kelvin Gastelum vs. Dustin Stoltzfus   | Kelvin Gastelum       | **Kelvin Gastelum** | ✅ Correct   |
| Rafa Garcia vs. Jared Gordon         | Rafa Garcia           | **Rafa Garcia** | ✅ Correct   |
| Diego Lopes vs. Jean Silva           | Jean Silva            | **Diego Lopes** | ❌ Incorrect |

> **Note**: Historically, the model could not predict some fights from recent cards because the static `ufc-master.csv` dataset was out of date. With the introduction of the automated scraper (`update_dataset.py`), this issue is largely resolved! As long as the scraper is run, the model has up-to-date fight data. (However, true debut fighters like Santiago Luna or David Martinez still will not have past UFC data to predict on).

The model was evaluated against the real-world results of **UFC 320**. It correctly predicted 2 out of the 4 fights it had sufficient data for, achieving **50% accuracy** on this event.

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Magomed Ankalaev vs. Alex Pereira       | Alex Pereira     | **Alex Pereira** | ✅ Correct   |
| Merab Dvalishvili vs. Cory Sandhagen    | Cory Sandhagen   | **Merab Dvalishvili** | ❌ Incorrect   |
| Jiří Procházka vs. Khalil Rountree      | Jiří Procházka   | **Jiří Procházka** | ✅ Correct   |
| Josh Emmett vs. Youssef Zalal           | Josh Emmett      | **Youssef Zalal** | ❌ Incorrect |

</details>
## 🤝 Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1.  Fork the Project

2.  Create your Feature Branch (git checkout -b feature/AmazingFeature)

3.  Commit your Changes (git commit -m 'Add some AmazingFeature')

4.  Push to the Branch (git push origin feature/AmazingFeature)

5.  Open a Pull Request

## 📜 License
This project is distributed under the MIT License. See the LICENSE file for more information.
