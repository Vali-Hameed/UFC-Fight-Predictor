![UFC Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png)

# 🥊 UFC Fight Predictor
A machine learning project that predicts the winner of UFC fights using a **Logistic Regression model**.The model is served via a **RESTful API** built with **FastAPI**, containerized with **Docker**, and deployed on **AWS ECS** for scalable performance.

## ✨ Features
* **RESTful API:** Exposes the prediction model through an API built with FastAPI.
* **Fight Winner Prediction:** Predicts the winner of a UFC match (Red or Blue corner).

* **Data-Driven:** Uses a comprehensive dataset of past UFC fights (ufc-master.csv).

* **Model Evaluation:** Includes model accuracy, a classification report, and a confusion matrix to evaluate performance.

* **Feature Importance:** Identifies key features that are most predictive of a fight's outcome.

* **Containerized Deployment:** Packaged as a Docker container for easy portability and deployment.

* **Cloud Deployment:** Successfully deployed and running on AWS ECS.

## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You'll need to have Python 3, pip and Docker installed. You can install the necessary Python libraries using the following command:
```
pip install -r requirements.txt
```

### Installation
Clone the repository to your local machine:
```
git clone [https://github.com/vali-hameed/ufc-fight-predictor.git](https://github.com/vali-hameed/ufc-fight-predictor.git)
```
Navigate into the project directory:
```
cd ufc-fight-predictor
```
### Usage
#### Running the API Server
To serve the model via the FastAPI application, run the following command from the root directory:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
You can then access the interactive API documentation at http://localhost:8000/docs.
To run the predictor and see the model's accuracy, execute the ufc_predictor.py script:
```
python app/ufc_predictor.py
```
To predict a different fight, change the names in the following line inside the app/ufc_predictor.py script:
```
predict_hypothetical_fight('Tom Aspinall', 'Jon Jones', model_pipeline, df, numerical_features + categorical_features)
```
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
## Deployment on AWS ECS
This project has been successfully deployed as a Docker container on AWS Elastic Container Service (ECS) using an image from Docker Hub. Here is a high-level overview of the process:

1. **Push Image to Docker Hub:** The Docker image was first built and then pushed to a public or private repository on Docker Hub.

2. **Set up an ECS Cluster:** An ECS cluster was created in the AWS Management Console to manage and run the containerized application.

3. **Create a Task Definition:** A task definition was configured for the application. This specifies the Docker image, **configures port mappings to expose the API**, and sets required resources.

4. **Run the Task:** The task was then run on the ECS cluster. which pulls the image and runs it as a container, making the API accessible.
This deployment method leverages AWS-managed infrastructure, making it a scalable and robust way to run containerized applications
## 📊 The Data
The model is trained on the ufc-master.csv dataset, which contains detailed statistics for each fighter in every match, including:

* **Physical Attributes:** Age, height, reach, and weight.

* **Performance Metrics:** Win streaks, losses, knockouts, submissions, and average significant strikes landed.

* **Betting Odds:** Odds for both the Red and Blue corner fighters.

## 🤖 The Model
This project uses a Logistic Regression model from the scikit-learn library to classify fight outcomes.

* **Data Preprocessing:** The model handles categorical features with one-hot encoding and scales numerical features using StandardScaler to ensure they are appropriately weighted.

* **Training:** The dataset is split into training (40%) and testing (60%) sets to train and then evaluate the model on unseen data.

* **Evaluation:** The model's performance is measured using its accuracy score, a detailed classification report, and a confusion matrix.
### Result with 50/50 train/test split
```
--- Model Evaluation ---
Model Accuracy: 0.6590

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.60      0.51      0.55      1340
    Red Wins       0.69      0.76      0.72      1924

    accuracy                           0.66      3264
   macro avg       0.65      0.64      0.64      3264
weighted avg       0.65      0.66      0.65      3264


Confusion Matrix:
[[ 687  653]
 [ 460 1464]]

Confusion Matrix Interpretation:
Correctly predicted 'Blue Wins': 687
Incorrectly predicted 'Red Wins' (False Positive): 653
Incorrectly predicted 'Blue Wins' (False Negative): 460
Correctly predicted 'Red Wins': 1464

--- Predicting: Tom Aspinall (Red) vs. Jon Jones (Blue) ---
Prediction Probabilities:
  - Jon Jones (Blue) wins: 29.04%
  - Tom Aspinall (Red) wins: 70.96%

Predicted Winner: Tom Aspinall
```
### Result with 80/20 train/test split
```
--- Model Evaluation ---
Model Accuracy: 0.6478

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.57      0.51      0.54       527
    Red Wins       0.69      0.74      0.71       779

    accuracy                           0.65      1306
   macro avg       0.63      0.63      0.63      1306
weighted avg       0.64      0.65      0.64      1306


Confusion Matrix:
[[271 256]
 [204 575]]

Confusion Matrix Interpretation:
Correctly predicted 'Blue Wins': 271
Incorrectly predicted 'Red Wins' (False Positive): 256
Incorrectly predicted 'Blue Wins' (False Negative): 204
Correctly predicted 'Red Wins': 575

--- Predicting: Tom Aspinall (Red) vs. Jon Jones (Blue) ---
Prediction Probabilities:
  - Jon Jones (Blue) wins: 24.28%
  - Tom Aspinall (Red) wins: 75.72%

Predicted Winner: Tom Aspinall
```
### result with 40/60 train/test split
```
--- Model Evaluation ---
Model Accuracy: 0.6602

Classification Report:
              precision    recall  f1-score   support

   Blue Wins       0.60      0.52      0.55      1602
    Red Wins       0.69      0.76      0.73      2315

    accuracy                           0.66      3917
   macro avg       0.65      0.64      0.64      3917
weighted avg       0.65      0.66      0.66      3917


Confusion Matrix:
[[ 828  774]
 [ 557 1758]]

Confusion Matrix Interpretation:
Correctly predicted 'Blue Wins': 828
Incorrectly predicted 'Red Wins' (False Positive): 774
Incorrectly predicted 'Blue Wins' (False Negative): 557
Correctly predicted 'Red Wins': 1758
```
## Model Performance

The model was evaluated against the real-world results of **Noche UFC: Lopes vs. Silva**. It correctly predicted 3 out of the 4 fights it had sufficient data for, achieving **75% accuracy** on this event.

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Alexander Hernandez vs. Diego Ferreira | Alexander Hernandez   | **Alexander Hernandez** | ✅ Correct   |
| Kelvin Gastelum vs. Dustin Stoltzfus   | Kelvin Gastelum       | **Kelvin Gastelum** | ✅ Correct   |
| Rafa Garcia vs. Jared Gordon         | Rafa Garcia           | **Rafa Garcia** | ✅ Correct   |
| Diego Lopes vs. Jean Silva           | Jean Silva            | **Diego Lopes** | ❌ Incorrect |

> **Note**: The model could not predict some fights from the card because the fighters were not present in the `ufc-master.csv` dataset (Santiago Luna, Jean Silva, Lee Quang and David Martinez made their debut fight so no past ufc data).

The model was evaluated against the real-world results of **UFC 320**. It correctly predicted 2 out of the 4 fights it had sufficient data for, achieving **50% accuracy** on this event.

| Matchup                              | Predicted Winner      | Actual Winner         | Result      |
| ------------------------------------ | --------------------- | --------------------- | ----------- |
| Magomed Ankalaev vs. Alex Pereira       | Alex Pereira     | **Alex Pereira** | ✅ Correct   |
| Merab Dvalishvili vs. Cory Sandhagen    | Cory Sandhagen   | **Merab Dvalishvili** | ❌ Incorrect   |
| Jiří Procházka vs. Khalil Rountree      | Jiří Procházka   | **Jiří Procházka** | ✅ Correct   |
| Josh Emmett vs. Youssef Zalal           | Josh Emmett      | **Youssef Zalal** | ❌ Incorrect |

> **Note**: The model could not predict some fights from the card because the fighters were not present in the `ufc-master.csv` dataset.
## 🤝 Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1.  Fork the Project

2.  Create your Feature Branch (git checkout -b feature/AmazingFeature)

3.  Commit your Changes (git commit -m 'Add some AmazingFeature')

4.  Push to the Branch (git push origin feature/AmazingFeature)

5.  Open a Pull Request

## 📜 License
This project is distributed under the MIT License. See the LICENSE file for more information.
