from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
from .ufc_predictor import predict_hypothetical_fight

app = FastAPI(title="UFC Fight Predictor API")
pickle_in = open('app/ufc_logistic_model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.get("/")
def read_root():
    return {"message": "Welcome to the UFC Fight Predictor API! Use the /predict endpoint to get fight predictions."}


@app.post("/predict")
def predict(red_fighter_name: str, blue_fighter_name: str):
    try:
        with open('app/ufc_other_artifacts.pkl', 'rb') as f:
            other_artifacts = pickle.load(f)
        dataframe = other_artifacts['data_for_lookups']
        feature_cols = other_artifacts['categorical_features'] + other_artifacts['numerical_features']
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Artifact file not found: {e.filename}. Please run the training script first.")


    prediction = predict_hypothetical_fight(red_fighter_name, blue_fighter_name, model, dataframe, feature_cols)
    if prediction is None:
        raise HTTPException(status_code=404, detail="One or both fighter names not found in the dataset.")
    
    return {
        "predicted_winner": prediction['predicted_winner'],
        "red_fighter_win_probability": prediction['red_win_prob'],
        "blue_fighter_win_probability": prediction['blue_win_prob'],
        "status": "success"
    }
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)




