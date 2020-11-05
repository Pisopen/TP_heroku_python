import pandas as pd
import uvicorn
import sklearn
import joblib
from fastapi import FastAPI

app= FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
async def root():
	return {"message" : "Hello World"}


# Using Get
@app.get('/predict/{tweet}')
async def predict(tweet:
	
    tweetX = pd.DataFrame({'tweet':[tweet]})

    Y_pred = model.predict_proba(tweetX)[0]
    
    if Y_pred[0] > Y_pred[1] and Y_pred[0]> Y_pred[2]:
        return {"tweet_type":'hate_speech',"prediction":Y_pred[0]}
    elif Y_pred[1] > Y_pred[0] and Y_pred[1]> Y_pred[2] :
        return {"tweet_type":'offensive_language',"prediction":Y_pred[1]}
    else :
        return {"tweet_type":'either',"prediction":Y_pred[2]}
        
if __name__ == '__main__' :
    uvicorn.run(app,host="0.0.0.0",port="8000")