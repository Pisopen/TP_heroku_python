import pandas as pd
import uvicorn
import sklearn
import joblib
from fastapi import FastAPI

tweet = st.text_input("tweet:","")
model = joblib.load("model.pkl")

if tweet:
    Y_pred = model.predict_proba(tweet)[0]
    
    if Y_pred[0] > Y_pred[1] and Y_pred[0]> Y_pred[2]:
        st.text("tweet_type":"hate_speech")
    elif Y_pred[1] > Y_pred[0] and Y_pred[1]> Y_pred[2] :
        st.text("tweet_type":"offensive_language")
    else :
        return {"tweet_type":"either"}
    
else:
    st.text("Tweet necessary") 