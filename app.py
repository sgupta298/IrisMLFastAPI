from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
from numpy.core.records import array
import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

modelfile = 'models/final_prediction.pickle'

model = p.load(open(modelfile, 'rb'))


@app.get('/')
async def main():
    return ('Predict Iris API')


@app.post('/api/')
async def makecalc(id:list):
	idarray=np.array(id).reshape(1,4)

	prediction = np.array2string(model.predict(idarray))
	
	return prediction

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')    