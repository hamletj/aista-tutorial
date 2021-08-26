#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:20:33 2019

@author: ji.l
"""

import pickle

class Model(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = pickle.load(open('clf.pickle', 'rb'))
        self.vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        pred_proba = y_proba[:, 1]
        return round(pred_proba[0], 3)

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        if y_pred == 1:
            pred_text = 'Negative'
        elif y_pred == 5:
            pred_text = 'Positive'
        else:
            pred_text = 'Neutral'
        return pred_text

    def main(self, X):
        """Call vectorizer, predict, and predict_proba
           X - "the food is great"
        """
        vectorized = self.vectorizer_transform([X])
        prediction = self.predict(vectorized)
        confidence = self.predict_proba(vectorized)
        output_json = {"prediction": prediction, "confidence": str(confidence)}
        return output_json
