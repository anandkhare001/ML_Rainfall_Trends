import unittest
import os
import pickle
import pytest
from prophet import Prophet

class TestClass(unittest.TestCase):
    
    def test_rainfall_trends(self):
        # Load Model
        model_path = 'model.pkl'
        model = pickle.load(open(model_path, 'rb'))

        # create a future dataframe for the next 20 years
        future = model.make_future_dataframe(periods=20, freq='YE')        
        forecast = model.predict(future)

        # Do assertions
        self.assertIsInstance(model, Prophet)
        self.assertEqual(future.ds[0], forecast.ds[0])
        self.assertEqual(future.ds[0], forecast.ds[0])
        self.assertEqual(forecast.trend[0], 1040.6390576)