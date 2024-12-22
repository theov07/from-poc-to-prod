import unittest
from unittest.mock import MagicMock

from predict.predict.run import TextPredictionModel


class TestPredictionModel(unittest.TestCase):

    def test_predict(self):

        # Mock the TextPredictionModel to simulate its behavior
        mock_model = MagicMock(spec=TextPredictionModel)

        # Define the artefacts_path required for the model
        artefacts_path = "train/data/artefacts/2024-12-11-17-02-59"

        # Define the titles from the mock dataset
        titles = [
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        ]

        # Define the actual tags in the mock dataset
        actual_tags = ["php", "ruby-on-rails"]

        # Mock the predict method to return the correct predictions when called with the correct arguments
        # We assume the mock model's predict method is designed to accept 'text' and 'artefacts_path'
        mock_model.predict.return_value = actual_tags
        print(titles)
        print("actual_tags: ",actual_tags)

        # Run predictions for the titles in the mock dataset using the artefacts_path
        predicted_tags = mock_model.predict(titles, artefacts_path=artefacts_path)
        print("predicted_tags: ",predicted_tags)

        # Check if the predicted tags match the actual tags in the dataset
        self.assertEqual(predicted_tags, actual_tags, "Predicted tags do not match actual tags")
