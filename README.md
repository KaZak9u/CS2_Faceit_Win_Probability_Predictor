# CS2_Faceit_Win_Probability_Predictor
## Overview
CS2_Faceit_Win_Probability_Predictor is a project aimed at predicting the probability of winning a given CS2 (Counter-Strike 2) match on Faceit. The project involves gathering historical match data, analyzing it through various visualizations, and employing several classification algorithms to build a predictive model.

## Features
- **Data Collection**: Retrieves historical match data from Faceit.
- **Data Visualization**: Generates plots to illustrate the relationships between different data features.
- **Model Training and Evaluation**: Tests various classifiers including SVM, LightGBM, Random Forest, and XGBoost. It uses grid search to find the best parameters for each classifier.
- **Prediction Script**: The main.py script calculates the win probability for a given match using a pre-trained XGBoost model.
## Getting Started
### Prerequisites
- Python 3.12
- Required Python packages listed in requirements.txt

### Installation

1. Clone the repository:

`git clone https://github.com/yourusername/CS2_Faceit_Win_Probability_Predictor.git` \
`cd CS2_Faceit_Win_Probability_Predictor`

2. Install the necessary dependencies:

`pip install -r requirements.txt`

### Usage

1. Data Collection:

&emsp; - Run the data collection script to retrieve historical match data. This script should be implemented in a file named matches_data.py.

&emsp; - Example:
&emsp; `python matches_data.py`

2. Data Visualization:

&emsp; - Generate plots to understand the data. This functionality is implemented in a file named plots.py.

&emsp; - Example:
&emsp; `python data_visualization.py`

3. Model Training and Evaluation:

&emsp; - Train and evaluate different classifiers. This is handled in a file named model_training.py.

&emsp; - Example:
&emsp; `python model_training.py`

4. Prediction:

&emsp; - Use the main.py script to calculate the win probability for a given match using the saved XGBoost model. Just swap in fil "matchrooom" variable with link to your faceit matchroom.

&emsp; - Example:

&emsp; `python main.py` 

## File Structure
- matches_data.py: Script for collecting historical match data from Faceit.
- plots.py: Script for generating plots to visualize data relationships.
- model_training.py: Script for training and evaluating classifiers with grid search for parameter optimization.
- main.py: Script for predicting the win probability of a given match using a pre-trained XGBoost model.
- requirements.txt: List of required Python packages.
## Credits
Developed by [Kacper Zakrzewski]

Enjoy predicting your matches and improving your strategies with the CS2_Faceit_Win_Probability_Predictor!
