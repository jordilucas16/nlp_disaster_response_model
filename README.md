# Disaster Response Pipeline Project

### Motivation:
The purpose of the project is to build a Natural Language Processing (NLP) tool that categorize messages from Twitter, for instance, building a model for an API that classifies disaster messages.
Using this application, an emergency worker can select the necessary resorces in function the disaster category type to help or save people.

### Screenshots:

![alt text](https://github.com/jordilucas16/disaster_response/blob/master/screenshot_1.PNG)

![alt text](https://github.com/jordilucas16/disaster_response/blob/master/screenshot_2.PNG)

### Dependencies:
* Python 3.7
* Machine Learning libraries
    * NumPy
    * Pandas
    * SciKit-Learn
    * SciPy
* Natural Language Processing (NLP) Libraries
    * NLTK 
* SQLlite DB Libraries
    * SQLAlchemy
* Python Web App ans Data visualization
    * Flask
    * Plotty
    
### Install
You can get the code by clone this repository:

`git clone https://github.com/jordilucas16/disaster_response`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_ETL.db'
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        'python models/train_classifier.py data/Disaster_ETL.db models/model.pkl'

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:5000/ 
