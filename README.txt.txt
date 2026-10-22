Project Title: Machine Learning-Based Scandinavian Hotel Price Prediction and Development of Smart Web Application

Student: Çağdaş Can Azim
Student Number: 230404012

This project uses a hotel dataset collected from the Booking application.
The original scraped dataset contained 13,958 hotel records and 40 columns.
After data cleaning, 10,214 usable hotel records were used for model training and evaluation.

The project includes hotel price category classification and numerical hotel price prediction.
The final web application was developed using Streamlit.

To run the Streamlit application:

1. Open terminal or command prompt in the project folder.
2. Install required packages:
   pip install -r requirements.txt

3. Run the application:
   python -m streamlit run app.py

If there is a scikit-learn version problem, install the same scikit-learn version used during model training.

Main files:
- app.py: Streamlit web application
- saved_model/: trained model and preprocessing files
- figures/: project figures
- tables/: project tables
- MLProject report PDF and DOCX files