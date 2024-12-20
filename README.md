# User Recommendation System

## Overview

This User Recommendation System is designed to analyze user data, calculate similarity scores between users based on selected attributes, and store the results in a MongoDB database. It leverages Natural Language Processing (NLP) techniques to compute similarity scores, allowing for effective recommendations based on user characteristics.

## Features

- Fetches user data from a MongoDB database.
- Calculates similarity scores using word embeddings.
- Performs sampling for efficient processing.
- Writes similarity scores and counts back to the MongoDB database.
- Configurable through environment variables for flexibility.

## Technologies Used

- Python
- MongoDB
- SpaCy (for NLP)
- Dask (for parallel computing)
- Logging module
- dotenv (for environment variable management)

## Installation

1. **Clone the repository:**

2. **Create a virtual environment (optional but recommended):**

    python -m venv env
    source env/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Inatall required packages**

    pip install -r requirements.txt

4. **Set up your MongoDB database**

To use this application, you need a MongoDB instance running. Follow these steps to set it up:

**Install MongoDB:**


**Start the MongoDB Server:**

   After installation, start the MongoDB server. Open a terminal (or command prompt) and run:


5. **Create a .envfile:**

## Usage 
 
To run the recommendation system, execute the following command:

    python main.py

## Logging
The application uses Python's built-in logging module to log the progress and any errors encountered during execution. Logs will be printed to the console, making it easier to monitor the system's operation.

## Error Handling

The application includes basic error handling to manage exceptions that may arise during data fetching, processing, or writing to MongoDB. Errors will be logged appropriately.

## Contribution

Feel free to contribute to the project by submitting issues or pull requests. Your feedback and enhancements are welcome!

