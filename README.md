# SuperPlay: AI-Powered Spotify Playlist Curator

**SuperPlay** is an AI-based project that curates Spotify playlists by analyzing audio features, mood prediction, and lyrical sentiment to provide highly personalized music recommendations. This project uses advanced machine learning models and provides an API to make predictions accessible.

## Project Structure

The project directory contains the following files:

1. **`collect.py`**: Script for collecting and preprocessing data from Spotify and Genius APIs.
2. **`datacleaning.py`**: Script for cleaning the dataset to prepare it for machine learning.
3. **`ml2.py`**: Script for training an XGBoost model, evaluating its performance, and creating a Flask API for predictions.
4. **`superplay.py`**: Main script for integrating different parts of the project.
5. **`tracks_dataset.csv`**: Raw dataset containing track features collected from Spotify.
6. **`tracks_dataset_cleaned.csv`**: Cleaned version of the dataset after preprocessing.
7. **`music_mode_classifier.pkl`**: Saved model file after training.
8. **`Dockerfile`**: Docker configuration for deploying the project.
9. **`correlation_heatmap.png`**: Visual representation of the correlation between different audio features.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker (optional for containerization)
- Spotify and Genius API keys

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/superplay.git
   cd superplay
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

### Running the Project

1. **Data Collection**:
   - Use `collect.py` to gather data from Spotify and Genius APIs.

2. **Data Cleaning**:
   - Run `datacleaning.py` to clean and prepare the dataset for machine learning.

3. **Model Training**:
   - Train the machine learning model using `ml2.py`. This script includes hyperparameter tuning and model evaluation.

4. **API Deployment**:
   - Run the Flask API locally:
     ```sh
     python ml2.py
     ```
   - Alternatively, use Docker to deploy the API:
     ```sh
     docker build -t superplay-api .
     docker run -p 5000:5000 superplay-api
     ```

5. **Testing the API**:
   - Use Postman or cURL to test the API endpoint `/predict`:
     ```sh
     curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '[{"danceability": 0.8, "energy": 0.7, "key": 5, "loudness": -5.4, "speechiness": 0.06, "acousticness": 0.1, "instrumentalness": 0.0, "liveness": 0.12, "valence": 0.9, "tempo": 120.0, "duration_ms": 210000, "time_signature": 4}]'
     ```

## Features

- **Audio Analysis**: Extracts features like danceability, energy, acousticness, etc.
- **Mood Prediction**: Uses transformers to predict the mood of tracks based on lyrics.
- **Playlist Curation**: Uses XGBoost classifier to create personalized playlists.
- **API Access**: Flask API to get predictions for given audio features.

## Docker Deployment

To deploy using Docker:

1. Build the Docker image:
   ```sh
   docker build -t superplay-api .
   ```

2. Run the Docker container:
   ```sh
   docker run -p 5000:5000 superplay-api
   ```

## Project Files

- **`collect.py`**: Collects data from Spotify and Genius.
- **`datacleaning.py`**: Prepares the dataset.
- **`ml2.py`**: Trains the model and serves the API.
- **`Dockerfile`**: Builds the project for Docker deployment.
- **`tracks_dataset.csv`**: Raw dataset.
- **`tracks_dataset_cleaned.csv`**: Cleaned dataset.
- **`music_mode_classifier.pkl`**: Trained model.
- **`correlation_heatmap.png`**: Correlation heatmap for feature analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Spotify** and **Genius** for providing API access to gather music data.
- **XGBoost** and **Scikit-Learn** for machine learning tools.
- **Flask** for API deployment.
- **Docker** for containerization.

## Contributing

Feel free to open issues and pull requests to contribute to this project.

## Contact

For any inquiries, please contact [your-email@example.com].

