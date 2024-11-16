import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import folium
from geopy.geocoders import Nominatim
import requests
from streamlit_folium import st_folium

# Streamlit Setup
st.title("Bangalore City Traffic Prediction")

# File uploader for dataset input
uploaded_file = st.file_uploader("Upload your CSV traffic dataset", type="csv")

if uploaded_file is not None:
    # Load dataset
    traffic_data = pd.read_csv(uploaded_file)
    st.write("Data preview:", traffic_data.head())

    # Area Name Dropdown Filter
    area_names = traffic_data['Area Name'].unique()
    selected_area = st.selectbox("Select Area Name", options=area_names)

    # Filter data based on selected Area Name
    filtered_data = traffic_data[traffic_data['Area Name'] == selected_area]

    # Intersection Name Dropdown Filter
    intersection_names = filtered_data['Road/Intersection Name'].unique()
    selected_intersection = st.selectbox("Select Road/Intersection Name", options=intersection_names)

    # Display filtered data
    filtered_intersection_data = filtered_data[filtered_data['Road/Intersection Name'] == selected_intersection]
    st.write(f"Filtered Data for {selected_area} - {selected_intersection}:", filtered_intersection_data)

    # Encode categorical features
    categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Roadwork and Construction Activity']
    for column in categorical_columns:
        encoder = LabelEncoder()
        traffic_data[column] = encoder.fit_transform(traffic_data[column])

    # Prepare features and target
    X = traffic_data.drop(['Date', 'Traffic Volume'], axis=1)
    y = traffic_data['Traffic Volume']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # SVR model
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)

    #LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),  # timesteps=1, features=X_train.shape[1]
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

    # CNN model
    cnn_model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    cnn_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=32, validation_data=(X_test_cnn, y_test))
    y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

    # Historical Average (HA) model
    ha_model = y_train.mean()
    y_pred_ha = np.array([ha_model] * len(y_test))

    # Ensemble predictions
    y_pred_svr_ha = (y_pred_svr + y_pred_ha) / 2
    y_pred_svr_lstm = (y_pred_svr + y_pred_lstm) / 2
    y_pred_ha_lstm = (y_pred_ha + y_pred_lstm) / 2
    y_pred_svr_lstm_ha = (y_pred_svr + y_pred_lstm + y_pred_ha) / 3

    # Evaluate models and ensembles
    results = {
        "SVR + HA": {
            "MAE": mean_absolute_error(y_test, y_pred_svr_ha),
            "MSE": mean_squared_error(y_test, y_pred_svr_ha),
            "R²": r2_score(y_test, y_pred_svr_ha),
        },
        "SVR + LSTM": {
            "MAE": mean_absolute_error(y_test, y_pred_svr_lstm),
            "MSE": mean_squared_error(y_test, y_pred_svr_lstm),
            "R²": r2_score(y_test, y_pred_svr_lstm),
        },
        "HA + LSTM": {
            "MAE": mean_absolute_error(y_test, y_pred_ha_lstm),
            "MSE": mean_squared_error(y_test, y_pred_ha_lstm),
            "R²": r2_score(y_test, y_pred_ha_lstm),
        },
        "SVR + LSTM + HA": {
            "MAE": mean_absolute_error(y_test, y_pred_svr_lstm_ha),
            "MSE": mean_squared_error(y_test, y_pred_svr_lstm_ha),
            "R²": r2_score(y_test, y_pred_svr_lstm_ha),
        },
        "CNN": {
            "MAE": mean_absolute_error(y_test, y_pred_cnn),
            "MSE": mean_squared_error(y_test, y_pred_cnn),
            "R²": r2_score(y_test, y_pred_cnn),
        }
    }

    # Display results
    st.write("Model Performance Metrics:")
    for model_name, metrics in results.items():
        st.write(f"{model_name} - MAE: {metrics['MAE']}, MSE: {metrics['MSE']}, R²: {metrics['R²']}")

    # Dropdown for model selection
    prediction_options = {
        "SVR + HA": y_pred_svr_ha,
        "SVR + LSTM": y_pred_svr_lstm,
        "HA + LSTM": y_pred_ha_lstm,
        "SVR + LSTM + HA": y_pred_svr_lstm_ha,
        "CNN": y_pred_cnn,
    }
    selected_model = st.selectbox("Select a prediction model for map visualization", list(prediction_options.keys()))
    selected_prediction = prediction_options[selected_model][0]

    traffic_volume_threshold = 30000
    route_type = "alternative" if selected_prediction > traffic_volume_threshold else "normal"

    st.write(f"**Selected Model Prediction Value**: {selected_prediction}")
    st.write(f"**Recommended Route Type**: {route_type.capitalize()}")

    st.subheader("Traffic-based Route Visualization")

    def get_geolocations(areas):
        geolocator = Nominatim(user_agent="myGeopyApp")
        geolocations = {}
        for area in areas:
            location = geolocator.geocode(f"{area}, Bangalore")
            if location:
                geolocations[area] = (location.latitude, location.longitude)
            else:
                geolocations[area] = None  # Handle missing locations
        return geolocations

    # Fetch and cache geolocations
    geolocations = get_geolocations(area_names)

    # Dropdowns for start and end locations
    start_location = st.selectbox("Select Start Location (Area Name)", options=area_names, key="unique_start_area")
    end_location = st.selectbox("Select End Location (Area Name)", options=area_names, key="unique_end_area")

    # Retrieve coordinates from cached geolocations
    start_coords = geolocations.get(start_location)
    end_coords = geolocations.get(end_location)

    if start_coords and end_coords:
        # Create and display map
        map_display = folium.Map(
            location=[start_coords[0], start_coords[1]],
            zoom_start=13,
            zoom_control=True,       # Disable zoom buttons
            dragging=True,           # Disable map dragging
            scrollWheelZoom=False,    # Disable scroll wheel zoom
            doubleClickZoom=False,    # Disable double click zoom
            touchZoom=False           # Disable touch zoom
        )

        # Route type based on traffic prediction
        route_profile = "driving-traffic" if route_type == "alternative" else "driving"

        # Get route data from Mapbox
        url = (
            f"https://api.mapbox.com/directions/v5/mapbox/{route_profile}/"
            f"{start_coords[1]},{start_coords[0]};"
            f"{end_coords[1]},{end_coords[0]}?geometries=geojson&access_token="
            f"pk.eyJ1IjoidmlzaGFscmsiLCJhIjoiY20zM2I0d3UzMTljejJrcjMxbm5qY3loeiJ9.A9CAu8GyGGxkHkq0AYfDDQ"
        )
        response = requests.get(url)
        route_data = response.json()

        if 'routes' in route_data and route_data['routes']:
            route_coords = [
                [lat, lon] for lon, lat in route_data['routes'][0]['geometry']['coordinates']
            ]
            folium.PolyLine(
                route_coords,
                color="blue" if route_type == "normal" else "red",
                weight=5,
                opacity=0.8
            ).add_to(map_display)

            # Display the map using streamlit_folium
            st_folium(
                map_display,
                width=800,
                returned_objects=[]
            )
        else:
            st.error("No route data available.")
    else:
        st.error("Could not retrieve coordinates for selected locations.")
