import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium

# Sample data for area names and intersections with latitude and longitude
area_intersections = {
    "Indiranagar": {
        "100 Feet Road": [12.9716, 77.6400],
        "CMH Road": [12.9720, 77.6408],
    },
    "Whitefield": {
        "Marathahalli Bridge": [12.9550, 77.7010],
        "ITPL Main Road": [12.9855, 77.7364],
    },
    "Koramangala": {
        "Sony World Junction": [12.9306, 77.6264],
        "Sarjapur Road": [12.9256, 77.6287],
    },
    "M.G. Road": {
        "Trinity Circle": [12.9743, 77.6208],
        "Anil Kumble Circle": [12.9753, 77.6033],
    },
    "Jayanagar": {
        "Jayanagar 4th Block": [12.9254, 77.5931],
        "South End Circle": [12.9345, 77.5881],
    },
    "Hebbal": {
        "Hebbal Flyover": [13.0358, 77.5970],
        "Ballari Road": [13.0446, 77.6061],
    },
    "Yeshwanthpur": {
        "Yeshwanthpur Circle": [13.0297, 77.5308],
        "Tumkur Road": [13.0280, 77.5282],
    },
    "Electronic City": {
        "Silk Board Junction": [12.9185, 77.6239],
        "Hosur Road": [12.8484, 77.6616],
    },
}

# Function to calculate the shortest distance based on road network
def calculate_shortest_road_path(start_coords, end_coords):
    G = ox.graph_from_place('Bangalore, India', network_type='drive')
    start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
    
    # Find the shortest path using the road network
    shortest_route = nx.shortest_path(G, start_node, end_node, weight='length')
    
    return G, shortest_route

# Streamlit app layout
st.title("Shortest Path Finder Based on Traffic Volume and Intersections")

# Dropdowns to select areas
start_area = st.selectbox("Select Starting Area", options=list(area_intersections.keys()))
start_intersection = st.selectbox("Select Starting Intersection", options=list(area_intersections[start_area].keys()))

end_area = st.selectbox("Select Destination Area", options=list(area_intersections.keys()))
end_intersection = st.selectbox("Select Destination Intersection", options=list(area_intersections[end_area].keys()))

if (start_area != end_area) or (start_intersection != end_intersection):
    # Get the coordinates of the selected intersections
    start_coords = area_intersections[start_area][start_intersection]
    end_coords = area_intersections[end_area][end_intersection]
    
    # Calculate the shortest path
    G, shortest_route = calculate_shortest_road_path(start_coords, end_coords)

    # Create the map centered at the starting area
    route_map = ox.plot_route_folium(G, shortest_route)
    
    # Add markers for start and end areas
    folium.Marker(location=start_coords, popup=f"{start_area} - {start_intersection} (Start)").add_to(route_map)
    folium.Marker(location=end_coords, popup=f"{end_area} - {end_intersection} (End)").add_to(route_map)

    # Display the map in Streamlit
    st_folium(route_map, width=700, height=500)
else:
    st.warning("Please select different intersections for the start and end points.")
