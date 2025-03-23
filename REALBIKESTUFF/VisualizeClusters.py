import os
import json
import folium
import random
from shapely.geometry import MultiPoint
from folium.plugins import FeatureGroupSubGroup

def get_random_color():
    """Generate a random HEX color."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def main():
    # Define file paths (adjust paths as needed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coords_file = os.path.join(base_dir, "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
    clusters_file = os.path.join(base_dir, "Region_Creation_Parsing", "station_clusters.json")
    
    # Load station coordinates and clusters.
    with open(coords_file, "r") as f:
        station_coords = json.load(f)
    
    with open(clusters_file, "r") as f:
        clusters_data = json.load(f)
    clusters = clusters_data.get("clusters", [])

    # Calculate center of the map based on all station coordinates.
    all_lat = [data["latitude"] for data in station_coords.values()]
    all_lon = [data["longitude"] for data in station_coords.values()]
    center_lat = sum(all_lat) / len(all_lat)
    center_lon = sum(all_lon) / len(all_lon)
    
    # Initialize Folium map centered on the computed location.
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Create feature groups for clusters and their respective polygons.
    base_group = folium.FeatureGroup(name='Base Map').add_to(m)
    cluster_groups = {}
    polygon_groups = {}
    cluster_colors = {}
    
    for idx in range(len(clusters)):
        color = get_random_color()
        cluster_colors[idx] = color
        cluster_groups[idx] = FeatureGroupSubGroup(base_group, f"Cluster {idx} Points")
        polygon_groups[idx] = FeatureGroupSubGroup(base_group, f"Cluster {idx} Zone")
        cluster_groups[idx].add_to(m)
        polygon_groups[idx].add_to(m)

    # For each cluster, compute and draw the convex hull to visualize the grouping.
    for idx, cluster in enumerate(clusters):
        points = []
        for station in cluster:
            data = station_coords.get(station)
            if data is None:
                print(f"Warning: No coordinate found for station '{station}'")
                continue
            points.append((data["longitude"], data["latitude"]))
        
        if len(points) >= 3:
            hull = MultiPoint(points).convex_hull
            hull_coords = list(hull.exterior.coords)
            hull_latlon = [[lat, lon] for lon, lat in hull_coords]
            folium.Polygon(
                locations=hull_latlon,
                color=cluster_colors[idx],
                fill=True,
                fill_opacity=0.2,
                popup=f"Cluster {idx} Boundary",
            ).add_to(polygon_groups[idx])
    
    # Add individual station markers.
    for idx, cluster in enumerate(clusters):
        for station in cluster:
            data = station_coords.get(station)
            if data is None:
                continue
            folium.CircleMarker(
                location=[data["latitude"], data["longitude"]],
                radius=5,
                color=cluster_colors[idx],
                fill=True,
                fill_color=cluster_colors[idx],
                fill_opacity=0.8,
                popup=f"{station} (Cluster {idx})"
            ).add_to(cluster_groups[idx])
    
    # Add custom JavaScript for the legend toggle
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: auto;
                 background-color: white; z-index:9999; padding: 10px; border-radius: 5px;
                 box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-size:14px;">
        <b>Cluster Legend</b><br>
        <script>
        function toggleLayer(layerName, isPolygon) {
            var layers = window.layers;
            if (layers) {
                var layer = layers[layerName];
                if (layer) {
                    if (isPolygon) {
                        layer.eachLayer(function (l) { l.setStyle({ fillOpacity: l.options.fillOpacity === 0 ? 0.2 : 0 }); });
                    } else {
                        if (map.hasLayer(layer)) {
                            map.removeLayer(layer);
                        } else {
                            map.addLayer(layer);
                        }
                    }
                }
            }
        }
        </script>
    """
    for idx in range(len(clusters)):
        legend_html += f'<input type="checkbox" onclick="toggleLayer(\'Cluster {idx} Zone\', true)" checked> Zone {idx}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control to toggle cluster visibility
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save the resulting map to an HTML file.
    output_file = os.path.join(base_dir, "station_clusters_detailed_map.html")
    m.save(output_file)
    print(f"Map has been saved to {output_file}")

if __name__ == "__main__":
    main()
