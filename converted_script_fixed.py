import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from scipy.interpolate import griddata, splprep, splev
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import distance, ConvexHull

# Check if mesh file exists before loading
mesh_path = "model.glb"  # Path to your custom 3D object
if not os.path.isfile(mesh_path):
    raise FileNotFoundError(f"Mesh file '{mesh_path}' not found. Please provide the correct path.")

# Load the custom 3D object
mesh = trimesh.load_mesh(mesh_path)

# Extract vertices, faces, and vertex colors from the mesh
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# Extract vertex colors if available
if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize colors to [0, 1]
else:
    colors = None

# Define cutting thresholds for x-axis (remove 50% of x-axis range)
x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
x_cut_threshold = x_min + 0.5 * (x_max - x_min)  # Cut the first 50% of the x-axis

# Define cutting thresholds for z-axis (remove 20% from each side of z-axis)
z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
z_lower_threshold = z_min + 0.2 * (z_max - z_min)  # Lower 20% cut
z_upper_threshold = z_max - 0.2 * (z_max - z_min)  # Upper 20% cut

# Identify vertices to keep (x > x_cut_threshold and z within central 60%)
keep_indices = np.where(
    (vertices[:, 0] > x_cut_threshold) &
    (vertices[:, 2] >= z_lower_threshold) &
    (vertices[:, 2] <= z_upper_threshold)
)[0]

# Create a mapping from old vertex indices to new ones
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

# Filter vertices
new_vertices = vertices[keep_indices]

# Filter faces to only include triangles with all vertices retained
new_faces = []
new_colors = []

for face in faces:
    if all(v_idx in index_map for v_idx in face):  # Keep only faces where all vertices are retained
        new_faces.append([index_map[v_idx] for v_idx in face])  # Map old indices to new ones
        if colors is not None:
            new_colors.append(colors[face])

new_faces = np.array(new_faces)
new_colors = np.array(new_colors) if colors is not None else None

# Scale down the object size (reduce size by scaling factor)
scaling_factor = 3.5
new_vertices *= scaling_factor

# Save the final result in a variable
preprocessed = {
    'vertices': new_vertices,
    'faces': new_faces,
    'colors': new_colors
}

# Prepare data for Plotly mesh visualization
x, y, z = new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2]

# Create the 3D plot
fig = go.Figure()

# Add the cut and adjusted portion of the 3D object (as a triangular mesh)
fig.add_trace(go.Mesh3d(
    x=x, y=y, z=z,
    i=new_faces[:, 0], j=new_faces[:, 1], k=new_faces[:, 2],  # Triangle vertex indices
    vertexcolor=new_colors.reshape(-1, 3) if colors is not None else None,  # Use vertex colors if available
    opacity=1.0  # Fully opaque object
))

# Expand the viewing area (adjust axes range)
padding = 1.0  # Add extra space around the object
fig.update_layout(
    title="Custom 3D Model with Bigger Plane and Smaller Object",
    scene=dict(
        xaxis=dict(range=[np.min(x) - padding, np.max(x) + padding], title='X'),
        yaxis=dict(range=[np.min(y) - padding, np.max(y) + padding], title='Y'),
        zaxis=dict(range=[np.min(z) - padding, np.max(z) + padding], title='Z'),
    )
)

# Display the interactive figure
fig.show()

print("The preprocessed model is saved in the variable 'preprocessed'.")


# Extract data from the preprocessed variable
vertices = preprocessed['vertices']

# Extract x, y, z coordinates
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z (keeping x as height)
grid_resolution = 100  # Number of steps in the grid
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata(
    points=(y, z),
    values=x,
    xi=(y_mesh, z_mesh),
    method='linear'
)

# Create a heatmap-like surface plot with x as height
fig = go.Figure()

fig.add_trace(go.Surface(
    z=x_grid,  # Use x as the height (topography)
    x=y_mesh,  # y-axis coordinates
    y=z_mesh,  # z-axis coordinates
    colorscale='Viridis',  # Heatmap color scale
    colorbar=dict(title="Height (X)")
))

# Adjust layout
fig.update_layout(
    title="Heatmap Surface Topography (Height on X-Axis)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"  # Equal scaling for all axes
    )
)

# Show the plot
fig.show()


# Generate contour lines in Y-direction
contour_levels = np.linspace(np.min(y), np.max(y), 100)  # Increased contour lines for better resolution
contour_points = {'x': [], 'y': [], 'z': [], 'level': []}  # Store contour level info

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 110
    contour_points['x'].extend(x_grid[mask])
    contour_points['y'].extend(y_mesh[mask])
    contour_points['z'].extend(z_mesh[mask])
    contour_points['level'].extend([level] * np.sum(mask))

contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T

# Detect local minima per contour level
local_minima = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    level_mask = contour_array[:, 3] == level
    level_points = contour_array[level_mask]

    if len(level_points) == 0:
        continue

    filtered_points = level_points.copy()
    if len(level_points) > 3:
        try:
            hull = ConvexHull(level_points[:, [1, 2]])
            filtered_points = np.array([level_points[i] for i in range(len(level_points)) if i not in hull.vertices])
        except:
            pass

    if len(filtered_points) > 0:
        filtered_points = filtered_points[np.argsort(filtered_points[:, 2])]

    minima_found = False
    for i, (xi, yi, zi, _) in enumerate(filtered_points):
        neighbors = filtered_points[
            (np.abs(filtered_points[:, 1] - yi) < (np.max(y) - np.min(y)) / 60) &
            (np.abs(filtered_points[:, 2] - zi) < (np.max(z) - np.min(z)) / 70)
        ]

        if np.all(xi <= neighbors[:, 0]):
            local_minima['x'].append(xi)
            local_minima['y'].append(yi)
            local_minima['z'].append(zi)
            minima_found = True

    if not minima_found and len(filtered_points) > 0:
        min_idx = np.argmin(filtered_points[:, 0])
        local_minima['x'].append(filtered_points[min_idx, 0])
        local_minima['y'].append(filtered_points[min_idx, 1])
        local_minima['z'].append(filtered_points[min_idx, 2])

# Plot local minima points
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=local_minima['y'],
    y=local_minima['z'],
    z=local_minima['x'],
    mode='markers',
    marker=dict(color='red', size=6, symbol='circle'),
    name="Local Minima",
    hoverinfo="text",
    text=[f"Min X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(local_minima['x'], local_minima['y'], local_minima['z'])]
))

fig.update_layout(
    title="Contour Lines with Highlighted Local Minima (Filtered & Aligned, 100 Lines)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

fig.show()


# Filter local minima points for Y range between -0.1 and 0.1
y_array = np.array(local_minima['y'])
z_array = np.array(local_minima['z'])
x_array = np.array(local_minima['x'])

y_range_mask = (z_array >= -0.1) & (z_array <= 0.1)

x_blue = x_array[y_range_mask]
y_blue = y_array[y_range_mask]
z_blue = z_array[y_range_mask]

# Remove noise using statistical thresholds
mean_y, std_y = np.mean(y_blue), np.std(y_blue)
mean_z, std_z = np.mean(z_blue), np.std(z_blue)

denoised_mask = (np.abs(y_blue - mean_y) <= 2 * std_y) & (np.abs(z_blue - mean_z) <= 2 * std_z)

x_blue, y_blue, z_blue = x_blue[denoised_mask], y_blue[denoised_mask], z_blue[denoised_mask]

# Apply DBSCAN clustering
data_points = np.vstack((y_blue, z_blue, x_blue)).T
clustering = DBSCAN(eps=0.2, min_samples=3).fit(data_points)

# Keep only the largest cluster
labels, counts = np.unique(clustering.labels_, return_counts=True)
main_cluster_label = labels[np.argmax(counts)]
main_cluster_mask = (clustering.labels_ == main_cluster_label)

x_blue, y_blue, z_blue = x_blue[main_cluster_mask], y_blue[main_cluster_mask], z_blue[main_cluster_mask]

# PCA-based filtering to remove outliers
def pca_filter(x, y, z, threshold=1.2):
    points = np.vstack((x, y, z)).T
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(points)

    mask = np.ones(len(points), dtype=bool)
    for i in range(3):
        q1, q3 = np.percentile(transformed[:, i], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask &= (transformed[:, i] >= lower_bound) & (transformed[:, i] <= upper_bound)
    return mask

pca_mask = pca_filter(x_blue, y_blue, z_blue, threshold=1.2)
x_blue, y_blue, z_blue = x_blue[pca_mask], y_blue[pca_mask], z_blue[pca_mask]

# Mahalanobis distance filtering
def mahalanobis_filter(x, y, z, threshold=3.0):
    points = np.vstack((x, y, z)).T
    mean = np.mean(points, axis=0)
    cov_inv = np.linalg.inv(np.cov(points, rowvar=False))
    distances = np.array([distance.mahalanobis(p, mean, cov_inv) for p in points])
    return distances < threshold

mahalanobis_mask = mahalanobis_filter(x_blue, y_blue, z_blue, threshold=3.0)
x_blue, y_blue, z_blue = x_blue[mahalanobis_mask], y_blue[mahalanobis_mask], z_blue[mahalanobis_mask]

# Sort points along PCA direction
points = np.vstack((x_blue, y_blue, z_blue)).T
pca = PCA(n_components=1)
order = np.argsort(pca.fit_transform(points).ravel())

x_ordered = x_blue[order]
y_ordered = y_blue[order]
z_ordered = z_blue[order]

# Fit a smooth 3D spline curve
tck, u = splprep([x_ordered, y_ordered, z_ordered], s=0.02)
u_fine = np.linspace(0, 1, 500)
x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

# Plot refined blue points and smooth curve
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=y_blue, y=z_blue, z=x_blue,
    mode='markers',
    marker=dict(color='blue', size=6),
    name='Refined Blue Points',
    hoverinfo='text',
    text=[f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(x_blue, y_blue, z_blue)]
))

fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Smooth Fit Curve'
))

fig.update_layout(
    title='Refined Blue Points with Fitted Smooth Curve',
    scene=dict(
        xaxis_title='Y',
        yaxis_title='Z',
        zaxis_title='X (Height)',
        aspectmode='data'
    )
)

fig.show()


# Load the spine mesh and split into parts
spine_mesh_path = 'Spine_NIH3D.stl'
if not os.path.isfile(spine_mesh_path):
    raise FileNotFoundError(f"Spine mesh file '{spine_mesh_path}' not found. Please provide the correct path.")

mesh = trimesh.load(spine_mesh_path)

# If empty faces, use convex hull
if mesh.faces.shape[0] == 0:
    mesh = mesh.convex_hull

# Split the mesh into separate components
spine_parts = mesh.split(only_watertight=False)

# Check if we got 24 parts
if len(spine_parts) == 24:
    print("✓ Spine split into 24 parts.")
else:
    print(f"⚠️ Warning: Got {len(spine_parts)} parts, not 24.")

# Save parts in order
spine_objects = []
for i, part in enumerate(spine_parts):
    spine_objects.append(part)

# Create Plotly figure for spine parts
fig = go.Figure()

for idx, part in enumerate(spine_objects):
    vertices = part.vertices
    faces = part.faces

    # Add mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5,
        name=f'Vertebra {idx + 1}'
    ))

    # Compute centroid for labeling
    centroid = part.centroid

    # Add label
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
        mode='text',
        text=[str(idx + 1)],
        textposition='top center',
        showlegend=False
    ))

fig.update_layout(
    title="24 Vertebrae Meshes with Labels",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()

# Save the vertebra numbers in order as they appeared
vertebra_order = [f"Vertebra {i+1}" for i in range(len(spine_objects))]

print("Vertebra order as visualized:\n", vertebra_order)

# PCA alignment and scaling of spine parts to fitted curve
curve_points = np.vstack((y_smooth, z_smooth, x_smooth)).T

pca_curve = PCA(n_components=3)
pca_curve.fit(curve_points)
curve_direction = pca_curve.components_[0]

all_spine_vertices = np.vstack([part.vertices for part in spine_objects])

pca_spine = PCA(n_components=3)
pca_spine.fit(all_spine_vertices)
spine_direction = pca_spine.components_[0]

rotation_axis = np.cross(spine_direction, curve_direction)
rotation_axis /= np.linalg.norm(rotation_axis)
rotation_angle = np.arccos(np.clip(np.dot(spine_direction, curve_direction), -1.0, 1.0))

K = np.array([
    [0, -rotation_axis[2], rotation_axis[1]],
    [rotation_axis[2], 0, -rotation_axis[0]],
    [-rotation_axis[1], rotation_axis[0], 0]
])
R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)

for part in spine_objects:
    part.vertices = (part.vertices @ R.T)

spine_min = np.min(np.vstack([p.vertices for p in spine_objects]), axis=0)
spine_max = np.max(np.vstack([p.vertices for p in spine_objects]), axis=0)
spine_center = (spine_min + spine_max) / 2
spine_size = np.linalg.norm(spine_max - spine_min)

curve_min = np.min(curve_points, axis=0)
curve_max = np.max(curve_points, axis=0)
curve_center = (curve_min + curve_max) / 2
curve_size = np.linalg.norm(curve_max - curve_min)

scale_factor = curve_size / spine_size

for part in spine_objects:
    part.vertices = (part.vertices - spine_center) * scale_factor + curve_center

# Visualize aligned spine with fitted curve
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Fitted Red Curve'
))

for part in spine_objects:
    vertices = part.vertices
    faces = part.faces
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5
    ))

fig.update_layout(
    title="Aligned Spine with Red Curve",
    scene=dict(aspectmode='data')
)

fig.show()

# Replace vertebra positions with numbered points
vertebrae_positions = np.vstack((y_smooth, z_smooth, x_smooth)).T

for i, part in enumerate(spine_objects):
    center = part.vertices.mean(axis=0)
    offset = vertebrae_positions[i] - center
    part.vertices += offset

# Visualize fitted spine onto vertebrae points
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Fitted Red Curve'
))

fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='markers+text',
    marker=dict(color='blue', size=5),
    text=[str(i + 1) for i in range(len(vertebrae_positions))],
    textposition='top center',
    name='Blue Vertebra Points'
))

for part in spine_objects:
    vertices = part.vertices
    faces = part.faces
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.7
    ))

fig.update_layout(
    title="Fitted Spine onto 24 Blue Vertebrae Points",
    scene=dict(aspectmode='data')
)

fig.show()
