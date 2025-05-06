import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import plotly.graph_objects as go
import trimesh

# Load the custom 3D object
# Replace 'model.obj' with the actual path to your 3D object file
mesh = trimesh.load_mesh("model.obj")

# Extract vertices, faces, and vertex colors from the mesh
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# Extract vertex colors if available
if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize colors to [0, 1]
else:
    colors = None

# Define cutting thresholds for x-axis (remove 40% of x-axis range)
x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
x_cut_threshold = x_min + 0.5 * (x_max - x_min)  # Cut the first 40% of the x-axis

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

# Scale down the object size (reduce size by 50%)
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

# Print a confirmation that the result is saved
print("The preprocessed model is saved in the variable 'preprocessed'.")


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

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


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# Generate contour lines in Y-direction
contour_levels = np.linspace(np.min(y), np.max(y), 25)  
contour_points = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 100  
    contour_points['x'].extend(x_grid[mask])  
    contour_points['y'].extend(y_mesh[mask])  
    contour_points['z'].extend(z_mesh[mask])  

# Convert extracted contour points into NumPy array for easier processing
contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z']]).T

# Identify local minima points
local_minima = {'x': [], 'y': [], 'z': []}
for i, (xi, yi, zi) in enumerate(contour_array):
    neighbors = contour_array[np.abs(contour_array[:, 1] - yi) < (np.max(y) - np.min(y)) / 50]  # Nearby points in Y
    neighbors = neighbors[np.abs(neighbors[:, 2] - zi) < (np.max(z) - np.min(z)) / 50]  # Nearby points in Z

    if all(xi <= neighbors[:, 0]):  # Check if xi is a local minimum
        local_minima['x'].append(xi)
        local_minima['y'].append(yi)
        local_minima['z'].append(zi)

# Create figure (contours only)
fig = go.Figure()

# Add extracted contour as a scatter plot
fig.add_trace(go.Scatter3d(
    x=contour_points['y'],  
    y=contour_points['z'],  
    z=contour_points['x'],  
    mode='markers',
    marker=dict(color='black', size=3),
    name="Contours",
    hoverinfo="text",
    text=[f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(contour_points['x'], contour_points['y'], contour_points['z'])]
))

# Add local minima points in red
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

# Adjust layout
fig.update_layout(
    title="Contour Lines with Highlighted Local Minima",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

# Show the plot
fig.show()


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# Generate contour lines in Y-direction
contour_levels = np.linspace(np.min(y), np.max(y), 25)  
contour_points = {'x': [], 'y': [], 'z': [], 'level': []}  # Store contour level info

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 100  
    contour_points['x'].extend(x_grid[mask])  
    contour_points['y'].extend(y_mesh[mask])  
    contour_points['z'].extend(z_mesh[mask])  
    contour_points['level'].extend([level] * np.sum(mask))  # Store level info

# Convert extracted contour points into NumPy array
contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T

# Identify local minima per contour level
local_minima = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    # Get only points from the current contour level
    level_mask = contour_array[:, 3] == level
    level_points = contour_array[level_mask]

    for i, (xi, yi, zi, _) in enumerate(level_points):
        # Find neighbors within the same contour level
        neighbors = level_points[np.abs(level_points[:, 1] - yi) < (np.max(y) - np.min(y)) / 50]
        neighbors = neighbors[np.abs(neighbors[:, 2] - zi) < (np.max(z) - np.min(z)) / 50]

        # Ensure xi is a local minimum in this level
        if np.all(xi <= neighbors[:, 0]):  
            local_minima['x'].append(xi)
            local_minima['y'].append(yi)
            local_minima['z'].append(zi)

# Create figure (contours only)
fig = go.Figure()

# Add extracted contour as a scatter plot
fig.add_trace(go.Scatter3d(
    x=contour_points['y'],  
    y=contour_points['z'],  
    z=contour_points['x'],  
    mode='markers',
    marker=dict(color='black', size=3),
    name="Contours",
    hoverinfo="text",
    text=[f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(contour_points['x'], contour_points['y'], contour_points['z'])]
))

# Add local minima points in red (highlight per contour level)
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

# Adjust layout
fig.update_layout(
    title="Contour Lines with Highlighted Local Minima (Per Level)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

# Show the plot
fig.show()


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# Generate contour lines in Y-direction
contour_levels = np.linspace(np.min(y), np.max(y), 35)  
contour_points = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 200  
    contour_points['x'].extend(x_grid[mask])  
    contour_points['y'].extend(y_mesh[mask])  
    contour_points['z'].extend(z_mesh[mask])  

# Convert extracted contour points into NumPy array for easier processing
contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z']]).T

# Identify local minima points
local_minima = {'x': [], 'y': [], 'z': []}
for i, (xi, yi, zi) in enumerate(contour_array):
    neighbors = contour_array[np.abs(contour_array[:, 1] - yi) < (np.max(y) - np.min(y)) / 50]  # Nearby points in Y
    neighbors = neighbors[np.abs(neighbors[:, 2] - zi) < (np.max(z) - np.min(z)) / 50]  # Nearby points in Z

    if all(xi <= neighbors[:, 0]):  # Check if xi is a local minimum
        local_minima['x'].append(xi)
        local_minima['y'].append(yi)
        local_minima['z'].append(zi)

# Create figure (contours only)
fig = go.Figure()



# Add local minima points in red
fig.add_trace(go.Scatter3d(
    x=local_minima['y'],  
    y=local_minima['z'],  
    z=local_minima['x'],  
    mode='markers',
    marker=dict(color='red', size=3, symbol='circle'),
    name="Local Minima",
    hoverinfo="text",
    text=[f"Min X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(local_minima['x'], local_minima['y'], local_minima['z'])]
))

# Adjust layout
fig.update_layout(
    title="Contour Lines with Highlighted Local Minima",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

# Show the plot
fig.show()


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# âœ… Increase contour resolution
contour_levels = np.linspace(np.min(y), np.max(y), 100)  # More contour lines
contour_points = {'x': [], 'y': [], 'z': [], 'level': []}

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 110
    contour_points['x'].extend(x_grid[mask])
    contour_points['y'].extend(y_mesh[mask])
    contour_points['z'].extend(z_mesh[mask])
    contour_points['level'].extend([level] * np.sum(mask))

# Convert to array for easier handling
contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T

# âœ… Dynamically adjust neighbor tolerance
level_spacing_y = (np.max(y) - np.min(y)) / len(contour_levels)
level_spacing_z = (np.max(z) - np.min(z)) / len(contour_levels)

# Detect local minima
local_minima = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    level_mask = contour_array[:, 3] == level
    level_points = contour_array[level_mask]

    if len(level_points) == 0:
        continue

    minima_found = False

    for i, (xi, yi, zi, _) in enumerate(level_points):
        neighbors = level_points[
            (np.abs(level_points[:, 1] - yi) < level_spacing_y * 1.5) &
            (np.abs(level_points[:, 2] - zi) < level_spacing_z * 1.5)
        ]

        if np.all(xi <= neighbors[:, 0]):
            local_minima['x'].append(xi)
            local_minima['y'].append(yi)
            local_minima['z'].append(zi)
            minima_found = True

    if not minima_found:
        min_idx = np.argmin(level_points[:, 0])
        local_minima['x'].append(level_points[min_idx, 0])
        local_minima['y'].append(level_points[min_idx, 1])
        local_minima['z'].append(level_points[min_idx, 2])

# âœ… Plot
fig = go.Figure()

# Black contour points
fig.add_trace(go.Scatter3d(
    x=contour_points['y'],
    y=contour_points['z'],
    z=contour_points['x'],
    mode='markers',
    marker=dict(color='black', size=3),
    name="Contours",
    hoverinfo="text",
    text=[f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(contour_points['x'], contour_points['y'], contour_points['z'])]
))

# Red local minima points
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
    title="Contour Lines with Local Minima",
    scene=dict(
        xaxis_title="Y (Horizontal)",
        yaxis_title="Z (Vertical)",
        zaxis_title="X (Depth/Height)",
        aspectmode="data"
    ),
    width=1000,
    height=800
)

fig.show()


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# Generate contour lines in Y-direction
contour_levels = np.linspace(np.min(y), np.max(y), 100)  
contour_points = {'x': [], 'y': [], 'z': [], 'level': []}  # Store contour level info

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 110  
    contour_points['x'].extend(x_grid[mask])  
    contour_points['y'].extend(y_mesh[mask])  
    contour_points['z'].extend(z_mesh[mask])  
    contour_points['level'].extend([level] * np.sum(mask))  # Store level info

# Convert extracted contour points into NumPy array
contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T

# âœ… Dynamically adjust neighbor tolerance
level_spacing_y = (np.max(y) - np.min(y)) / len(contour_levels)
level_spacing_z = (np.max(z) - np.min(z)) / len(contour_levels)

# Detect local minima
local_minima = {'x': [], 'y': [], 'z': []}

for level in contour_levels:
    level_mask = contour_array[:, 3] == level
    level_points = contour_array[level_mask]

    if len(level_points) == 0:
        continue

    minima_found = False

    for i, (xi, yi, zi, _) in enumerate(level_points):
        neighbors = level_points[
            (np.abs(level_points[:, 1] - yi) < level_spacing_y * 1.5) &
            (np.abs(level_points[:, 2] - zi) < level_spacing_z * 1.5)
        ]

        if np.all(xi <= neighbors[:, 0]):
            local_minima['x'].append(xi)
            local_minima['y'].append(yi)
            local_minima['z'].append(zi)
            minima_found = True

    if not minima_found:
        min_idx = np.argmin(level_points[:, 0])
        local_minima['x'].append(level_points[min_idx, 0])
        local_minima['y'].append(level_points[min_idx, 1])
        local_minima['z'].append(level_points[min_idx, 2])

# Create figure (contours only)
fig = go.Figure()



# Add local minima points in red (highlight per contour level)
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

# Adjust layout
fig.update_layout(
    title="Contour Lines with Highlighted Local Minima (Every Line)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

# Show the plot
fig.show()


import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

# Assume 'preprocessed' contains the 3D vertices
vertices = preprocessed['vertices']
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create a 2D grid for y and z
grid_resolution = 100
y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)

# Interpolate x-values (height) onto the grid
x_grid = griddata((y, z), x, (y_mesh, z_mesh), method='linear')

# âœ… Increase contour line resolution here
contour_levels = np.linspace(np.min(y), np.max(y), 100)  # 100 lines now

contour_points = {'x': [], 'y': [], 'z': [], 'level': []}

for level in contour_levels:
    mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 110
    contour_points['x'].extend(x_grid[mask])
    contour_points['y'].extend(y_mesh[mask])
    contour_points['z'].extend(z_mesh[mask])
    contour_points['level'].extend([level] * np.sum(mask))

contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T

# Filtering and alignment remain unchanged
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

# Plot
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


import numpy as np
import plotly.graph_objects as go

# Assume 'local_minima' is already defined and populated
# Define the condition for blue points (Z between -0.1 and 0.1)
y_range_mask = (np.array(local_minima['z']) >= -0.1) & (np.array(local_minima['z']) <= 0.1)

# Extract blue points
x_blue = np.array(local_minima['x'])[y_range_mask]
y_blue = np.array(local_minima['y'])[y_range_mask]
z_blue = np.array(local_minima['z'])[y_range_mask]

# Remove noise using statistical thresholds
# Calculate mean and standard deviation for the Y and Z axes
mean_y, std_y = np.mean(y_blue), np.std(y_blue)
mean_z, std_z = np.mean(z_blue), np.std(z_blue)

# Keep points within 2 standard deviations
denoised_mask = (np.abs(y_blue - mean_y) <= 2 * std_y) & (np.abs(z_blue - mean_z) <= 2 * std_z)

# Create a new figure for denoised blue points
denoised_fig = go.Figure()

# Add denoised blue points
denoised_fig.add_trace(go.Scatter3d(
    x=y_blue[denoised_mask],
    y=z_blue[denoised_mask],
    z=x_blue[denoised_mask],
    mode='markers',
    marker=dict(color='blue', size=6, symbol='circle'),
    name="Denoised Blue Points",
    hoverinfo="text",
    text=[f"Min X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(
        x_blue[denoised_mask],
        y_blue[denoised_mask],
        z_blue[denoised_mask]
    )]
))

# Adjust layout for the new figure
denoised_fig.update_layout(
    title="Denoised Blue Points (Y between -0.1 and 0.1)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

# Show the denoised plot
denoised_fig.show()

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import distance

# Assume 'local_minima' is already defined and populated
y_range_mask = (np.array(local_minima['z']) >= -0.1) & (np.array(local_minima['z']) <= 0.1)

x_blue = np.array(local_minima['x'])[y_range_mask]
y_blue = np.array(local_minima['y'])[y_range_mask]
z_blue = np.array(local_minima['z'])[y_range_mask]

# Remove noise using statistical thresholds
mean_y, std_y = np.mean(y_blue), np.std(y_blue)
mean_z, std_z = np.mean(z_blue), np.std(z_blue)

denoised_mask = (np.abs(y_blue - mean_y) <= 2 * std_y) & (np.abs(z_blue - mean_z) <= 2 * std_z)

x_blue, y_blue, z_blue = x_blue[denoised_mask], y_blue[denoised_mask], z_blue[denoised_mask]

# Apply DBSCAN with increased sensitivity to identify clusters
data_points = np.vstack((y_blue, z_blue, x_blue)).T
clustering = DBSCAN(eps=0.2, min_samples=3).fit(data_points)

# Keep only the largest cluster
labels, counts = np.unique(clustering.labels_, return_counts=True)
main_cluster_label = labels[np.argmax(counts)]

# Extract the main cluster
main_cluster_mask = (clustering.labels_ == main_cluster_label)
x_blue, y_blue, z_blue = x_blue[main_cluster_mask], y_blue[main_cluster_mask], z_blue[main_cluster_mask]

# Apply PCA for alignment and remove extremes
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

# Apply Mahalanobis distance to remove further outliers
def mahalanobis_filter(x, y, z, threshold=3.0):
    points = np.vstack((x, y, z)).T
    mean = np.mean(points, axis=0)
    cov_inv = np.linalg.inv(np.cov(points, rowvar=False))

    distances = np.array([distance.mahalanobis(p, mean, cov_inv) for p in points])
    return distances < threshold

mahalanobis_mask = mahalanobis_filter(x_blue, y_blue, z_blue, threshold=3.0)

x_blue, y_blue, z_blue = x_blue[mahalanobis_mask], y_blue[mahalanobis_mask], z_blue[mahalanobis_mask]

# Create a new figure for refined points
denoised_fig = go.Figure()

denoised_fig.add_trace(go.Scatter3d(
    x=y_blue,
    y=z_blue,
    z=x_blue,
    mode='markers',
    marker=dict(color='blue', size=6, symbol='circle'),
    name="Refined Blue Points",
    hoverinfo="text",
    text=[f"Min X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(x_blue, y_blue, z_blue)]
))

denoised_fig.update_layout(
    title="Refined Blue Points (DBSCAN + PCA + Mahalanobis Filtered)",
    scene=dict(
        xaxis=dict(title="Y"),
        yaxis=dict(title="Z"),
        zaxis=dict(title="X (Height)"),
        aspectmode="data"
    )
)

denoised_fig.show()


import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.interpolate import splprep, splev

# ==== Assume 'local_minima' is already defined ====
# It should be a dictionary with 'x', 'y', and 'z' keys containing lists or arrays

# Step 1: Filter by Z range
y_range_mask = (np.array(local_minima['z']) >= -0.1) & (np.array(local_minima['z']) <= 0.1)

x_blue = np.array(local_minima['x'])[y_range_mask]
y_blue = np.array(local_minima['y'])[y_range_mask]
z_blue = np.array(local_minima['z'])[y_range_mask]

# Step 2: Remove noise with statistical thresholds
mean_y, std_y = np.mean(y_blue), np.std(y_blue)
mean_z, std_z = np.mean(z_blue), np.std(z_blue)

denoised_mask = (np.abs(y_blue - mean_y) <= 2 * std_y) & (np.abs(z_blue - mean_z) <= 2 * std_z)

x_blue, y_blue, z_blue = x_blue[denoised_mask], y_blue[denoised_mask], z_blue[denoised_mask]

# Step 3: Apply DBSCAN to identify clusters
data_points = np.vstack((y_blue, z_blue, x_blue)).T
clustering = DBSCAN(eps=0.2, min_samples=3).fit(data_points)

# Keep only the largest cluster
labels, counts = np.unique(clustering.labels_, return_counts=True)
main_cluster_label = labels[np.argmax(counts)]
main_cluster_mask = (clustering.labels_ == main_cluster_label)

x_blue, y_blue, z_blue = x_blue[main_cluster_mask], y_blue[main_cluster_mask], z_blue[main_cluster_mask]

# Step 4: PCA-based filtering to remove outliers
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

# Step 5: Mahalanobis distance filtering
def mahalanobis_filter(x, y, z, threshold=3.0):
    points = np.vstack((x, y, z)).T
    mean = np.mean(points, axis=0)
    cov_inv = np.linalg.inv(np.cov(points, rowvar=False))
    distances = np.array([distance.mahalanobis(p, mean, cov_inv) for p in points])
    return distances < threshold

mahalanobis_mask = mahalanobis_filter(x_blue, y_blue, z_blue, threshold=3.0)
x_blue, y_blue, z_blue = x_blue[mahalanobis_mask], y_blue[mahalanobis_mask], z_blue[mahalanobis_mask]

# Step 6: Sort points along PCA direction
points = np.vstack((x_blue, y_blue, z_blue)).T
pca = PCA(n_components=1)
order = np.argsort(pca.fit_transform(points).ravel())

x_ordered = x_blue[order]
y_ordered = y_blue[order]
z_ordered = z_blue[order]

# Step 7: Fit a smooth 3D spline curve
tck, u = splprep([x_ordered, y_ordered, z_ordered], s=0.02)
u_fine = np.linspace(0, 1, 500)
x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

# Step 8: Plot everything
fig = go.Figure()

# Refined blue points
fig.add_trace(go.Scatter3d(
    x=y_blue, y=z_blue, z=x_blue,
    mode='markers',
    marker=dict(color='blue', size=6),
    name='Refined Blue Points',
    hoverinfo='text',
    text=[f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(x_blue, y_blue, z_blue)]
))

# Smooth red curve
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


# === New Cell: Extract and Plot Fitted Curve Only (Y in [-1, 1]) ===

# Evaluate the spline again if needed
u_fine = np.linspace(0, 1, 500)
x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

# Apply Y-range filter
curve_mask = (y_smooth >= -1.0) & (y_smooth <= 1.0)
x_smooth = np.array(x_smooth)[curve_mask]
y_smooth = np.array(y_smooth)[curve_mask]
z_smooth = np.array(z_smooth)[curve_mask]

# Plot only the filtered curve
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=5),
    name='Filtered Curve [-1 < Y < 1]'
))

fig.update_layout(
    title='Smoothed Fitted Curve (Y range limited to [-1, 1])',
    scene=dict(
        xaxis_title='Y',
        yaxis_title='Z',
        zaxis_title='X (Height)',
        aspectmode='data'
    )
)

fig.show()


# Vertebrae weights: Cervical(7), Thoracic(12), Lumbar(5)
weights = [2]*5 + [1.5]*12 + [1]*7  # ط§ظ„ط±ظ‚ط¨ط© ط£ظ‚ظ„طŒ ط§ظ„ط¶ظ‡ط± ط£ظƒط¨ط± ط´ظˆظٹط©طŒ ط§ظ„ظ‚ط·ظ†ظٹط© ط£ظƒط¨ط± ظƒظ…ط§ظ†
total_weight = sum(weights)
cumulative = np.cumsum(weights) / total_weight

# Generate 24 indices based on the proportional spacing
num_points = 24
# indices in ascending visual direction first
indices = np.round(cumulative * (len(x_smooth) - 1)).astype(int)

# Reverse to start numbering from bottom to top (ط²ظٹ ط§ظ„ظƒظˆط¯ ط¨طھط§ط¹ظƒ)
indices = indices[::-1]

# Extract corresponding coordinates
x_vertebrae = x_smooth[indices]
y_vertebrae = y_smooth[indices]
z_vertebrae = z_smooth[indices]

# Plot the fitted curve and numbered points
fig = go.Figure()

# Fitted curve
fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Fitted Curve'
))

# Numbered vertebrae points
fig.add_trace(go.Scatter3d(
    x=y_vertebrae, y=z_vertebrae, z=x_vertebrae,
    mode='markers+text',
    marker=dict(color='blue', size=5),
    text=[str(i + 1) for i in range(num_points)],
    textposition='top center',
    name='Vertebrae Points'
))

fig.update_layout(
    title='24 Numbered Vertebrae Points (Proportional Spacing, Correct Order)',
    scene=dict(
        xaxis_title='Y',
        yaxis_title='Z',
        zaxis_title='X (Height)',
        aspectmode='data'
    )
)

fig.show()

import trimesh
import numpy as np
import plotly.graph_objects as go

# Load the mesh
mesh = trimesh.load('Spine_NIH3D.stl')

# If empty faces, use convex hull
if mesh.faces.shape[0] == 0:
    mesh = mesh.convex_hull

# Split the mesh into separate components
spine_parts = mesh.split(only_watertight=False)

# Check if we got 24 parts
if len(spine_parts) == 24:
    print("âœ… Spine split into 24 parts.")
else:
    print(f"âڑ ï¸ڈ Warning: Got {len(spine_parts)} parts, not 24.")

# Save parts in order
spine_objects = []
for i, part in enumerate(spine_parts):
    spine_objects.append(part)

# Create Plotly figure
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

# Print the list
print("ًں¦´ Vertebra order as visualized:\n", vertebra_order)


from sklearn.decomposition import PCA

# Assume x_smooth, y_smooth, z_smooth ظ…ظˆط¬ظˆط¯ظٹظ† ط¬ط§ظ‡ط²ظٹظ†

# Stack fitted curve points
curve_points = np.vstack((y_smooth, z_smooth, x_smooth)).T

# PCA on the curve to get direction
pca_curve = PCA(n_components=3)
pca_curve.fit(curve_points)
curve_direction = pca_curve.components_[0]

# Stack all spine vertices
all_spine_vertices = np.vstack([part.vertices for part in spine_objects])

# PCA on the spine
pca_spine = PCA(n_components=3)
pca_spine.fit(all_spine_vertices)
spine_direction = pca_spine.components_[0]

# Calculate rotation
rotation_axis = np.cross(spine_direction, curve_direction)
rotation_axis /= np.linalg.norm(rotation_axis)
rotation_angle = np.arccos(np.clip(np.dot(spine_direction, curve_direction), -1.0, 1.0))

# Rodrigues' rotation formula
K = np.array([
    [0, -rotation_axis[2], rotation_axis[1]],
    [rotation_axis[2], 0, -rotation_axis[0]],
    [-rotation_axis[1], rotation_axis[0], 0]
])
R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)

# Apply rotation to each part
for part in spine_objects:
    part.vertices = (part.vertices @ R.T)

# Now scale and translate the spine
spine_min = np.min(np.vstack([p.vertices for p in spine_objects]), axis=0)
spine_max = np.max(np.vstack([p.vertices for p in spine_objects]), axis=0)
spine_center = (spine_min + spine_max) / 2
spine_size = np.linalg.norm(spine_max - spine_min)

curve_min = np.min(curve_points, axis=0)
curve_max = np.max(curve_points, axis=0)
curve_center = (curve_min + curve_max) / 2
curve_size = np.linalg.norm(curve_max - curve_min)

scale_factor = curve_size / spine_size

# Apply scaling and translation
for part in spine_objects:
    part.vertices = (part.vertices - spine_center) * scale_factor + curve_center

# Visualize spine fitted to red curve
fig = go.Figure()

# Red fitted curve
fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Fitted Red Curve'
))

# Aligned spine parts
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


# Assume x_vertebrae, y_vertebrae, z_vertebrae ظ…ظˆط¬ظˆط¯ظٹظ† (24 ظ†ظ‚ط·ط©)

vertebrae_positions = np.vstack((y_vertebrae, z_vertebrae, x_vertebrae)).T

# Replace positions
for i, part in enumerate(spine_objects):
    # Move each vertebra center to the corresponding blue point
    center = part.vertices.mean(axis=0)
    offset = vertebrae_positions[i] - center
    part.vertices += offset

# Visualization
fig = go.Figure()

# Red fitted curve
fig.add_trace(go.Scatter3d(
    x=y_smooth, y=z_smooth, z=x_smooth,
    mode='lines',
    line=dict(color='red', width=4),
    name='Fitted Red Curve'
))

# Blue points
fig.add_trace(go.Scatter3d(
    x=y_vertebrae, y=z_vertebrae, z=x_vertebrae,
    mode='markers+text',
    marker=dict(color='blue', size=5),
    text=[str(i + 1) for i in range(len(vertebrae_positions))],
    textposition='top center',
    name='Blue Vertebra Points'
))

# Fitted vertebrae
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












import trimesh
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import plotly.graph_objects as go

# Load the spine mesh
mesh = trimesh.load('Spine_NIH3D.stl')
vertices = mesh.vertices
faces = mesh.faces

x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Assuming y_blue and z_blue are already defined arrays
# Sort and fit polynomial to get the red curve
sorted_indices = np.argsort(y_blue)
y_sorted = y_blue[sorted_indices]
z_sorted = z_blue[sorted_indices]

poly = Polynomial.fit(y_sorted, z_sorted, deg=6)
smooth_y = np.linspace(y_sorted.min(), y_sorted.max(), 500)
smooth_z = poly(smooth_y)

# Generate constant X values just to plot the red line (not aligned)
curve_x = np.full_like(smooth_y, fill_value=np.mean(x))  # just to visualize

# --- First Figure: Spine Mesh Only ---
fig_spine = go.Figure()
fig_spine.add_trace(go.Mesh3d(
    x=x, y=y, z=z,
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color='lightblue',
    opacity=0.8,
    name='Spine'
))

fig_spine.update_layout(
    title="3D Spine Mesh Only",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
    ),
    width=1000,
    height=800
)

# --- Second Figure: Red Curve Only ---
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter3d(
    x=curve_x,
    y=smooth_y,
    z=smooth_z,
    mode='lines',
    line=dict(color='red', width=10),
    marker=dict(size=4),
    name='Red Curve'
))

fig_curve.update_layout(
    title="3D Red Curve Only",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
    ),
    width=1000,
    height=800
)

# Show the plots
fig_spine.show()
fig_curve.show()


import trimesh

# Load the full spine mesh
mesh = trimesh.load('Spine_NIH3D.stl')

# Split it into parts (likely vertebrae)
vertebrae = mesh.split(only_watertight=False)

# Show how many parts were found
print(f"Found {len(vertebrae)} parts")

# Show them one by one or manipulate individually
for i, part in enumerate(vertebrae):
    part.show()