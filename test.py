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

# Function to load .glb mesh
def load_glb_mesh(glb_path):
    try:
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(f"GLB file '{glb_path}' not found")
        mesh = trimesh.load_mesh(glb_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded GLB file is not a valid Trimesh object")
        return mesh
    except Exception as e:
        raise RuntimeError(f"Error loading GLB mesh: {str(e)}")

# Simulate spine mesh (24 vertebrae) if not available
def create_synthetic_spine():
    spine_parts = []
    for i in range(24):
        cube = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        cube.apply_translation([0, 0, i * 0.15])  # Stack vertically
        spine_parts.append(cube)
    return spine_parts

# Main processing function
def process_glb(glb_path):
    try:
        # Load GLB mesh
        mesh = load_glb_mesh(glb_path)
        
        # Extract vertices, faces, and colors
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Extract vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize to [0, 1]
        else:
            colors = np.ones((vertices.shape[0], 3)) * [0.5, 0.5, 0.5]  # Default gray
        
        # Define cutting thresholds
        x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        x_cut_threshold = x_min + 0.5 * (x_max - x_min)
        
        z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
        z_lower_threshold = z_min + 0.2 * (z_max - z_min)
        z_upper_threshold = z_max - 0.2 * (z_max - z_min)
        
        # Filter vertices
        keep_indices = np.where(
            (vertices[:, 0] > x_cut_threshold) &
            (vertices[:, 2] >= z_lower_threshold) &
            (vertices[:, 2] <= z_upper_threshold)
        )[0]
        
        if len(keep_indices) == 0:
            raise ValueError("No vertices remain after cutting")
        
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        new_vertices = vertices[keep_indices]
        
        # Filter faces
        new_faces = []
        new_colors = []
        for face in faces:
            if all(v_idx in index_map for v_idx in face):
                new_faces.append([index_map[v_idx] for v_idx in face])
                new_colors.append(colors[face])
        
        new_faces = np.array(new_faces)
        new_colors = np.array(new_colors) if new_colors else None
        
        # Scale down
        scaling_factor = 3.5
        new_vertices *= scaling_factor
        
        # Save preprocessed data
        preprocessed = {
            'vertices': new_vertices,
            'faces': new_faces,
            'colors': new_colors
        }
        
        # Visualize preprocessed mesh
        x, y, z = new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2]
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=new_faces[:, 0], j=new_faces[:, 1], k=new_faces[:, 2],
            vertexcolor=new_colors.reshape(-1, 3) if new_colors is not None else None,
            opacity=1.0
        ))
        padding = 1.0
        fig.update_layout(
            title="Preprocessed 3D Model from GLB",
            scene=dict(
                xaxis=dict(range=[np.min(x) - padding, np.max(x) + padding], title='X'),
                yaxis=dict(range=[np.min(y) - padding, np.max(y) + padding], title='Y'),
                zaxis=dict(range=[np.min(z) - padding, np.max(z) + padding], title='Z'),
            )
        )
        fig.show()
        
        # Topography heatmap
        grid_resolution = 100
        y_grid = np.linspace(np.min(y), np.max(y), grid_resolution)
        z_grid = np.linspace(np.min(z), np.max(z), grid_resolution)
        y_mesh, z_mesh = np.meshgrid(y_grid, z_grid)
        
        x_grid = griddata(
            points=(y, z),
            values=x,
            xi=(y_mesh, z_mesh),
            method='linear'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=x_grid, x=y_mesh, y=z_mesh,
            colorscale='Viridis',
            colorbar=dict(title="Height (X)")
        ))
        fig.update_layout(
            title="Heatmap Surface Topography",
            scene=dict(
                xaxis=dict(title="Y"),
                yaxis=dict(title="Z"),
                zaxis=dict(title="X (Height)"),
                aspectmode="data"
            )
        )
        fig.show()
        
        # Contour lines and local minima
        contour_levels = np.linspace(np.min(y), np.max(y), 100)
        contour_points = {'x': [], 'y': [], 'z': [], 'level': []}
        
        for level in contour_levels:
            mask = np.abs(y_mesh - level) < (np.max(y) - np.min(y)) / 110
            contour_points['x'].extend(x_grid[mask])
            contour_points['y'].extend(y_mesh[mask])
            contour_points['z'].extend(z_mesh[mask])
            contour_points['level'].extend([level] * np.sum(mask))
        
        contour_array = np.array([contour_points['x'], contour_points['y'], contour_points['z'], contour_points['level']]).T
        
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=local_minima['y'], y=local_minima['z'], z=local_minima['x'],
            mode='markers',
            marker=dict(color='red', size=6),
            name="Local Minima",
            hoverinfo="text",
            text=[f"Min X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}" for x, y, z in zip(local_minima['x'], local_minima['y'], local_minima['z'])]
        ))
        fig.update_layout(
            title="Local Minima",
            scene=dict(
                xaxis=dict(title="Y"),
                yaxis=dict(title="Z"),
                zaxis=dict(title="X (Height)"),
                aspectmode="data"
            )
        )
        fig.show()
        
        # Filter and fit curve
        y_array = np.array(local_minima['y'])
        z_array = np.array(local_minima['z'])
        x_array = np.array(local_minima['x'])
        
        y_range_mask = (z_array >= -0.1) & (z_array <= 0.1)
        x_blue, y_blue, z_blue = x_array[y_range_mask], y_array[y_range_mask], z_array[y_range_mask]
        
        if len(x_blue) == 0:
            print("No points in Z range [-0.1, 0.1], skipping curve fitting")
            return
        
        # Denoise
        mean_y, std_y = np.mean(y_blue), np.std(y_blue)
        mean_z, std_z = np.mean(z_blue), np.std(z_blue)
        denoised_mask = (np.abs(y_blue - mean_y) <= 2 * std_y) & (np.abs(z_blue - mean_z) <= 2 * std_z)
        x_blue, y_blue, z_blue = x_blue[denoised_mask], y_blue[denoised_mask], z_blue[denoised_mask]
        
        if len(x_blue) < 3:
            print("Not enough points for clustering, skipping curve fitting")
            return
        
        # DBSCAN
        data_points = np.vstack((y_blue, z_blue, x_blue)).T
        clustering = DBSCAN(eps=0.2, min_samples=3).fit(data_points)
        labels, counts = np.unique(clustering.labels_, return_counts=True)
        if len(labels) == 0 or max(counts) < 3:
            print("No valid clusters found, skipping curve fitting")
            return
        main_cluster_label = labels[np.argmax(counts)]
        main_cluster_mask = (clustering.labels_ == main_cluster_label)
        x_blue, y_blue, z_blue = x_blue[main_cluster_mask], y_blue[main_cluster_mask], z_blue[main_cluster_mask]
        
        # PCA filter
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
        
        # Mahalanobis filter
        def mahalanobis_filter(x, y, z, threshold=3.0):
            points = np.vstack((x, y, z)).T
            mean = np.mean(points, axis=0)
            cov = np.cov(points, rowvar=False)
            if np.linalg.cond(cov) < 1e10:  # Check for singularity
                cov_inv = np.linalg.inv(cov)
                distances = np.array([(distance.mahalanobis(p, mean, cov_inv) if not np.isnan(distance.mahalanobis(p, mean, cov_inv)) else np.inf) for p in points])
                return distances < threshold
            return np.ones(len(points), dtype=bool)
        
        mahalanobis_mask = mahalanobis_filter(x_blue, y_blue, z_blue, threshold=3.0)
        x_blue, y_blue, z_blue = x_blue[mahalanobis_mask], y_blue[mahalanobis_mask], z_blue[mahalanobis_mask]
        
        if len(x_blue) < 3:
            print("Not enough points after filtering, skipping curve fitting")
            return
        
        # Fit spline
        points = np.vstack((x_blue, y_blue, z_blue)).T
        pca = PCA(n_components=1)
        order = np.argsort(pca.fit_transform(points).ravel())
        x_ordered = x_blue[order]
        y_ordered = y_blue[order]
        z_ordered = z_blue[order]
        
        tck, u = splprep([x_ordered, y_ordered, z_ordered], s=0.02)
        u_fine = np.linspace(0, 1, 500)
        x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
        
        # Visualize curve
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=y_blue, y=z_blue, z=x_blue,
            mode='markers',
            marker=dict(color='blue', size=6),
            name='Refined Points'
        ))
        fig.add_trace(go.Scatter3d(
            x=y_smooth, y=z_smooth, z=x_smooth,
            mode='lines',
            line=dict(color='red', width=4),
            name='Smooth Curve'
        ))
        fig.update_layout(
            title='Refined Points with Fitted Curve',
            scene=dict(
                xaxis_title='Y',
                yaxis_title='Z',
                zaxis_title='X (Height)',
                aspectmode='data'
            )
        )
        fig.show()
        
        # Simulate spine
        spine_objects = create_synthetic_spine()
        
        # Visualize spine
        fig = go.Figure()
        for idx, part in enumerate(spine_objects):
            vertices = part.vertices
            faces = part.faces
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                opacity=0.5,
                name=f'Vertebra {idx + 1}'
            ))
            centroid = part.centroid
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                mode='text',
                text=[str(idx + 1)],
                showlegend=False
            ))
        fig.update_layout(
            title="Synthetic Spine Vertebrae",
            scene=dict(aspectmode='data')
        )
        fig.show()
        
        # Align spine to curve
        curve_points = np.vstack((y_smooth, z_smooth, x_smooth)).T
        pca_curve = PCA(n_components=3)
        pca_curve.fit(curve_points)
        curve_direction = pca_curve.components_[0]
        
        all_spine_vertices = np.vstack([part.vertices for part in spine_objects])
        pca_spine = PCA(n_components=3)
        pca_spine.fit(all_spine_vertices)
        spine_direction = pca_spine.components_[0]
        
        rotation_axis = np.cross(spine_direction, curve_direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) if np.linalg.norm(rotation_axis) > 0 else np.array([0, 0, 1])
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
        spine_size = np.linalg.norm(spine_max - spine_min) if np.linalg.norm(spine_max - spine_min) > 0 else 1.0
        
        curve_min = np.min(curve_points, axis=0)
        curve_max = np.max(curve_points, axis=0)
        curve_center = (curve_min + curve_max) / 2
        curve_size = np.linalg.norm(curve_max - curve_min) if np.linalg.norm(curve_max - curve_min) > 0 else 1.0
        
        scale_factor = curve_size / spine_size
        
        for part in spine_objects:
            part.vertices = (part.vertices - spine_center) * scale_factor + curve_center
        
        # Visualize aligned spine
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=y_smooth, y=z_smooth, z=x_smooth,
            mode='lines',
            line=dict(color='red', width=4),
            name='Fitted Curve'
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
            title="Aligned Spine with Curve",
            scene=dict(aspectmode='data')
        )
        fig.show()
        
        # Fit spine to curve points
        vertebrae_positions = np.vstack((y_smooth, z_smooth, x_smooth)).T[::20][:24]  # Sample 24 points
        for i, part in enumerate(spine_objects):
            center = part.vertices.mean(axis=0)
            offset = vertebrae_positions[i] - center
            part.vertices += offset
        
        # Final visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=y_smooth, y=z_smooth, z=x_smooth,
            mode='lines',
            line=dict(color='red', width=4),
            name='Fitted Curve'
        ))
        fig.add_trace(go.Scatter3d(
            x=vertebrae_positions[:, 0], y=vertebrae_positions[:, 1], z=vertebrae_positions[:, 2],
            mode='markers+text',
            marker=dict(color='blue', size=5),
            text=[str(i + 1) for i in range(len(vertebrae_positions))],
            name='Vertebra Points'
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
            title="Fitted Spine onto Vertebrae Points",
            scene=dict(aspectmode='data')
        )
        fig.show()
        
    except Exception as e:
        print(f"Error processing GLB: {str(e)}")

# Example usage
glb_path = 'image.glb'
if os.path.isfile(glb_path):
    process_glb(glb_path)
else:
    print(f"GLB file '{glb_path}' not found. Please provide a valid .glb file path.")