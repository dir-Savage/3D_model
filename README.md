
---

## Endpoints

### 1. Process File

- **Endpoint**: `/process`  
- **Method**: `POST`  
- **Description**: Accepts a base64-encoded `.glb` file, processes it to generate 3D mesh visualizations, and returns base64-encoded PNG images of the results.

#### Request Body (JSON):

```json
{
  "file": "<base64-encoded-string>",
  "filename": "<filename-with-extension>"
}
```

### Response Body (JSON):
Success (HTTP 200)

```json
{
  "status": "success",
  "results": {
    "preprocessed_mesh": "<base64-encoded-png>",
    "heatmap": "<base64-encoded-png>",
    "local_minima": "<base64-encoded-png>",
    "fitted_curve": "<base64-encoded-png>",
    "synthetic_spine": "<base64-encoded-png>",
    "aligned_spine": "<base64-encoded-png>",
    "final_spine": "<base64-encoded-png>",
    "message": "Processing completed successfully"
  }
}
```


### Error message
```json
{
  "error": "<error-message>"
}
```
