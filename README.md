# Open3D lab on objects detection
### Dependancies
- Jupyter Notebook
- numpy
- open3d
- einops
- pyquaternion

Last Python version compatible with open3D: 3.11  

### Dataset
Download the data for the lab0 and lab1 :
[https://uncloud.univ-nantes.fr/index.php/s/oiMzTpqiBGniZng](https://uncloud.univ-nantes.fr/index.php/s/9W4ZjtkBLnKyp8m)


## Overview  
This lab exercise involved extrapolating useful features from a raw point cloud, this involved filtering and clustering points to tell objects apart and creating bounding boxes.  
![Screenshot 2024-11-29 184707](https://github.com/user-attachments/assets/a53d85d3-ba6d-480d-b636-ac7aa1d2ec2b)
The first step involved separating the ground from the objects, this was done through height data of the points, plus the estimated normals obtained by grouping neighboring points that could belong to the same surface.  
![Screenshot 2024-11-29 184732](https://github.com/user-attachments/assets/edf69be5-dd27-4b52-a9ca-5c156ee59352)

Next, we developed different functions that could perform the filtering of the features of the point cloud: `ground_filtering` and `object_filtering`, both based on a threshold system, discarding everything not belonging to ground or objects respectively.  
![Screenshot 2024-11-29 184755](https://github.com/user-attachments/assets/da9abbe3-17b3-4a79-ad9f-c098f6bdd104)
![Screenshot 2024-11-29 184807](https://github.com/user-attachments/assets/7b125124-f763-4f1e-a3b4-ef4d083e627a)

Once the ground was filtered, we could proceed to implement a function for clustering, grouping points likely belonging to the same object. This was done on a neighboring principle, introducing a threshold for a minimum number of points per cluster.  
![Screenshot 2024-11-29 184824](https://github.com/user-attachments/assets/dbc793b2-3900-489e-8a40-de64c478a06e)

At last we performed bounding boxes estimation using the built-in function `get_axis-aligned_bounding_boxes`
![Screenshot 2024-11-29 184853](https://github.com/user-attachments/assets/0cf4ddbc-f88f-44b8-b8fc-d405b83f7a1f)

