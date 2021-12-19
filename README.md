# High-Throughput Phenotyping of Cotton Plants

Phenotype measurements of a cotton plant (such as height and of a number of cotton bolls) are currently carried out by hand. This project's goal is to allow for automatic detection and measurement of those features. To accomplish its goals, the project implements image processing and machine learning techniques. The images are captured by cameras mounted on a robot or a tractor.

Solution: Currently designing and implementing a graph-based SLAM system to scan and detect the cotton bolls using RGB-D mapping, point cloud estimation techniques, and instance segmentation models.

This is currently an ongoing project!

## Data Acquisition

For this phase an ROS based SLAM System was designed which was assembled on a tractor.

Here's an overview of this system:

Two RGB-D Cameras were installed on a tractor so we would be able to scan the plants from top to bottom.
The cameras were being operated by a ROS system running on NVIDIA's Jetson machines. Using the RGB-D images and IMU information from the cameras we scanned 32 rows of cotton plants at different stages of growth during a 6 month period.

Here's a link to a video demonstrating an instance of the scanning sessions being operated on ROS:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=KzjfbDj-uP8
" target="_blank"><img src="http://img.youtube.com/vi/KzjfbDj-uP8/0.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="640" height="360" border="10" /></a>

