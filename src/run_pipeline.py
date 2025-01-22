from utils import *

# # Chemins des images
# image_paths = ["data/raw/cubic_scene/IMG_3931.jpeg", "data/raw/cubic_scene/IMG_3937.jpeg"]

# # Étape 1 : Charger les images
# data_loader = DataLoader(image_paths)
# images = data_loader.load_images()

# # Étape 2 : Extraire les caractéristiques
# feature_extractor = FeatureExtractor(method="ORB")
# keypoints1, descriptors1 = feature_extractor.extract_features(images[0])
# keypoints2, descriptors2 = feature_extractor.extract_features(images[1])

# # Étape 3 : Associer les caractéristiques
# feature_matcher = FeatureMatcher(method="BF", cross_check=True)
# matches = feature_matcher.match_features(descriptors1, descriptors2)

# # Étape 4 : Reconstruction 3D
# points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
# points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
# intrinsic_matrix = np.eye(3)  # Remplacez par votre vraie matrice intrinsèque
# reconstructor = Reconstruction3D(intrinsic_matrix)
# F, mask = reconstructor.estimate_fundamental_matrix(points1, points2)
# proj_matrix1 = np.eye(3, 4)  # Matrice de projection de la première caméra
# proj_matrix2 = np.hstack((np.eye(3), np.array([[1], [0], [0]])))  # Exemple
# points_3d = reconstructor.triangulate_points(proj_matrix1, proj_matrix2, points1, points2)

# # Étape 5 : Visualisation
# visualizer = Visualizer()
# visualizer.display_point_cloud(points_3d)

import cv2
import numpy as np
import open3d as o3d

# Étape 1 : Charger les images stéréo
img_left = cv2.imread('data/raw/cubic_scene/IMG_3931.jpeg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('data/raw/cubic_scene/IMG_3937.jpeg', cv2.IMREAD_GRAYSCALE)

if img_left is None or img_right is None:
    raise FileNotFoundError("Impossible de charger les images. Vérifiez les chemins.")

# Étape 2 : Paramètres de la correspondance stéréo
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,  # Doit être multiple de 16
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Calculer la disparité
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# Étape 3 : Générer les points 3D
# Paramètres de la caméra (à ajuster selon votre configuration)
focale_mm = 4.25  # Distance focale en mm
largeur_capteur_mm = 4.8  # Largeur du capteur en mm
hauteur_capteur_mm = 3.6  # Hauteur du capteur en mm
largeur_image_pixels = 4032  # Largeur de l'image en pixels
hauteur_image_pixels = 3024  # Hauteur de l'image en pixels
baseline = 0.06  # Distance entre les caméras en mètres

# Calcul des paramètres intrinsèques
f_x = (focale_mm / largeur_capteur_mm) * largeur_image_pixels
f_y = (focale_mm / hauteur_capteur_mm) * hauteur_image_pixels
c_x = largeur_image_pixels / 2
c_y = hauteur_image_pixels / 2

Q = np.array([
    [1, 0, 0, -c_x],
    [0, -1, 0, c_y],
    [0, 0, 0, f_x],
    [0, 0, -1 / baseline, 0]
])

points_3d = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.cvtColor(cv2.imread('data/raw/cubic_scene/IMG_3931.jpeg'), cv2.COLOR_BGR2RGB)

# Masquer les points sans disparité
mask = disparity > disparity.min()
points_3d = points_3d[mask]
colors = colors[mask]

print(f"Min des coordonnées 3D : {np.min(points_3d, axis=0)}")
print(f"Max des coordonnées 3D : {np.max(points_3d, axis=0)}")


# Étape 4 : Créer et visualiser le nuage de points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normaliser les couleurs

o3d.visualization.draw_geometries([pcd], window_name="Nuage de points dense")
