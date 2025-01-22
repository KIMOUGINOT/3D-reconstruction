import cv2
import numpy as np
import open3d as o3d

class DataLoader:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = []

    def load_images(self):
        """Charge les images depuis les chemins spécifiés."""
        self.images = [cv2.imread(path) for path in self.image_paths]
        return self.images

class FeatureExtractor:
    def __init__(self, method="ORB"):
        if method == "ORB":
            self.detector = cv2.ORB_create()
        elif method == "SIFT":
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError("Méthode inconnue. Utilisez 'ORB' ou 'SIFT'.")

    def extract_features(self, image):
        """Détecte les points clés et extrait les descripteurs."""
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

class FeatureMatcher:
    def __init__(self, method="BF", cross_check=True):
        if method == "BF":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        elif method == "FLANN":
            index_params = dict(algorithm=1, trees=5)  # FLANN pour SIFT/ORB
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Méthode inconnue. Utilisez 'BF' ou 'FLANN'.")

    def match_features(self, descriptors1, descriptors2):
        """Associe les descripteurs entre deux images."""
        matches = self.matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)  # Trier par distance
        return matches

class Reconstruction3D:
    def __init__(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix

    def estimate_fundamental_matrix(self, points1, points2):
        """Calcule la matrice fondamentale à partir des correspondances."""
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
        return F, mask

    def triangulate_points(self, proj_matrix1, proj_matrix2, points1, points2):
        """Triangule les points 3D à partir des correspondances."""
        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convertir en coordonnées 3D
        return points_3d.T

class Visualizer:
    def __init__(self):
        pass

    def display_point_cloud(self, points_3d):
        """Affiche le nuage de points 3D."""
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        o3d.visualization.draw_geometries([point_cloud])