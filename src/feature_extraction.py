
# import cv2
# import numpy as np
# import math


# def distance(p1, p2):
#     """Calculate Euclidean distance between two points."""
#     return math.hypot(
#         p1[0] - p2[0],
#         p1[1] - p2[1]
#     )


# def get_palm_center(hand_mask):
#     """
#     Estimate the palm centre and palm radius
#     using the distance transform.
#     """

#     dist = cv2.distanceTransform(
#         hand_mask,
#         cv2.DIST_L2,
#         5
#     )

#     _, max_val, _, max_loc = cv2.minMaxLoc(dist)

#     return max_loc, max_val


# def extract_fingertips(contour, palm_center, palm_radius):
#     """
#     Find fingertip candidates using convex hull geometry.

#     This is an initial geometric approach.
#     It does not guarantee that every candidate is a real fingertip.
#     """

#     # Step 1: Calculate convex hull
#     hull = cv2.convexHull(contour)

#     # Extract hull points as a simple list of (x, y) coordinates
#     hull_points = hull[:, 0, :]

#     candidates = []

#     # Step 2: Find hull points sufficiently far from palm centre
#     min_distance = 1.25 * palm_radius

#     for point in hull_points:

#         x, y = int(point[0]), int(point[1])

#         d = distance(
#             (x, y),
#             palm_center
#         )

#         if d > min_distance:
#             candidates.append((x, y, d))

#     # Step 3: Sort candidates by distance from palm centre
#     candidates.sort(
#         key=lambda p: p[2],
#         reverse=True
#     )

#     # Step 4: Remove nearby points that probably belong
#     # to the same fingertip
#     fingertips = []

#     min_separation = 0.55 * palm_radius

#     for candidate in candidates:

#         x, y, d = candidate

#         too_close = False

#         for existing in fingertips:

#             if distance((x, y), existing) < min_separation:
#                 too_close = True
#                 break

#         if not too_close:
#             fingertips.append((x, y))

#         # A hand has at most five fingertips
#         if len(fingertips) == 5:
#             break

#     return hull, fingertips


# def extract_hand_features(contour, hand_mask):
#     """
#     Main function:
#     Takes a hand contour and its binary mask.

#     Returns:
#         palm_center
#         palm_radius
#         hull
#         fingertips
#     """

#     # Ensure the contour has enough points
#     if contour is None or len(contour) < 4:
#         return None, None, None, []

#     # Step 1: Calculate palm centre and radius
#     palm_center, palm_radius = get_palm_center(hand_mask)

#     if palm_radius <= 0:
#         return None, None, None, []

#     # Step 2: Extract convex hull and fingertip candidates
#     hull, fingertips = extract_fingertips(
#         contour,
#         palm_center,
#         palm_radius
#     )

#     return palm_center, palm_radius, hull, fingertips





import math
import cv2
import numpy as np


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def get_palm_center(hand_mask):
    """
    Estimate the palm centre and palm radius
    using the distance transform.
    """
    dist = cv2.distanceTransform(
        hand_mask,
        cv2.DIST_L2,
        5
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc, max_val


def extract_fingertips(contour, palm_center, palm_radius):
    """
    Find fingertip candidates using convex hull geometry.
    Rejects points below the palm center (like the wrist) 
    assuming an upright hand orientation.
    """
    # Step 1: Calculate convex hull
    hull = cv2.convexHull(contour)
    
    # Extract hull points as a simple list of (x, y) coordinates
    hull_points = hull[:, 0, :]
    candidates = []

    # Step 2: Find hull points far from palm centre and pointing upwards
    min_distance = 1.25 * palm_radius
    
    for point in hull_points:
        x, y = int(point[0]), int(point[1])

        # Upright hand check: fingertips are above or near the level of the palm center 
        # (allowing a slight threshold below for the thumb/side fingers).
        if y < palm_center[1] + (0.3 * palm_radius):
            d = distance((x, y), palm_center)
            if d > min_distance:
                candidates.append((x, y, d))

    # Step 3: Sort candidates by distance from palm centre
    candidates.sort(
        key=lambda p: p[2],
        reverse=True
    )

    # Step 4: Remove nearby points that belong to the same fingertip
    fingertips = []
    min_separation = 0.55 * palm_radius

    for candidate in candidates:
        x, y, d = candidate
        too_close = False

        for existing in fingertips:
            if distance((x, y), existing) < min_separation:
                too_close = True
                break

        if not too_close:
            fingertips.append((x, y))

        # A hand has at most five fingertips
        if len(fingertips) == 5:
            break

    return hull, fingertips


def extract_hand_features(contour, hand_mask):
    """
    Main function:
    Takes a hand contour and its binary mask.

    Returns:
        palm_center
        palm_radius
        hull
        fingertips
    """
    if contour is None or len(contour) < 4:
        return None, None, None, []

    palm_center, palm_radius = get_palm_center(hand_mask)

    if palm_radius <= 0:
        return None, None, None, []

    hull, fingertips = extract_fingertips(
        contour,
        palm_center,
        palm_radius
    )

    return palm_center, palm_radius, hull, fingertips