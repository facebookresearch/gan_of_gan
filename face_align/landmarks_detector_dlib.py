"""
Adapted from https://github.com/jrosebr1/imutils
"""

import numpy as np


def shape_to_np_dlib(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


class LandmarksDetector_dlib:
    def __init__(self, detector, predictor):
        """init.

        Parameters
        ----------
        num_landmarks : init
            Number of landmarks to detect, could be 68 or 100
        """
        # define face and landmark detectors
        self.detector = detector
        self.predictor = predictor

    def get_landmarks(self, img):
        """
        apply dlib to get face bbox and landmarks
        :param predictor:
        :param detector:
        :param img:
        :return: bbox and landmarks (largest face)
        """

        dets = self.detector(img, 1)
        # print('detected face {}'.format(len(dets)))
        if len(dets) == 0:  # no face
            return None, None

        face_len = 0
        landmarks = None

        for _, d in enumerate(dets):
            cur_bbox = [d.left(), d.top(), d.right(), d.bottom()]
            # Get the landmarks/parts for the face in box d.
            if face_len < (cur_bbox[3] - cur_bbox[1]):
                shape = self.predictor(img, d)
                landmarks = shape_to_np_dlib(shape)
                face_len = cur_bbox[3] - cur_bbox[1]

        return landmarks

