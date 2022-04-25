import cv2 as cv
import numpy as np
import imutils
import os

from scipy.interpolate import UnivariateSpline

class Image_Transformer:
    def __init__(self, image: str):
        self.image = cv.imread(os.getcwd() + "/" + image)

    @staticmethod
    def show_image(window, image):
        cv.imshow(window, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


    def find_orb_features(self):
        orb = cv.ORB_create()
        keypoints = orb.detect(self.image, None)
        keypoints, _ = orb.compute(self.image, keypoints)

        orbImage = cv.drawKeypoints(self.image, keypoints, None, color = (0, 255, 0), flags = 0)
        self.show_image('Find orb', orbImage)

    def find_sift_features(self):
        sift = cv.SIFT_create()
        keypoints = sift.detect(self.image, None)

        siftImage = cv.drawKeypoints(
            self.image,
            keypoints,
            self.image,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        self.show_image('Find sift', siftImage)

    def find_canny_edges(self):
        edges = cv.Canny(self.image, 25, 255, L2gradient=False)
        self.show_image('Canny edges', edges)

    def to_grayscale(self):
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.show_image('Gray image', gray)

    def to_hsv(self):
        hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        self.show_image('HSV image', hsv)

    def mirror_right(self):
        mirrorImage = np.fliplr(self.image)
        self.show_image('Horizontal mirror', mirrorImage)

    def mirror_bottom(self):
        mirrorImage = cv.flip(self.image, 0)
        self.show_image('Vertical mirror', mirrorImage)

    def rotate_image(self):
        rotatedImage = imutils.rotate(self.image, angle = 45)
        self.show_image('Rotated image', rotatedImage)

    def rotate_image_around_point(self):
        (h, w) = self.image.shape[:2]
        M = cv.getRotationMatrix2D((h, w), 30, 1.0)

        rotatedImage = cv.warpAffine(self.image, M, (w, h))
        self.show_image('Rotated image', rotatedImage)

    def move_right(self):
        h, w = self.image.shape[:2]
        translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])

        movedImage = cv.warpAffine(self.image, translation_matrix, (w, h))
        self.show_image('Moved_image', movedImage)

    def adjust_brightness(self):
        hsvImage = cv.add(self.image, np.array([50.0]))
        self.show_image('Brighter image', hsvImage)

    def adjust_contrast(self):
        alpha = 1.5
        adjusted = cv.convertScaleAbs(self.image, alpha=alpha)
        self.show_image('Adjusted image', adjusted)

    def gamma_conversion(self):
        gamma = 0.5
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        adjusted = cv.LUT(self.image, table)
        self.show_image('Gamma converted image', adjusted)


    def histogram_equalization(self):
        grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        equalized = cv.equalizeHist(grayscale)
        self.show_image('Histogram equalized', equalized)

    @staticmethod
    def color_shift(image, isCooler):
        increaseLookupTable = UnivariateSpline([0, 64, 128, 256], [0, 80, 160, 256])(range(256))
        decreaseLookupTable = UnivariateSpline([0, 64, 128, 256], [0, 50, 100, 256])(range(256))
        red, green, blue = cv.split(image)
        if isCooler:
            red = cv.LUT(red, increaseLookupTable).astype(np.uint8)
            blue = cv.LUT(blue, decreaseLookupTable).astype(np.uint8)
        else:
            blue = cv.LUT(red, increaseLookupTable).astype(np.uint8)
            red = cv.LUT(blue, decreaseLookupTable).astype(np.uint8)
        return cv.merge((red, green, blue))

    def warmer_image(self):
        warmerImage = self.color_shift(self.image, False)
        self.show_image('Warmer image', warmerImage)

    def cooler_image(self):
        coolerImage = self.color_shift(self.image, True)
        self.show_image('Cooler image', coolerImage)

    def change_palette(self):
        newPalette = cv.applyColorMap(self.image, cv.COLORMAP_BONE)
        self.show_image('New palette', newPalette)

    @staticmethod
    def binarize(image):
        median = np.median(image)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))

        return cv.Canny(image, lower, upper)
    
    def image_binarization(self):
        binarized = self.binarize(self.image)
        self.show_image('Binarized image', binarized)

    def find_contours(self):
        binarized = self.binarize(self.image)
        contours, hierarchy = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        imageContours = np.zeros(self.image.shape)

        cv.drawContours(imageContours, contours, -1, (0,255,0), 3)
        self.show_image('Image contours', imageContours)

    def Sobel_filter(self):
        sobel = cv.Sobel(self.image, cv.CV_64F, 0, 1, ksize=3)
        self.show_image('Sobel filter', sobel)

    def blur_image(self):
        blurredImage = cv.GaussianBlur(self.image, ksize=(9, 9), sigmaX=0, sigmaY=0)
        self.show_image('Blurred image', blurredImage)

    def filter_high_freq(self):
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)

        rows, cols = self.image.shape[:2]
        crow,ccol = rows//2 , cols//2
        fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)

        self.show_image('High frequency filter', img_back)

    def filter_low_freq(self):
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)

        magnitudeSpectrum = 20 * np.log(np.abs(fshift))
        self.show_image('Low frequency filter', magnitudeSpectrum)

    def erode(self):
        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(self.image, kernel, iterations = 1)
        self.show_image('Eroded image', erosion)

    def dilate(self):
        kernel = np.ones((5, 5), 'uint8')
        dilation = cv.dilate(self.image, kernel, iterations=1)
        self.show_image('Dilated image', dilation)