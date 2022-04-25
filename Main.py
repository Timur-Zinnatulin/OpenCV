from image_transformer import Image_Transformer
from task_options import Task_Options
import cv2 as cv
import os
import time

imageName = "JeremySuspicious.jpeg"

solver = Image_Transformer(imageName)
functionCall = [solver.find_orb_features, solver.find_sift_features, solver.find_canny_edges, solver.to_grayscale,
                solver.to_hsv, solver.mirror_right, solver.mirror_bottom, solver.rotate_image, solver.rotate_image_around_point,
                solver.move_right, solver.adjust_brightness, solver.adjust_contrast, solver.gamma_conversion, solver.histogram_equalization,
                solver.warmer_image, solver.cooler_image, solver.change_palette, solver.image_binarization, solver.find_contours,
                solver.Sobel_filter, solver.blur_image, solver.filter_high_freq, solver.filter_low_freq, solver.erode, solver.dilate]

print(f"Изображение '{imageName}' загружено и готово к коррекции")
print()
print("Выберите опцию коррекции")

i = int(0)
for option in Task_Options:
    i += 1 
    print(f"{i} - {option}")

print()

while(True):
    choice = int(input())
    if (choice == 0):
        break
    functionCall[choice - 1]()
    solver.image = cv.imread(os.getcwd() + "/" + imageName)
