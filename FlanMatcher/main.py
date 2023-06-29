import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img(img, name):
    #print("Show image- running")
    figure = plt.figure(figsize=(8, 8))
    a_x = figure.gca()
    a_x.get_xaxis().set_visible(False)
    a_x.get_yaxis().set_visible(False)
    a_x.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.show()

def find_key_points(img1, img2):
    #Zamiana na czarno białe
    grey_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Obliczenie punktów kluczowych oraz deskryptorów
    orb = cv2.ORB_create()
    k_p_1, des_1 = orb.detectAndCompute(grey_1, None)
    k_p_2, des_2 = orb.detectAndCompute(grey_2, None)

    return des_1, des_2, k_p_1, k_p_2

def print_key_points(base_img, kp_1):
    print('print key points- running')
    show1 = cv2.drawKeypoints(base_img,
                              kp_1,
                              base_img,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img(show1, 'Punkty kluczowe')


def flann_matcher(des1, des2, kp1, kp2, show_stages, base_img, img2):

    if show_stages:
        print('Flann matcher- running')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img4 = cv2.drawMatches(base_img, kp1, img2, kp2, matches[:10], None, flags=2)


    good_matches = matches[:11]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width = base_img.shape[:2]
    pts = np.float32([[0, 0],
                      [0, height - 1],
                      [width - 1, height - 1],
                      [width - 1, 0]]
                     ).reshape(-1, 1, 2)

    #Obrysowanie
    dst = cv2.perspectiveTransform(pts, m) + (width,0)
    img4 = cv2.polylines(img4, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    img4 = cv2.resize(img4, (1920, 1080), interpolation=cv2.INTER_AREA)

    if show_stages:
        show_img(img4,'Flann matcher + obrysowanie')
    else:
        cv2.imshow('Flann matcher', img4)



def find_obj_on_video_flann_matcher(base_img):

    print('Find object on video flann- running')
    video = cv2.VideoCapture('sawmovie.mp4')
    captured, frame = video.read()
    while captured:

        des_1, des_2, kp_1, kp_2 = find_key_points(base_img, frame)
        base_img = cv2.resize(base_img,(1920,1080),interpolation= cv2.INTER_AREA)
        flann_matcher(des_1,des_2, kp_1,kp_2,False,base_img,frame)

        cv2.waitKey(20)
        captured, frame = video.read()



base_img = cv2.imread('saw1.jpg')
img2 = cv2.imread('saw3.jpg')

#Określenie punktór kluczowych
des_1, des_2, kp_1, kp_2 =find_key_points(base_img, img2)
print_key_points(base_img,kp_1)

#Prezentacja matchera Flann
flann_matcher(des_1, des_2, kp_1, kp_2, True, base_img, img2)

#Wyszukanie piły na filmie przy użyciu matchera Flann
find_obj_on_video_flann_matcher(base_img)
