import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import os
from PIL import Image
import deepdish as dd
import copy
import face_alignment
from skimage import exposure

def feature_extract():
    # shape predictor
    shape_pred = r'D:\PycharmProjects\LBP-Gaze-Estimation\eye_detector\shape_predictor_68_face_landmarks.dat'
    print("[INFO] loading facial landmark predictor...")

    # dlib facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_pred)

    # facial landmarks for eye detection
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    Dir = [
        'p15_S.h5',
        'p16_S.h5',
        'p17_S.h5',
        'p18_S.h5',
        'p19_S.h5',
        'p20_S.h5',
        'p21_S.h5',
        'p22_S.h5',
        'p23_S.h5',
        'p24_S.h5',
        'p25_S.h5',
        'p26_S.h5',
        'p27_S.h5',
        'p28_S.h5'
    ]
    sessions = []
    type = ['S']
    # p = [2, 16]
    for t in type:
        for P in range(1, 17):
            if P < 12 or P > 13:
                sessions.append(str(P) + "_A_CS_" + t)
        j = 0
        for session in sessions:
            session_str = session
            # os.mkdir(r'D:\PycharmProjects\shallow_network\eyediap_test\\' + session)
            output_path = (r'D:/EYEDIAP/h5_files/')

            # paths to data
            images_path = r'D:\EYEDIAP\Annotations\annotated\data_%s.txt' % session_str
            # gaze_vector_path = r'D:\EYEDIAP\Annotations\annotated\gtv_%s.txt' % session_str
            gaze_angle_path = r'D:\EYEDIAP\Annotations\annotated\gt_cam_%s.txt' % session_str
            headpose_path = r'D:\EYEDIAP\Annotations\annotated\gth_cam_%s.txt' % session_str

            face_features = dd.io.load(r'D:\EYEDIAP\Annotations\annotated\face_feats_%s.h5'%session)
            # face_conv = face_features['face_conv']
            # leye_warp = face_features['leye_warp']
            # face_WARP = face_features['face_warp']
            frameindex = face_features['frameindex']
            # reye_warp = face_features['reye_warp']
            gaze_conv = face_features['face_gaze']

            images = np.loadtxt(images_path, dtype=str)
            headposes = np.loadtxt(headpose_path, dtype=np.float32)
            # gaze_vector = np.loadtxt(gaze_vector_path, dtype=np.float32)
            gaze_angle = np.loadtxt(gaze_angle_path, dtype=np.float32)
            leyelist = []
            reyelist = []
            gazelist = []
            headposelist = []
            # faces =[]
            indexes = []
            # angle_list = []
            roi_size = [80, 80]
            # i = 0
            # while i < len(images):
            for i, img_path in enumerate(images):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)


                if rects == [] or rects == None or rects == 0:
                    print('this image has no landmarks ', img_path)

                for rect in rects:
                    # predict the shapes
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # get points for each landmark point
                    # (xf, yf, wf, hf) = face_utils.rect_to_bb(rect)
                    # face = img[yf:(yf + hf), xf: (xf + wf)]
                    lefteyeX = np.array(shape[lStart:lEnd])
                    righteyeX = np.array(shape[rStart:rEnd])

                    # get the coordinates for each eye
                    (lx, ly, lw, lh) = cv2.boundingRect(lefteyeX)
                    # Leye = img[ly - 37: (ly + lh + 27), lx - 15:lx + lw + 15]
                    # face = img[fy:fy + fh, fx:fx + fw]
                    # face = cv2.resize(face, dsize=(250, 250))

                    Leye = img[ly - 13: ly + lh + 10, lx - 4: lx + lw + 5]
                    # Leye = cv2.resize(Leye, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)

                    rx, ry, rw, rh = cv2.boundingRect(righteyeX)
                    Reye = img[ry - 13:(ry + rh + 10), rx - 5:(rx + rw + 5)]
                    # Reye = cv2.resize(Reye, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
                    # normal_leye = warp_image(img, leye_warp[0], [60, 60])
                    # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                    # normal_leye = exposure.rescale_intensity(normal_leye, out_range=(0, 255))
                    # normal_leye = cv2.resize(normal_leye, dsize=(60, 60))
                    # face_warp = warp_image(img, face_WARP[0], [250, 250])

                    # debugging starts
                    # cv2.imwrite(output_path + 'Leye_%05d' % i + t + '.jpg', Leye)
                    # cv2.imwrite(output_path + 'Reye_%05d' % i + t + '.jpg', Reye)
            #         continue
            #         # debugging ends
            #
            #         print(frameindex[0])
            #         # cv2.waitKey(0)
            #         rx, ry, rw, rh = cv2.boundingRect(righteyeX)
            #         Reye = img[ry - 13:(ry + rh + 10), rx - 5:(rx + rw + 5)]
            #         Reye = cv2.resize(Reye, dsize=(70, 70), interpolation=cv2.INTER_CUBIC)
            #         cv2.imwrite('right_eye.jpg', Reye)
            #         Rwarp = warp_image(img, reye_warp[0], [40, 40])
            #         cv2.imwrite('right_warp.jpg', Rwarp)
            #
                    try:
                        # print('working', images[i])
                        Reye = cv2.resize(Reye, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
                        Leye = cv2.resize(Leye, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
                        # face = cv2.resize(face, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
                        # normal_gaze = normalize_gaze(gaze_vector[i], gaze_conv[i])
                        #
                        Reye = np.reshape(Reye, -1)
                        Leye = np.reshape(Leye, -1)
                        # face = np.reshape(face, -1)
            #
                        leyelist.append(Leye)
                        reyelist.append(Reye)
                        gazelist.append(gaze_angle[i])
                        headposelist.append(headposes[i])
                        # faces.append(face)
                        indexes.append(frameindex[i])
                        # angle_list.append(gaze_angle[i])
                    except:
                        print('image failed to be wrapped', img_path)

                # if i == 10:
                #     break

            Leyes = np.asarray(leyelist)
            Reyes = np.asarray(reyelist)
            gazes = np.asarray(gazelist)
            poses = np.asarray(headposelist)
            # facesArray = np.asarray(faces)
            FrameIndex = np.asarray(indexes)
            # Gaze_angle = np.asarray(angle_list)
            dd.io.save(output_path+'%s'%Dir[j], {'leye': Leyes, 'reye': Reyes, 'gaze': gazes, 'headpose': poses, 'index':FrameIndex})

            leyelist.clear()
            reyelist.clear()
            gazelist.clear()
            headposelist.clear()
            j = j+1
            # faces.clear()



def features_study():
    face_feat_file = r'F:\downloads\EYEDIAP\Annotations\face_features_CS_S.h5'
    face_features = dd.io.load(face_feat_file)
    print('done')

def warp_image(img, warp_mat, roi_size):
    """
    Warp given image according to warp_mat and roi_size.
    :param img: original image
    :param warp_mat: transformation matrix
    :param roi_size: output image size
    :return: warped image
    """
    warp_mat = warp_mat.reshape(3, 3)
    warped_image = cv2.warpPerspective(img, warp_mat, (roi_size[0], roi_size[1]))
    return warped_image

def normalize_gaze(gaze, norm_mat):
    """
    Convert gaze to normalized space.
    :param gaze: 3D unit gaze vector
    :param norm_mat: normalization matrix
    :return: normalized unit gaze vector
    """
    norm_mat = norm_mat.reshape(3, 3)
    y = norm_mat.dot(gaze)
    y = y / np.linalg.norm(y)
    y = vector2angles(y)
    return y

def transform_landmarks(landmarks, conv_mat, mean_face):
    """
    Convert landmarks to normalized space
    :param landmarks: list of 3D landmarks
    :param conv_mat: conversion matrix
    :param mean_face: mean face coordinates
    :return: normalized landmarks and mean face point
    """
    conv_mat = conv_mat.reshape(3, 3)
    landmarks = conv_mat.dot(landmarks.transpose()).transpose()
    mean_face = conv_mat.dot(mean_face)
    return landmarks, mean_face

def preprocess_input_metadata(info: dict):
    """
    Preprocess input metadata (landmarks) by substracting the mean face coordinate.
    :param info: preprocessing information for this specific frame.
    :return: mean centered landmarks
    """
    mean_face = np.mean(info["landmarks"], axis=0)
    return info["landmarks"] - mean_face



def preprocess_input_metadata_norm(landmarks, face_conv):
    """
    If landmarks have to be converted to normalized space, normalize them first and then mean center them
    :param info: preprocessing information for this specific frame.
    :return: normalized, mean centered landmarks
    """
    mean_face = np.mean(landmarks, axis=0)
    landmarks, mean_face = transform_landmarks(landmarks, face_conv, mean_face)
    return landmarks - mean_face

def vector2angles(gaze_vector: np.ndarray):
    """
    Transforms a gaze vector into the angles yaw and elevation/pitch.
    :param gaze_vector: 3D unit gaze vector
    :return: 2D gaze angles
    """
    gaze_angles = np.empty((1, 2), dtype=np.float32)
    gaze_angles[0, 1] = np.arctan(-gaze_vector[0]/-gaze_vector[2])  # phi= arctan2(x/z)
    gaze_angles[0, 0] = np.arcsin(-gaze_vector[1])  # theta= arcsin(y)
    return gaze_angles

def normal_feature_extraction():

    out_path = r'D:/EYEDIAP/h5_files/'

    sessions = []
    type = ['S']
    p = [1, 5, 8, 16]
    for t in type:
        for P in p:
            # if P < 12 or p > 13:
            sessions.append(str(P) + "_A_CS_" + t)

        for session in sessions:
            session_str = session

            face_features = dd.io.load(r'D:\EYEDIAP\Annotations\annotated\face_feats_%s.h5' % session_str)
            leye_warp = face_features['leye_warp']
            gaze_conv = face_features['face_gaze']
            frameindex = face_features['frameindex']
            reye_warp = face_features['reye_warp']

            images_path = r'D:\EYEDIAP\Annotations\annotated\data_%s.txt' % session_str
            # gaze_vector_path = r'D:\EYEDIAP\Annotations\annotated\gtv_%s.txt' % session_str
            gaze_angle_path = r'D:\EYEDIAP\Annotations\annotated\gt_cam_%s.txt' % session_str
            headpose_path = r'D:\EYEDIAP\Annotations\annotated\gth_cam_%s.txt' % session_str

            images = np.loadtxt(images_path, dtype=str)
            # gaze_vector = np.loadtxt(gaze_vector_path, dtype=np.float32)
            gaze_angle = np.loadtxt(gaze_angle_path, dtype=np.float32)
            headposes = np.loadtxt(headpose_path, dtype=np.float32)
            eye_roiSize = [70, 58]

            leyelist = []
            reyelist = []
            gazelist = []
            headposelist = []
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            # i = 0
            # while i < len(images):
            for i, img_path in enumerate(images):
                img = images[i]
                img = cv2.imread(img)

                # get eyes patches
                frame = frameindex[i]  # to debug if the image and warp is correct
                leye_img = warp_image(img, leye_warp[i], eye_roiSize)
                reye_img = warp_image(img, reye_warp[i], eye_roiSize)

                # normal_gaze = normalize_gaze(gaze_vector[i], gaze_conv[i])
                # normal_gaze = gaze_angle[i]

                # reshape
                cv2.imwrite(os.path.join(out_path, ('Leye/' + frame + '.jpg')), leye_img)
                cv2.imwrite(os.path.join(out_path, ('Reye/' + frame + '.jpg')), reye_img)
                print('images no {} printing'.format(frame))

            #     Reye = np.reshape(leye_img, -1)
            #     Leye = np.reshape(reye_img, -1)
            #
            #     # append all data in list
            #     leyelist.append(Leye)
            #     reyelist.append(Reye)
            #     gazelist.append(gaze_angle[i])
            #     headposelist.append(headposes[i])
            #
            #     # # change the illumination
            #     # gamma = 1.25
            #     # gamma_leye = adjust_gamma(leye_img, gamma=gamma)
            #     # gamma_reye = adjust_gamma(reye_img, gamma)
            #     # leyelist.append(gamma_leye.reshape(-1))
            #     # reyelist.append(gamma_reye.reshape(-1))
            #     # gazelist.append(normal_gaze)
            #     # headposelist.append(headposes[i])
            #     #
            #     # # flip the images
            #     # flip_leye = flip_img(leye_img)
            #     # flip_reye = flip_img(reye_img)
            #     # leyelist.append(flip_leye.reshape(-1))
            #     # reyelist.append(flip_reye.reshape(-1))
            #     # gazelist.append(normal_gaze)
            #     # headposelist.append(headposes[i])
            #     #
            #     #
            #     #
            #     #
            #     # # sharpen the images
            #     # sharpen_leye_img = cv2.filter2D(leye_img, -1, kernel=kernel)
            #     # sharpen_reye_img = cv2.filter2D(reye_img, -1, kernel)
            #     #
            #     # leyelist.append(sharpen_leye_img.reshape(-1))
            #     # reyelist.append(sharpen_reye_img.reshape(-1))
            #     # gazelist.append(normal_gaze)
            #     # headposelist.append(headposes[i])
            #     #
            #     # # adding noise to images
            #     # l_noisy1, l_noisy2 = smoothing(leye_img)
            #     # r_noisy1, r_noisy2 = smoothing(reye_img)
            #     #
            #     # leyelist.append(l_noisy1.reshape(-1))
            #     # reyelist.append(r_noisy1.reshape(-1))
            #     # gazelist.append(normal_gaze)
            #     # headposelist.append(headposes[i])
            #     #
            #     # leyelist.append(l_noisy2.reshape(-1))
            #     # reyelist.append(r_noisy2.reshape(-1))
            #     # gazelist.append(normal_gaze)
            #     # headposelist.append(headposes[i])
            #
            #     print('processed {}/ {}'.format(i, len(frameindex)))
            #
            #     # i += 10
            #
            # # convert list to array
            # leye_array = np.array(leyelist)
            # reye_array = np.array(reyelist)
            # gaze_array = np.array(gazelist)
            # headpose_array = np.array(headposelist)
            #
            # dd.io.save(r'D:\PycharmProjects\shallow_network\eyediap_test\eyeDiap_%s.h5' % session_str,
            #            {'leye': leye_array, 'reye': reye_array, 'gaze': gaze_array, 'headpose': headpose_array})

        # load the features for normalization


def smoothing(img):
    # smooth_img = cv2.GaussianBlur(img, (7, 7), 0)
    gaussian_noise = img.copy()
    cv2.randn(gaussian_noise, 0, 150)
    noisy1 = (img + gaussian_noise)

    uniform_noise = img.copy()
    cv2.randu(uniform_noise, 0, 1)
    noisy = img+gaussian_noise
    return noisy, noisy1
def flip_img(img):
    vert_flip = img.copy()
    vert_flip = cv2.flip(img, 1)
    return vert_flip

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

if __name__ == '__main__':
    feature_extract()
    # normal_feature_extraction()
    #
    # sessions = []
    # type = ['S']
    # for t in type:
    #     for P in range(1, 17):
    #         if P < 12 or P > 13:
    #             sessions.append(str(P) + "_A_CS_" + t)
    #
    #     for session in sessions:
    #         data = dd.io.load(r'D:\PycharmProjects\shallow_network\eyediap_test\eyeDiap_%s'%session + '.h5')
    #         leye = data['leye']
    #         reye = data['reye']
    #         os.mkdir(r'D:\PycharmProjects\shallow_network\eyediap_test\\'+session)
    #         output_path = (r'D:\PycharmProjects\shallow_network\eyediap_test\%s\\'%session)
    #         for i in range(0, len(leye)):
    #             imLeye = leye[i].reshape(60, 60, 3)
    #             imReye = reye[i].reshape(60, 60, 3)
    #             cv2.imwrite(output_path + 'Leye_%05d' % i + t + '.jpg', imLeye)
    #             cv2.imwrite(output_path + 'Reye_%05d' % i + t + '.jpg', imReye)
    #         print('done....')
    #
    # print(data)