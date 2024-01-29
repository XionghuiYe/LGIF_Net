import os
import shutil
import glob
import natsort
import pickle
import dlib
import numpy as np
import cv2

def crop_images(dataset_name):
    face_detector = dlib.get_frontal_face_detector()
    import os
    import shutil

    if dataset_name == 'CASME_sq':
        # Save the images into folder 'rawpic_crop'
        for subjectName in glob.glob(os.path.join(dataset_name, 'rawpic', '*')):
            dataset_rawpic = os.path.join(dataset_name, 'rawpic', str(os.path.basename(subjectName)), '*')

            # Create new directory for 'rawpic_crop'
            dir_crop = os.path.join(dataset_name, 'rawpic_crop')
            if not os.path.exists(dir_crop):
                os.mkdir(dir_crop)

            # Create new directory for each subject
            dir_crop_sub = os.path.join(dataset_name, 'rawpic_crop', str(os.path.basename(subjectName)))
            if os.path.exists(dir_crop_sub):
                shutil.rmtree(dir_crop_sub)
            os.mkdir(dir_crop_sub)
            print('Subject', os.path.basename(subjectName))

            for vid in glob.glob(dataset_rawpic):
                dir_crop_sub_vid = os.path.join(dir_crop_sub, os.path.basename(vid))  # Get dir of video
                if os.path.exists(dir_crop_sub_vid):
                    shutil.rmtree(dir_crop_sub_vid)
                os.mkdir(dir_crop_sub_vid)

                for dir_crop_sub_vid_img in natsort.natsorted(glob.glob(os.path.join(vid, 'img*.jpg'))):  # Read images
                    img = os.path.basename(dir_crop_sub_vid_img)
                    count = img[3:-4]  # Get img num Ex 001,002,...,2021
                    # Load the image
                    image = cv2.imread(dir_crop_sub_vid_img)
                    # Run the HOG face detector on the image data
                    detected_faces = face_detector(image, 1)

                    if count == '001':  # Use first frame as reference frame
                        for face_rect in detected_faces:
                            face_top = face_rect.top()
                            face_bottom = face_rect.bottom()
                            face_left = face_rect.left()
                            face_right = face_rect.right()

                    face = image[face_top:face_bottom, face_left:face_right]  # Crop the face region
                    face = cv2.resize(face, (128, 128))  # Resize to 128x128

                    cv2.imwrite(os.path.join(dir_crop_sub_vid, "img{}.jpg".format(count)), face)



    elif(dataset_name == 'SAMMLV'):
        if os.path.exists(dataset_name + '/SAMM_longvideos_crop'): #Delete dir if exist and create new dir
          shutil.rmtree(dataset_name + '/SAMM_longvideos_crop')
        os.mkdir(dataset_name + '/SAMM_longvideos_crop')
    
        for vid in glob.glob(dataset_name + '/SAMM_longvideos/*'):
            count = 0
            dir_crop = dataset_name + '/SAMM_longvideos_crop/' + os.path.basename(vid)
    
            if os.path.exists(dir_crop): #Delete dir if exist and create new dir
              shutil.rmtree(dir_crop)
            os.mkdir(dir_crop)
            print('Video', os.path.basename(vid))
            for dir_crop_img in natsort.natsorted(glob.glob(vid+'/*.jpg')):
                img = os.path.splitext(os.path.basename(dir_crop_img))[0]
                count = img[-4:] #Get img num Ex 0001,0002,...,2021
                # Load the image
                image = cv2.imread(dir_crop_img)
    
                # Run the HOG face detector on the image data
                detected_faces = face_detector(image, 1)
    
                # Loop through each face we found in the image
                if (count == '0001'): #Use first frame as reference frame
                    for i, face_rect in enumerate(detected_faces):
                        face_top = face_rect.top()
                        face_bottom = face_rect.bottom()
                        face_left = face_rect.left()
                        face_right = face_rect.right()
    
                face = image[face_top:face_bottom, face_left:face_right]
                face = cv2.resize(face, (128, 128)) 
    
                cv2.imwrite(dir_crop + "/{}.jpg".format(count), face)
    
    
def load_images(dataset_name):
    images = []
    subjects = []
    subjectsVideos = []

    if dataset_name == 'CASME_sq':
        for i, dir_sub in enumerate(natsort.natsorted(glob.glob(os.path.join(dataset_name, "rawpic_crop", "*")))):
            print('Subject: ' + os.path.basename(dir_sub))
            subjects.append(os.path.basename(dir_sub))
            subjectsVideos.append([])

            for dir_sub_vid in natsort.natsorted(glob.glob(os.path.join(dir_sub, "*"))):
                subjectsVideos[-1].append(os.path.basename(dir_sub_vid).split('_')[1][:4])

                image = []
                for dir_sub_vid_img in natsort.natsorted(glob.glob(os.path.join(dir_sub_vid, "img*.jpg"))):
                    image.append(cv2.imread(dir_sub_vid_img, 0))

                images.append(np.array(image))
        
    elif(dataset_name == 'SAMMLV'):
        for i, dir_vid in enumerate(natsort.natsorted(glob.glob(dataset_name + "/SAMM_longvideos_crop/*"))):
          print('Subject: ' + os.path.basename(dir_vid).split('_')[0])
          subject = os.path.basename(dir_vid).split('_')[0]
          subjectVideo = os.path.basename(dir_vid)
          if (subject not in subjects): #Only append unique subject name
            subjects.append(subject)
            subjectsVideos.append([])
          subjectsVideos[-1].append(subjectVideo)
    
          image = []
          for dir_vid_img in natsort.natsorted(glob.glob(dir_vid + "/*.jpg")):
            image.append(cv2.imread(dir_vid_img, 0))
          image = np.array(image)
          images.append(image)
    
    return images, subjects, subjectsVideos




