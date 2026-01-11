import numpy as np
import cv2
import os
import csv
from create_feature import *
# ملاحظة: إذا لم تجد ملف calorie_calc.py، قم بإنشاء ملف فارغ بهذا الاسم أو تجاهله
# from calorie_calc import * # تعريف SVM ليتوافق مع OpenCV 4+ في Colab
def get_svm():
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    return svm

def training():
    feature_mat = []
    response = []
    # المسار المعدل (مباشرة في المجلد الحالي)
    base_path = "./All_Images/" 
    
    for j in range(1, 15):
        for i in range(1, 21):
            img_path = f"{base_path}{j}_{i}.jpg"
            if os.path.exists(img_path):
                print(f"جاري معالجة: {img_path}")
                fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
                feature_mat.append(fea)
                response.append(float(j))

    if len(feature_mat) == 0:
        print("❌ لم يتم العثور على صور في مجلد All_Images!")
        return

    trainData = np.float32(feature_mat).reshape(-1, 94)
    responses = np.int32(response).reshape(-1, 1)

    svm = get_svm()
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    print("✅ تم التدريب وحفظ النموذج بنجاح!")

if __name__ == 'main':
    training()
