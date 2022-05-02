import os
from flask_pymongo import pymongo
import io
import numpy as np
import cv2

MONGO_URL = os.getenv('MONGO_URL', None)

if MONGO_URL == None:
    print('Could not connect to database.')

client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('eyedb')
eye_collection = pymongo.collection.Collection(db, 'eyes')

img_dtype = np.uint8
kp_dtype = '<U108'
des_dtype = np.float32

def convert_bytes(arr):
    return arr.tobytes()

def convert_img_np(bytes):
    arr = np.frombuffer(bytes, dtype=img_dtype)
    return arr.reshape((512, 512))

def convert_des_np(bytes, shape):
    arr = np.frombuffer(bytes, dtype=des_dtype)
    return arr.reshape(shape)

def kp_to_np(kp):
    return ["{}|{}|{}|{}|{}".format(k.pt, k.size, k.angle, k.response, k.octave) for k in kp]
    
def np_to_kp(arr):
    kp_list = []
    for k in arr:
        a = k.split('|')
        pt = a[0].split(', ')
        x = float(pt[0][1:])
        y = float(pt[1][:-1])
        kp_list.append(cv2.KeyPoint(x, y, float(a[1]), float(a[2]), float(a[3]), int(a[4]), -1))
    return kp_list

def insert_eye(name, img, kp, des):
    eye_collection.insert_one({
        "name": name,
        "img": convert_bytes(img),
        "kp": kp_to_np(kp),
        "des": convert_bytes(des),
        "des-shape": des.shape,
    })