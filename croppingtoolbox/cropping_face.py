import cv2
import numpy as np




class cropping_face():
    def __init__(self,weightpath='',cfgpath=''):
        
        self.inputimage=[]
        self.net=None
        self.model=None
        self.classes=[]
        self.scores=[]
        self.boxes=[]
        
        self.weightpath=weightpath
        self.cfgpath=cfgpath
        
    def __getitem__(self, item):
        return object.__getattribute__(self, item)    
    
    def loadmodel(self):
        
        self.net = cv2.dnn.readNet(self.weightpath, self.cfgpath)
        print('model: Load model successful !')

    def setinput(self,inputimage):
        
        self.inputimage=inputimage.copy()
        print('model: cropping_face setinput successfull')
        
    def setpara(self,CONFIDENCE_THRESHOLD,NMS_THRESHOLD):
        
        self.CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD
        self.NMS_THRESHOLD=NMS_THRESHOLD
        print('model: cropping_face setpara successfull')

        
    def inference(self):
        
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.classes, self.scores, self.boxes = self.model.detect(self.inputimage, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        
        print('model: cropping_face inference successfull')
        
        if len(self.classes)==0:
            print ("model: There is no object detected !")
            return False
        else:
            print ("model: There has object detected !")
            return True