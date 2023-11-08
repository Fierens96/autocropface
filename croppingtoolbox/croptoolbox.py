import os
import cv2
import numpy as np
import scipy.io
import croppingtoolbox.cropping_face as cropface


class autocrop():
    def __init__(self):
        
        #RAWIMAGE
        self.rawimg=[]
        self.labelimg=[]
        
        #PARA
        self.imagename=''
        self.cropmode=''
        self.class_names=[]
        self.scaleimagenum=9 #預設切割張數
        self.scale=0 #擴張比例
        self.detection = []
        self.center_x=[]
        self.center_y=[]
        
        #OBJECTPARA
        self.isobject=False

        #DRAWING
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        #FRAME
        self.centerframe_w=[]
        self.centerframe_h=[]
        self.rawimg_w=[]
        self.rawimg_h=[]
        
        #MODELOUTPUT
        self.classes=[]
        self.scores=[]
        self.boxes=[]

        #MUTIDATA
        self.MUTICROP=[]
        print('autocrop init success !')
        
    def __getitem__(self, item):
        return object.__getattribute__(self, item)
    

    def presetting(self,rawimg,imagename,cropmode='face'):
        
        self.rawimg=rawimg
        self.imagename=imagename
        self.labelimg=self.rawimg.copy()
        self.cropmode=cropmode
        print('autocrop: presetting success !')

        
    def setmodel(self,weightpath,cfgpath):
        
        self.weightpath=weightpath
        self.cfgpath=cfgpath
        print('autocrop: set model path success !')
        
    def setmodelpara(self,CONFIDENCE_THRESHOLD,NMS_THRESHOLD):
        
        self.CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD
        self.NMS_THRESHOLD=NMS_THRESHOLD
        print('autocrop: set model para success !')
        
    def loadmodel(self):

        if self.cropmode=='face':

            self.class_names = ['face']
            
            # call cropface
            self.cropface=cropface.cropping_face(self.weightpath,self.cfgpath)
            self.cropface.loadmodel()
            self.cropface.setinput(self.rawimg)
            self.cropface.setpara(self.CONFIDENCE_THRESHOLD,self.NMS_THRESHOLD)
            print('autocrop: load model success !')

    def runmodel(self):
        
        if self.cropmode=='face':
            self.isobject=self.cropface.inference()
            self.classes=self.cropface['classes']
            self.scores=self.cropface['scores']
            self.boxes=self.cropface['boxes']
            print('autocrop: run model success !')
            

    def makelabelimg(self):
        
        if self.isobject:
            for (classid, score, box) in zip(self.classes, self.scores, self.boxes):
                color = self.COLORS[int(classid) % len(self.COLORS)]
                label = "%s : %f" % (self.class_names[classid], score)
                cv2.rectangle(self.labelimg, box, color, 2)
                cv2.putText(self.labelimg, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(self.labelimg, 'Object', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        else:
            print('No object detect so can not makelabelimg !')        
    
        
    def setcenteroffset(self,top_offset=0, bottom_offset=0, left_offset=0, right_offset=0):
        
        self.top_offset=top_offset
        self.bottom_offset=bottom_offset
        self.left_offset=left_offset
        self.right_offset=right_offset
        print('autocrop: setcenteroffset success !')
        
    def getcenter(self):
        
        self.center_x=[]
        self.center_y=[]
        
        # Calculating center
        self.rawimg_h,self.rawimg_w=self.rawimg.shape[0],self.rawimg.shape[1]

        # Suppose there are n boxes
        n = len(self.boxes)

        # Calculate the center point coordinates of each box
        center_points = []
        for i in range(n):
            # Calculate the coordinates of the upper left corner and lower right corner of the box
            x1, y1, w, h = self.boxes[i]
            x2, y2 = x1 + w, y1 + h
            # Calculate center point coordinates
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_points.append((center_x, center_y))

        # Calculate the average coordinates of the center points of all boxes
        avg_center_x = 0
        avg_center_y = 0
        for i in range(n):
            avg_center_x += center_points[i][0]
            avg_center_y += center_points[i][1]

        avg_center_x = int(avg_center_x / n)
        avg_center_y = int(avg_center_y / n)

        self.center_x,self.center_y = self.get_offset_center([avg_center_x, avg_center_y], self.rawimg_h, self.rawimg_w, self.top_offset, self.bottom_offset, self.left_offset, self.right_offset)

        # Calculate the length and width of the original box
        # Find the largest and smallest x and y values
        min_x = min(box[0] for box in self.boxes)
        min_y = min(box[1] for box in self.boxes)
        max_x = max(box[0] + box[2] for box in self.boxes)
        max_y = max(box[1] + box[3] for box in self.boxes)

        # Calculate the width and height of the box
        self.centerframe_w = max_x - min_x
        self.centerframe_h = max_y - min_y

        # Take the smallest x and y values ​​as the coordinates of the upper left corner of the box
        x = min_x
        y = min_y


        # Set the proportion to be expanded
        rx=self.rawimg.shape[1]
        ry=self.rawimg.shape[0]
        raw_box_scale=int((rx+ry)/(self.centerframe_w+self.centerframe_h))
        self.scale = round(raw_box_scale/self.scaleimagenum,4)

    def getdetection(self):
        
        self.detection=[]
        # Continue to expand until it does not exceed the scope of the original image
        for i in range(10):
            
            # Calculate the new length and width after expansion
            new_w = int(self.centerframe_w * (1 + (i) * self.scale))
            new_h = int(self.centerframe_h * (1 + (i) * self.scale))

            # Set the new length and width to the same values ​​to ensure a square shape
            new_size = max(new_w, new_h)
            new_w = new_size
            new_h = new_size

            # Calculate the coordinates of the upper left corner and the lower right corner after expansion
            new_x1 = max(0, int(self.center_x - (new_w / 2)))
            new_y1 = max(0, int(self.center_y - (new_h / 2)))
            new_x2=new_x1+new_size
            new_y2=new_y1+new_size
        #     new_x2 = min(self.rawimg_w, int(center_x + (new_w / 2)))
        #     new_y2 = min(self.rawimg_h, int(center_y + (new_h / 2)))

            if new_x2>self.rawimg_w or new_y2>self.rawimg_h:
                diffx=self.rawimg_w-new_x1
                diffy=self.rawimg_h-new_y1

                decision=min(diffx,diffy)
                new_x2=new_x1+decision
                new_y2=new_y1+decision          

            # update detection results
            self.detection.append([new_x1, new_y1, new_x2, new_y2])

#         print(self.detection)
        # Remove duplicate rows and convert each row into a tuple
        self.detection = list(set(tuple(row) for row in self.detection))

        # sorted
        self.detection = sorted(self.detection, key=lambda x: -x[0])

    def startcropping(self):
        
        if self.isobject:
            # get center and detection
            self.getcenter()
            self.getdetection()

#             print(len(self.detection))

            # len() maybe bug if detection list y less then 4
            self.MUTICROP=np.empty((len(self.detection)),dtype=object)

            for i, det in enumerate(self.detection):
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                crop = self.rawimg[y1:y2, x1:x2,:]
                self.MUTICROP[i]=crop.copy()
        else:
            print('No object detect so can not start cropping !')
            
    def saveresult(self,folder,image_quality=80):
        
        if self.isobject:

            cropsavepath=os.path.join(folder,'cropimgs_'+self.imagename)

            if not os.path.exists(cropsavepath):
                os.mkdir(cropsavepath)


            for i in range(len(self.MUTICROP)):
                savepath=os.path.join(cropsavepath,str(i+1)+'.png')
                cv2.imwrite(savepath, self.MUTICROP[i].astype(np.uint8), [
                        cv2.IMWRITE_JPEG_QUALITY, image_quality])
            print('Object detect save successed !')        
       
        else:
            print('No object detect so can not saveresult !')        
            
#TOOL#######################################################################################################################################
    def get_offset_center(self,center, img_height, img_width, top_offset=0, bottom_offset=0, left_offset=0, right_offset=0):
        """
        Get the center coordinates after applying offset in specified directions.

        Args:
        - center (tuple): Tuple of (x, y) representing the current center coordinates.
        - img_height (int): Height of the image.
        - top_offset (float): The percentage of offset to apply in the upward direction. Default is 0.
        - bottom_offset (float): The percentage of offset to apply in the downward direction. Default is 0.
        - left_offset (float): The percentage of offset to apply in the left direction. Default is 0.
        - right_offset (float): The percentage of offset to apply in the right direction. Default is 0.

        Returns:
        - Tuple of (x, y) representing the center coordinates after applying offset in specified directions.
        """
        # Calculate the amount of offset to apply in each direction
        top_offset_px = int(top_offset * center[1])
        bottom_offset_px = int(bottom_offset * abs(img_height-center[1]))
        left_offset_px = int(left_offset * center[0])
        right_offset_px = int(right_offset * abs(img_width-center[0]))

        # Apply the offset to the current center coordinates
        new_center_x = center[0] - left_offset_px + right_offset_px
        new_center_y = center[1] - top_offset_px + bottom_offset_px

        # Ensure the new center coordinates are within the bounds of the image
        new_center_x = max(min(new_center_x, img_width), 0)
        new_center_y = max(min(new_center_y, img_height), 0)

        return new_center_x, new_center_y






