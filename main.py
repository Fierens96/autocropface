import os
import argparse
import cv2
import numpy as np
from croppingtoolbox import croptoolbox
from tkinter import filedialog
from tkinter import *



def main(cfgpath, weightpath,cropmode='face',CONFIDENCE_THRESHOLD=0,NMS_THRESHOLD=0.2,offset=[0,0,0,0]):
    
    # set model info 
    # your model and data path
    cfgpath=cfgpath
    weightpath=weightpath    
    
    #Open pop-up window
    desktop_path = os.path.join(os.environ["HOMEPATH"], "Desktop")
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir=desktop_path, title="Select a File", filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*")))
    root.quit()
    root.destroy()
    savepath=os.path.dirname(root.filename)
    imagename=os.path.basename(root.filename)
    print(root.filename)


    # Load raw image
    print('Main: Loading image')
    savepath=os.path.dirname(root.filename)
    rawimage = cv2.imdecode(np.fromfile(root.filename,dtype=np.uint8),-1)
    if len(rawimage.shape) == 3 and rawimage.shape[2] == 4:
        # Image has alpha channel
        rawimage = cv2.cvtColor(rawimage, cv2.COLOR_BGRA2BGR)
    else:
        # Image does not have alpha channel
        rawimage = rawimage[:, :, :3]
    print('raw image shape :'+str(rawimage.shape))

    # Start inference
    print('Main: start inference')
    autocrop=croptoolbox.autocrop()
    autocrop.presetting(rawimage,imagename,cropmode)
    autocrop.setmodel(weightpath,cfgpath)
    autocrop.setmodelpara(CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
    autocrop.loadmodel()
    autocrop.runmodel()
    autocrop.makelabelimg()


    # set center offset can recrop the image 0-1
    try:
        autocrop.setcenteroffset(top_offset=offset[0], bottom_offset=offset[1], left_offset=offset[2], right_offset=offset[3])
    except Exception as e:
        print(f"Error: {e}")
        print("Input Error: offset")
        
    autocrop.startcropping()
    autocrop.saveresult(savepath)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get face from image')
    
    # add args
    parser.add_argument('--cfgpath', type=str, help='path of cfg file')
    parser.add_argument('--weightpath', type=str, help='path of weights')
    parser.add_argument('--cropmode', type=str, help='cropmode')
    parser.add_argument('--CONFIDENCE_THRESHOLD', type=float, help='CONFIDENCE_THRESHOLD')
    parser.add_argument('--NMS_THRESHOLD', type=float, help='NMS_THRESHOLD')
    parser.add_argument('--offset', nargs='+', type=float, help='offset as a list, including top,bottom,left,right')
    
    args = parser.parse_args()
    
    # pass args
    main(args.cfgpath, args.weightpath,args.cropmode,args.CONFIDENCE_THRESHOLD,args.NMS_THRESHOLD,args.offset)
