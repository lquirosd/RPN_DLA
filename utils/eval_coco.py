import sys                                                                                                            
from pycocotools.coco import COCO                                                                                     
from pycocotools.cocoeval import COCOeval                                                                             
import numpy as np                                                                                                    
                                                                                                                      
                                                                                                                      
annType = ['segm','bbox','keypoints']                                                                                 
annType = annType[int(sys.argv[1])]                                                                                   
annFile = sys.argv[2]                                                                                                 
resFile = sys.argv[3]                                                                                                 
dets = map(int, sys.argv[4].strip('[]').split(','))
only_reg = int(sys.argv[5])


prefix = 'person_keypoints' if annType=='keypoints' else 'instances'                                                  
                                                                                                                      
cocoGt=COCO(annFile)                                                                                                  
                                                                                                                      
cocoDt=cocoGt.loadRes(resFile)                                                                                        
imgIds=sorted(cocoGt.getImgIds())                                                                                     
cocoEval = COCOeval(cocoGt,cocoDt,annType)                                                                            
cocoEval.params.imgIds  = imgIds                                                                                      
cocoEval.params.maxDets  = dets #---use dets[2] as large es the max num of samples
if only_reg:
    lid = cocoGt.getCatIds(catNms=['textline'])
    fids = cocoGt.getCatIds()
    fids.remove(lid[0])
    cocoEval.params.catIds = fids
cocoEval.evaluate()                                                                                                   
cocoEval.accumulate()                                                                                                 
cocoEval.summarize()
#cocoEval.summarize._summarize(1, maxDets=1000)
#exec(cocoEval.summarize.__code__.co_consts[1](1, maxDets=1000))
