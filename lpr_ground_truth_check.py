from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
import numpy as np

encoder=SSDInputEncoder(224,224, 4, [2,2], two_boxes_for_ar1=False, aspect_ratios_global=[1.0], neg_iou_limit=0.1, min_scale=0.5, max_scale=0.5)

ground_truth_label=np.zeros([1,1,5]) #batch_size, #boxes, (class,xmin,ymin,xmax,ymax)

ground_truth_label[0,0,0]=1  #class
ground_truth_label[0,0,1]=50 #xmin
ground_truth_label[0,0,2]=50 #ymin
ground_truth_label[0,0,3]=224 #xmax
ground_truth_label[0,0,4]=224 #ymax
#print(ground_truth_label)   

y_encoded=encoder(ground_truth_label) #batch_size, #boxes, (class, 4 box coord, 4 dummy, 4 variances)
#print(y_encoded.shape)
print(y_encoded[0,0])