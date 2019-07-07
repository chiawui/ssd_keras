from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
import numpy as np

anchor_box_size=1/3
encoder=SSDInputEncoder(img_height=96,img_width=128, n_classes=4, predictor_sizes=[3,4], two_boxes_for_ar1=False, 
                        aspect_ratios_global=[1.0], neg_iou_limit=0.1, min_scale=anchor_box_size, max_scale=anchor_box_size)

ground_truth_label=np.zeros([1,1,5]) #batch_size, #boxes, (class,xmin,ymin,xmax,ymax)

ground_truth_label[0,0,0]=1  #class
ground_truth_label[0,0,1]=0 #xmin
ground_truth_label[0,0,2]=0 #ymin
ground_truth_label[0,0,3]=24 #xmax
ground_truth_label[0,0,4]=24 #ymax
#print(ground_truth_label)   

y_encoded=encoder(ground_truth_label) #batch_size, #boxes, (class, 4 box coord, 4 dummy, 4 variances)
print(y_encoded.shape)
#print(y_encoded)