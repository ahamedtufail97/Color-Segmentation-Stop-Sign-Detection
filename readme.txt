ECE 276A PR1
Siddarth Meenakshi Sundaram
A53299801

Code organisation:

mask_img_create.py :  Used to create the mask.
logistic_regression.py : Code for supervised learning.
stop_sign_detector.py : Code for detecting the stop sign
final_weight.npy : Contains the final weights of the trained model.

To replicate this work:

Use mask_img_create.py to create masks. Then use the logistic_regression.py to train the model. Then to detect the stop-sign, run the stop_sign_detector.py with the final_weight.npy obtained from running logistic_regression.py