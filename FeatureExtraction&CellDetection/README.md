
Detecting the location and draw boundary of nuclei from tissue microscopic images (H&E stained).
Model is based on U-net [1] with contour enhancement in loss function. Overlap patch based strategy is used to 1) adapt to variant input image size (resize image may stretch features); 2) use random clip and rotation for data augmentation; 3) each region in output mask is determined by combining inference result from multiple patches. More details can be found in [2] .

Based on the model above, we extract features folloing paper [3], the calculating details of calculating these features are programed in FeatureCalculation.py

 
### Dependencies
- Tensorflow 1.4
- OpenCV
- Scikit-image
- Numpy
- Matplotlib
- Pandas

### Files
nuclei_DS.py : main entry, read files and detecting cells using model and then call the function of feature extration.
FeatureCalculation.py : calculation of ten features mentioned in [3]
extractedFeatures.csv : extracted features of images in data fold after running all the programs.
data : some breast diagnosis cell images for feature extraction. Notice, xxx_label and xxx_mask are the files that we perform detecting cells program.
models : trained models
util : some useful functions that we use in the main function


#### Reference
[1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation,  	arXiv:1505.04597.</br>
[2] K.Chen, N. Zhang, L.S.Powers, J.M.Roveda, Cell Nuclei Detection and Segmentation for Computational Pathology Using Deep Learning, SpringSim 2019 Modeling and Simulation in Medicine, Society for Modeling and Simuation (SCS) International (accepted).
[3] W. Nick Street, W. H. Wolberg, and O. L. Mangasarian "Nuclear feature extraction for breast tumor diagnosis", Proc. SPIE 1905, Biomedical Image Processing and Biomedical Visualization, (29 July 1993);