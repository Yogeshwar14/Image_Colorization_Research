# Image_Colorization_Research
Image Colorization model provides modern alternative and an estimate of how old black and white images and videos can be colorized in few seconds.

## Model 1
This model for image colorization is trained from scratch with 2042 training images. The model is trained using a Convolution Neural Network through 8 layers.

![image](https://github.com/Yogeshwar14/Image_Colorization_Research/assets/71761505/7ca4553d-7910-4aad-8d97-3b24ee76f8ea)


## Model 2
This model for image colorization is developed using Auto-Encoders. Input images are encoded to a latent space representation and then decoded to give the output image. 
1000 images are used for training and another 1000 are used for testing

![image](https://github.com/Yogeshwar14/Image_Colorization_Research/assets/71761505/864f340e-1883-41d7-80e5-c87dcda98b14)

## Model 3

The third model is based upon the approach discussed by Zhang et al. in their paper, ‘Colorful Image Colorization’. As an extension of the third model, the L and ab color channel manipulation has been tried to
colorize videos. Caffemodel is the pre-trained model used, along with numpy cluster points and prototxt files. The imagenet dataset with 12,81,167 training images, 50,000 validation images and 1,00,000 test
images has been used.

![image](https://github.com/Yogeshwar14/Image_Colorization_Research/assets/71761505/3db95306-2763-4d07-b27e-d3c8ace2648b)
