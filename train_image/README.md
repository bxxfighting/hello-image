### 背景:
> 我们要训练一个手写数字识别的模型  
> 先通过canvas人工手写一些训练图片，这些图片规格为透明背景、黑色字体、740 * 844大小的png格式  
> 透明背景是为了以后可以随意更换背景  
> 现在根据算法工程师的要求，需要将图片转换成白色背景，128 * 128大小的jpg  
> 同时，由于训练图片集有限，需要通过旋转、平移，将一张图片当十张图片使用  

python版本: 3.6.8
依赖库：Pillow==6.0.0  
        opencv-python==4.1.0.25  
        numpy==1.16.3