# dlib_darknet
用dlib和darknet 分类器实现的盹睡系统，其中shape_predictor_68_face_landmarks.dat过大没有上传
将dlib文件夹放到目录下，下载`shape_predictor_68_face_landmarks.dat`放到`data`目录下，将`libdarknet.so`放在目录下
### 编译
```
mkdir build
cd build
cmake ..
make
```
注意源文件中包含的配置和参数文件的路径
