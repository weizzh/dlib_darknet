#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h> //include this headfile to use extract_image_4points()
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <time.h>
#include "include/CameraApi.h"
#include <dark.h>
#include <iostream>

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
using namespace std;
unsigned char * g_pRgbBuffer;
unsigned char * g_pGrayBuffer;

#define DISPLAY 0
#include <string>
using namespace dlib;
using namespace std;

int main()
{
	int iCameraCounts =4;
	int iStatus = -1;
	tSdkCameraDevInfo       tCameraEnumList[4];
	int                     hCamera;
	tSdkCameraCapbility     tCapability;
	tSdkFrameHead           sFrameInfo;
	BYTE*			        pbyBuffer;
	tSdkImageResolution     sImageSize;
	int i=0, j=0;
	int num=0;
/*
		cv::VideoCapture cap("../1.avi");
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}	
*/
#if DISPLAY
	image_window win, win_faces, win_left_roi;
#endif
	CameraSdkInit(1);
	CameraEnumerateDevice(tCameraEnumList,&iCameraCounts);
	printf("iCameralCounts = %d\n", iCameraCounts);
    if(iCameraCounts==0)
	{
		printf("No gige camera is found.\n");
		return -1;
    }
	iStatus = CameraInit(&tCameraEnumList[0],-1,-1,&hCamera);
	if(iStatus!=CAMERA_STATUS_SUCCESS)
	{
			printf("gige camera init failed.\n");
			return -1;
    }
	CameraGetCapability(hCamera,&tCapability);
    printf("CameraGetCapability \n");
	g_pGrayBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax);
//
	CameraSetAeState(hCamera, FALSE);
    double fExposureTime = 10000;
    double pfExposureTime;
    CameraSetExposureTime(hCamera, fExposureTime);
    CameraGetExposureTime(hCamera, &pfExposureTime);
    printf("The Exposure Time= %f\n", pfExposureTime);

	CameraPlay(hCamera);
    printf("CameraPlay \n");
	memset(&sImageSize,0,sizeof(tSdkImageResolution));
    sImageSize.iIndex=0xff;
    sImageSize.iHOffsetFOV=0;
    sImageSize.iVOffsetFOV=0;
    sImageSize.iWidthFOV=640;
    sImageSize.iHeightFOV=480;
    sImageSize.iWidth=640;
    sImageSize.iHeight=480;
    CameraSetImageResolution(hCamera,&sImageSize);
	
    if(tCapability.sIspCapacity.bMonoSensor)
	{
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    }
    printf("CameraSetIspOutFormat \n");
//    sleep(1);

    if(CameraSetOutPutIOMode(hCamera,1,IOMODE_GP_OUTPUT) == CAMERA_STATUS_SUCCESS)
        printf("set gpio out put mode success!\n");


	
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
	initial_network();

	array2d<unsigned char> gray_image(480, 640);
	array2d<unsigned char> gray_image_resized(240, 320);

	printf("make an empty gray image(%d, %d)\n",480, 640);
	long int frame_counter = 0;
	string roi_path = "./roi_eye/";
	clock_t begin, mid, end;


while(1)
{	begin = clock();
	printf("try to get Image buffer...");
	if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,100) == CAMERA_STATUS_SUCCESS)
	{
		printf("Get ImageBuffer Success!\n");
		frame_counter ++;
		cout << "get image: " << double( clock()-begin ) / CLOCKS_PER_SEC << endl;
		printf("The image size: %d\n", sFrameInfo.uBytes);
//		memcpy(g_pGrayBuffer, pbyBuffer,tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax);
		
		for(i=0; i<480; i++)
			for(j=0; j<640; j++)
			{
				gray_image[i][j] = *((unsigned char * )(pbyBuffer+i*640 +j));
			}
		CameraReleaseImageBuffer(hCamera, pbyBuffer);
		
	}
	else
	{
		printf("time out, fail to get image\n");
		return -1;	
	}
	resize_image(gray_image, gray_image_resized);
	cv::Mat cv_img=toMat(gray_image_resized);
	cv::namedWindow("temp");
	cv::imshow("temp", cv_img);
	cv::waitKey(10);




/*
	cv::Mat temp,resized,resized_gray;
	if(!cap.read(temp))
	{
		break;
	}
*/	
/*
	frame_counter ++;
	cout << "frame: " << frame_counter <<endl;
	cv::resize(temp, resized, cv::Size(640, 480));
	cv::cvtColor(resized, resized_gray, CV_BGR2GRAY);
	cv_image<bgr_pixel> cimg(resized);
	cv_image<unsigned char> cimg_gray(resized_gray);


	
	array2d<bgr_pixel> dlib_cimg;
	array2d<unsigned char> dlib_cimg_gray;
	assign_image(dlib_cimg, cimg);
	assign_image(dlib_cimg_gray, cimg_gray);

	cout << "risize time: "<< double( clock()-begin ) / CLOCKS_PER_SEC << endl;
	*/	

	mid = clock();
	array2d<unsigned char> left_roi(28, 28);
//			cout <<"set image size." << endl;
	std::vector<rectangle> dets;
	dets = detector(gray_image_resized);
	cout << "detect time: " << double( clock()-mid ) / CLOCKS_PER_SEC << endl;
	mid = clock();
	std::vector<full_object_detection> shapes;
	if (dets.size() == 0) 
	{
		cout<< "no face detected."<<endl;
		end = clock();
		cout << "total time: "<< double(end - begin)/CLOCKS_PER_SEC << end <<endl;
		
		cout << "FPS: " << CLOCKS_PER_SEC / double(end - begin) << endl;					
		continue;
	}
	for(unsigned long i =0; i<dets.size(); ++i)
		shapes.push_back(pose_model(gray_image_resized, dets[i]));
	cout<< "alignment time: "  << double( clock()-mid ) / CLOCKS_PER_SEC << endl;
	mid = clock();
	double scale = 2.; 
	int LEFT_EYE = 42;
	double center_x = 0.5 * ( shapes[0].part(LEFT_EYE + 3).x() + shapes[0].part(LEFT_EYE).x() );
	double center_y = 0.5 * ( shapes[0].part(LEFT_EYE + 5).y() + shapes[0].part(LEFT_EYE + 1).y() );
	double x_scale = scale * 0.5 * ( shapes[0].part(LEFT_EYE + 3).x() - shapes[0].part(LEFT_EYE).x() );
	double y_scale = scale * 0.5 * ( shapes[0].part(LEFT_EYE + 5).y() - shapes[0].part(LEFT_EYE + 1).y() );	
	std::array<dpoint, 4> LEFT_ROI;
//			cout<< "set the 4 points."<<endl;
		//the bottomleft point, note the y coordinate.
		#if 1
	LEFT_ROI[0](0) = center_x - x_scale;
	LEFT_ROI[0](1) = center_y + y_scale;
		//the topleft point
	LEFT_ROI[1](0) = LEFT_ROI[0](0);
	LEFT_ROI[1](1) = center_y - y_scale;
		//the topright point
	LEFT_ROI[2](0) = center_x + x_scale;
	LEFT_ROI[2](1) = LEFT_ROI[1](1);
		//the bottomright point
	LEFT_ROI[3](0) = LEFT_ROI[2](0);
	LEFT_ROI[3](1) = LEFT_ROI[0](1);
		#endif
//			cout<< "get the 4 points." <<endl;
//			for( int i = 0; i<4; ++i) {cout << "the LEFT_ROI is: " << LEFT_ROI[i] <<endl;}
//			cout << "the size of left_roi is :" << left_roi.size() << " " << left_roi.nc() << " " << left_roi.nr() <<endl;
	extract_image_4points(gray_image_resized, left_roi, LEFT_ROI);
	save_png(left_roi, roi_path+to_string(frame_counter)+".png");
	cout << "extract image time: " << double( clock()-mid ) / CLOCKS_PER_SEC << endl;

	cv::Mat cv_left_roi= toMat(left_roi);
	cv::namedWindow("left_roi");
	cv::imshow("left_roi", cv_left_roi);
	cv::waitKey(20);

	mid = clock();
	float X[2352];
	for(int chl=0; chl<3; chl ++)
		for(int het=0; het<28; het++)
			for(int wih=0; wih<28; wih++)
			{
				X[28*28*chl+28*het+wih] = left_roi[wih][het] /255.0;
			}
//			cout<< "get the network input data."<<endl;
			cout<< "get input data: "<< double( clock()-mid ) / CLOCKS_PER_SEC << endl;
			mid = clock();			
			bool predict_result = predict_class(X);
			string result_class= predict_result?"close":"open";
			
			cout<< "predict time: "<< double( clock()-mid ) / CLOCKS_PER_SEC  <<endl;							
			cout<< "the result is: " << result_class <<endl;		
#if DISPLAY
			win.clear_overlay();
			win.set_image(gray_image_resized);
			win.add_overlay(render_face_detections(shapes));
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(gray_image_resized, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));
			win_left_roi.set_image(left_roi);
#endif			
			end = clock();
			cout << "total time: " << double(end - begin)/CLOCKS_PER_SEC <<endl; 
			cout << "FPS: " << CLOCKS_PER_SEC / double(end - begin) << endl;
}	

}
