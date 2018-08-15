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

#include <dark.h>
#include <iostream>

#define DISPLAY 0
#include <string>
using namespace dlib;
using namespace std;

int main()
{

	try
	{
		cv::VideoCapture cap("../1.avi");
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}	

#if DISPLAY
		image_window win, win_faces, win_left_roi;
#endif
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
		initial_network();
		long int frame_counter = 0;
		clock_t begin, mid, end;
		while(1)
		{
			

			begin = clock();
			cout << "begin time: " << begin <<endl;
			cv::Mat temp,resized,resized_gray;
			if(!cap.read(temp))
			{
				break;
			}
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
			mid = clock();
			array2d<bgr_pixel> left_roi(28, 28);
//			cout <<"set image size." << endl;
			std::vector<rectangle> dets;
			dets = detector(dlib_cimg_gray);
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
				shapes.push_back(pose_model(dlib_cimg_gray, dets[i]));
			cout<< "alignment time: "  << double( clock()-mid ) / CLOCKS_PER_SEC << endl;
			mid = clock();
			double scale = 1.4; 
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
			extract_image_4points(dlib_cimg, left_roi, LEFT_ROI);	
			cout << "extract image time: " << double( clock()-mid ) / CLOCKS_PER_SEC << endl;
			mid = clock();
			float X[2352];
			for(int chl=0; chl<3; chl ++)
				for(int het=0; het<28; het++)
					for(int wih=0; wih<28; wih++)
						{
							if(chl==0) X[28*28*chl+28*het+wih] = left_roi[wih][het].red /255.0;
							if(chl==1) X[28*28*chl+28*het+wih] = left_roi[wih][het].green / 255.0;
							if(chl==2) X[28*28*chl+28*het+wih] = left_roi[wih][het].blue / 255.0;
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
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));
			win_left_roi.set_image(left_roi);
#endif			
			end = clock();
			cout << "total time: " << double(end - begin)/CLOCKS_PER_SEC <<endl; 
			cout << "FPS: " << CLOCKS_PER_SEC / double(end - begin) << endl;	

		}

	}
	catch(serialization_error& e)
	{
		cout<< "need face landmarking model file." <<endl;
	}

	catch(exception& e)
	{
		cout << e.what() <<endl;
	}


}