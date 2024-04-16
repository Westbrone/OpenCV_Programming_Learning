#include<opencv2/opencv.hpp> 
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <iostream>
#include <typeinfo>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <cstdio>






int main()
{
	/*
	cv::Mat picture = cv::imread("11.jpg",cv::IMREAD_GRAYSCALE); //��ȡһ��ͼ��
	cv::namedWindow("windows", cv::WINDOW_NORMAL);
	cv::imshow("windows", picture);
	*/
	

	/*
	//�˲�������
	cv::Mat bilateralImge;
	cv::bilateralFilter(picture, bilateralImge, 15, 99, 45);
	cv::imshow("�˲�������", bilateralImge);

	//��˹ģ��
	cv::Mat gaussImg;
	cv::GaussianBlur(picture, gaussImg,cv::Size(15,15),0);
	cv::imshow("��˹ģ������", gaussImg);

	//��ֵģ��
	cv::Mat medianImg;
	cv::medianBlur(picture, medianImg, 3);
	cv::imshow("��ֵģ������", medianImg);
	*/

	
	/* 
    //��ʴ������
	cv::mat erodemat, dilatemat, dilatemat1;
	cv::mat elementkernel = cv::getstructuringelement(cv::morph_rect, cv::size(3, 3));

	cv::erode(picture, erodemat, elementkernel, cv::point(-1, -1), -1);
	cv::imshow("��ʴ", erodemat);

	cv::dilate(picture, dilatemat, elementkernel, cv::point(-1, -1), -1);
	cv::imshow("���Ͳ���", dilatemat);

	//���Ը�ʴ�����ͣ�������Ҫ�����Ե��ϸ
	cv::dilate(erodemat, dilatemat1, elementkernel, cv::point(-1, -1), -1);
	cv::imshow("��ʴ->���Ͳ���", dilatemat1);
	*/


	/*
	//sobel����
	cv::Mat outputImg;
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	int dx = 1, dy = 1, sobelKernelSize = 3, scaleFactor = 1, deltaValue = 1;

	while (1) {
		cv::Sobel(picture, outputImg, CV_8UC1, dx, dy, sobelKernelSize, scaleFactor, deltaValue);

		int c = cv::waitKey(1);

		if ((char)c == 'q')
		{
			std::cout << "pressed to q" << std::endl;
			break;
		}
		else if ((char)c == 'a')
		{
			std::cout << "pressed to a" << std::endl;
			if (dx && dy)
				dx = 0;
			else
				dx = 1;
		}
		else if ((char)c == 'w')
		{
			std::cout << "pressed to w" << std::endl;
			if (dx && dy)
				dy = 0;
			else
				dy = 1;
		}
		else if ((char)c == 'e')
		{
			std::cout << "pressed to e" << std::endl;
			sobelKernelSize += 2;
		}
		else if ((char)c == 'r')
		{
			std::cout << "pressed to r" << std::endl;
			scaleFactor--;
		}
		else if ((char)c == 'c')
		{
			std::cout << "pressed to c" << std::endl;
			deltaValue--;
		}

		cv::imshow("windows", picture);
		cv::imshow("output", outputImg);
	}
	*/


	/*
	//filter2D
	cv::Mat_ < float>custom(3, 3);
	cv::Mat_ < float>kernel(2, 2);
	custom << 1, 2, 5, 6, 8, 1, 0, 1, 2;
	kernel << 1, 1, 1, 1;
	cv::namedWindow("Custom", cv::WINDOW_NORMAL);
	cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
	cv::namedWindow("Filter2D", cv::WINDOW_NORMAL);
	cv::Mat custom2, kernel2, filter2D, filter2D2;

	cv::filter2D(custom, filter2D, -1, kernel, cv::Point(-1, -1)); //���Եֻ���ظ���Ե�ľͽ���ֵ
	custom.convertTo(custom2, CV_8UC1);
	kernel.convertTo(kernel2, CV_8UC1);
	filter2D.convertTo(filter2D2, CV_8UC1);

	cv::imshow("Custom", custom2);
	cv::imshow("Kernel", kernel2);
	cv::imshow("Filter2D", filter2D2);
	cv::waitKey(0);
	*/


    /*
    //������˹��Ե���
    cv::Mat inputImg = cv::imread("11.jpg", cv::IMREAD_GRAYSCALE);
	cv::namedWindow("input", cv::WINDOW_NORMAL);
	cv::imshow("input", inputImg);

	cv::Mat outputImg;
	cv::Laplacian(inputImg, outputImg,-1,9,1,0);
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::imshow("output", outputImg);
	cv::waitKey(0);
	*/


    /*
    //���ƾ���ͼ��
    cv::Mat maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
    cv::namedWindow("mask", 0);
	
	int point1 = 100;
	int point2 = 200;
	cv::createTrackbar("Point1", "mask", &point1, 500);
	cv::createTrackbar("Point2", "mask", &point2, 500);
    
	while(1){
		cv::rectangle(maskImg, cv::Point(point1,point1), cv::Point(point2, point2), cv::Scalar(0, 255, 255), 3, cv::LINE_4);
		cv::imshow("mask", maskImg);
		cv::waitKey(1);

		maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);//���³�ʼ�����ͼ�񣬲�Ȼ�����ѵ�
	}
	*/


   /*
    //����ֱ�ߣ�ԲȦ������
	cv::Mat maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
	cv::namedWindow("mask", 0);

	int point1 = 100;
	int point2 = 200;
	int ratio = 1;
	int thickness = 1;
	int radius = 3;
	int shift = 1;
	cv::createTrackbar("Point1", "mask", &point1, 600);
	cv::createTrackbar("Point2", "mask", &point2, 600);
	cv::createTrackbar("Ratio", "mask", &shift, 10);
	cv::createTrackbar("Thickness", "mask", &thickness, 25);
	cv::createTrackbar("Radius", "mask", &radius, 100);

	while (1) {
		//cv::arrowedLine(maskImg, cv::Point(point1, point1), cv::Point(point2, point2), cv::Scalar(0, 255, 255),thickness,cv::LINE_AA,0,(double)ratio/10.0);
		//cv::circle(maskImg, cv::Point(point1,point2), radius, cv::Scalar(0,255,255), thickness,cv::LINE_AA,shift);

		cv::Point rr = cv::Point(point1, point2);
	    cv::Point tt = cv::Point(500, 500);
		cv::Point kk = cv::Point(30, 30);

		cv::rectangle(maskImg, kk, tt, cv::Scalar(0, 0, 255), 5, cv::LINE_8);
		cv::line(maskImg, rr, tt, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
		std::cout << cv::clipLine(cv::Size(500, 500), rr, tt) << std::endl;
		cv::imshow("mask", maskImg);
		int c  = cv::waitKey(1);

		if ((char)c == 'c')
			break;

		maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);//���³�ʼ�����ͼ�񣬲�Ȼ�����ѵ�
	}
	*/



    /*
    //��Բ��ͼ
	cv::Mat maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
	cv::namedWindow("mask", 0);

	int width = 50;
	int height = 100;
	int angle = 0;
	int startAngle = 0;
	int endAngle = 360;
	int jiange = 1;
	cv::createTrackbar("Point1", "mask", &width, 400);
	cv::createTrackbar("Point2", "mask", &height, 400);
	cv::createTrackbar("Ratio", "mask", &angle, 360);
	cv::createTrackbar("Thickness", "mask", &startAngle, 360);
	cv::createTrackbar("Radius", "mask", &endAngle, 360);
	cv::createTrackbar("Jiange", "mask", &jiange, 20);

	while (1) {

		//cv::ellipse(maskImg,cv::Point(500,500), cv::Size(width,height), angle, startAngle,endAngle ,cv::Scalar(0, 0, 255), 3);
		std::vector< cv::Point > vvv;
		cv::ellipse2Poly(cv::Point(500, 500), cv::Size(width, height), angle, startAngle, endAngle, jiange, vvv);

		for (int i = 0; i < vvv.size(); i++)
		{
			maskImg.at<cv::Vec3b>(vvv[i])[0] = 0;
			maskImg.at<cv::Vec3b>(vvv[i])[1] = 255;
			maskImg.at<cv::Vec3b>(vvv[i])[2] = 255;
		}


		cv::imshow("mask", maskImg);
		int c = cv::waitKey(1);

		if ((char)c == 'c')
			break;

		maskImg = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);//���³�ʼ�����ͼ�񣬲�Ȼ�����ѵ�
	}
	*/



   /*
    //ͼ��������ı���Ϣ
	cv::Mat maskImg = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
	cv::namedWindow("output", 0);

	cv::putText(maskImg, "Heloo", cv::Point(100, 100), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
	int BL = 0;
	std::cout << cv::getTextSize("Heloo", cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, 2, &BL) << std::endl;
	cv::addText(maskImg, "Heloo", cv::Point(100, 200), "Times", 30, cv::Scalar(0, 255, 255), cv::QT_FONT_NORMAL);


	cv::imshow("output", maskImg);

	cv::waitKey(0);
	*/


   /*
   //���㹹�������
   cv::Mat maskImg = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
   cv::namedWindow("output", 0);
   std::vector<cv::Point> pts = { cv::Point(100,100),cv::Point(200,100), cv::Point(300,300),cv::Point(400,450) };

   cv::fillConvexPoly(maskImg, pts,cv::Scalar(0, 255, 255));
   cv::polylines(maskImg, pts, 0, cv::Scalar(0, 255, 255), 5);
   cv::imshow("output", maskImg);
   cv::waitKey(0);
   */

   /*
   //���Ƽ�����
   cv::Mat maskImg = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
   cv::namedWindow("output", 0);
   std::vector<std::vector<cv::Point >> contours = {
	   {cv::Point(100,100),cv::Point(100,150),cv::Point(150,150)},
	   {cv::Point(300,300),cv::Point(400,400)},
	   {cv::Point(100,300),cv::Point(200,400)}
   };
   std::cout << contours.size() << std::endl;

   for (int i = 0; i < (int)contours.size(); i++)
   cv::drawContours(maskImg, contours, i, cv::Scalar(0, 255, 255), 5);
   cv::imshow("output",maskImg);
   cv::waitKey(0);
   */


   /*
   //����ͼ���С
    cv::Mat inputImg = cv::imread("11.jpg");
	cv::namedWindow("input", 0);
	cv::Mat outputImg;
	cv::namedWindow("output", 0);
	cv::resize(inputImg, outputImg, cv::Size(100, 200), 0, 0);
	cv::imshow("input", inputImg);
	cv::imshow("output", outputImg);
	cv::waitKey(0);
	*/


    /*
    //͸�ӱ任�ӽ�
	cv::Mat inputImg = cv::imread("warpperspective.jpg");
	cv::namedWindow("input", 0);
	cv::Point2f srcPoints[] = {
	cv::Point(561,1053),
	cv::Point(1902,935),
	cv::Point(1867,4694),
	cv::Point(3085,3988),
	};

	cv::Point2f dstPoints[] = {
	cv::Point(0,0),
	cv::Point(1350,0),
	cv::Point(0,3700),
	cv::Point(1350,3700),
	};

	cv::Mat Matrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
	cv::Mat outputImg;
	cv::namedWindow("output", 0);
	cv::warpPerspective(inputImg, outputImg,Matrix,cv::Size(1350,3700));
	cv::imshow("input", inputImg);
	cv::imshow("output", outputImg);
	cv::waitKey(0);
	*/

    

    /*
    //����任
    int cnt = 1;
    while (cnt++) {
		cv::Mat inputImg = cv::imread("22.jpg");
		cv::namedWindow("input", 0);
		cv::Mat outputImg = cv::Mat::zeros(cv::Size(inputImg.cols, inputImg.rows), inputImg.type());
		cv::namedWindow("output", 0);

		cv::Point2f inpMat[3];
		cv::Point2f outMat[3];

		inpMat[0] = cv::Point2f(0.0,0.0);
		inpMat[1] = cv::Point2f(inputImg.cols,0.0);
		inpMat[2] = cv::Point2f(0.0,inputImg.rows);
		outMat[0] = cv::Point2f(0, 200);
		outMat[1] = cv::Point2f(500, 100+cnt);
		outMat[2] = cv::Point2f(170, 520-cnt);

		cv::Mat M = cv::getAffineTransform(inpMat, outMat);
		cv::warpAffine(inputImg, outputImg, M, outputImg.size());

		cv::Point2f center(inputImg.cols / 2, inputImg.rows / 2);
		double angle = 50;
		double scale = 1;
		cv::Mat MM = cv::getRotationMatrix2D(center, angle, scale);
		cv::Mat dstMat;
		cv::warpAffine(inputImg, dstMat, MM, inputImg.size());
		cv::imshow("input", inputImg);
		cv::imshow("output",dstMat);
		cv::waitKey(1);
	    }
	*/



	/*
    //��ӳ�䡢
    cv::Mat src = cv::imread("22.jpg");
	cv::namedWindow("input", 0);
	cv::namedWindow("output", 0);
	cv::namedWindow("mapx", 0);
	cv::namedWindow("mapy", 0);


	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	cv::Mat map_x = cv::Mat::zeros(src.size(),CV_32FC1);
	cv::Mat map_y = cv::Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.cols; i++) 
	{
		for (int j = 0; j < src.rows; j++)
		{
			map_x.at<float>(cv::Point(i, j)) = src.cols - i;
			map_y.at<float>(cv::Point(i, j)) = src.rows - j;
		}
	}

	cv::remap(src, dst, map_x, map_y, CV_INTER_LINEAR);

	cv::imshow("input", src);
	cv::imshow("output", dst);
	cv::imshow("mapx", map_x);
	cv::imshow("mapy", map_y);
	cv::waitKey(0);
	*/


    /*
    //������չƽ
	cv::Mat src = cv::imread("33.jpg");
	cv::namedWindow("input", 0);
	cv::namedWindow("output", 0);
	cv::namedWindow("output2", 0);
	cv::Mat img1, img2;
	cv::Point2f center = cv::Point2f(src.cols / 2, src.rows / 2); //��������ͼ���е�ԭ��
   //��������任
	cv::warpPolar(src, img1, cv::Size(300, 600), center, center.x, cv::InterpolationFlags::INTER_LINEAR + cv::WarpPolarMode::WARP_POLAR_LINEAR);
	cv::warpPolar(src, img2, cv::Size(300, 600), center, center.y, cv::InterpolationFlags::INTER_LINEAR + cv::WarpPolarMode::WARP_POLAR_LINEAR);
	cv::imshow("input", src);
	cv::imshow("output", img1);
	cv::imshow("output2", img2);
	cv::waitKey(0);
	*/
	

	/*
   //����ͼ��
	cv::Mat srcImg = cv::imread("11.jpg");
	cv::Mat dstImg;
	cv::getRectSubPix(srcImg, cv::Size(srcImg.cols / 2, srcImg.rows / 2), cv::Point2f(srcImg.cols / 2, srcImg.rows / 2), dstImg, -1);

	cv::imshow("srcImg", srcImg);
	cv::imshow("dstImg", dstImg);
	cv::waitKey(0);
	*/


	/*
    //����Ӧ��ֵ��
	cv::Mat dst,dst1;
	cv::Mat srcImg = cv::imread("11.jpg", cv::IMREAD_GRAYSCALE);
	int maxVal = 255;
	int blockSize = 11;
	double C = 0;
	cv::adaptiveThreshold(srcImg, dst, maxVal, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
	cv::adaptiveThreshold(srcImg, dst1, maxVal, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);

	cv::imshow("MEAN", dst);
	cv::imshow("GAUSSIAN", dst1);
	cv::waitKey(0);
	*/


     /*
    //����ͼ
    cv::Mat image = cv::imread("11.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat imageIntegral;
	cv::integral(image, imageIntegral, CV_32F); //�������ͼ
	cv::normalize(imageIntegral, imageIntegral, 0, 255, CV_MINMAX);  //��һ����������ʾ
	cv::Mat imageIntegralNorm;
	std::cout << imageIntegralNorm.type() << std::endl;

	cv::convertScaleAbs(imageIntegral, imageIntegralNorm); //����ת��Ϊ8λint����
	
	cv::imshow("Source Image", image);
	cv::imshow("Integral Image", imageIntegralNorm);
	cv::waitKey(0);
	*/




    /*���þ�����зָ�
    // Show the source image
    cv::Mat src = cv::imread("3.jpg");
	cv::imshow("Source Image", src);
	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	cv::Mat mask;
	inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);//����ɫ���1���������0
	src.setTo(cv::Scalar(0, 0, 0), mask);//��1��Ӧ�ı�ɺ�ɫ
	// Show output image
	imshow("Black Background Image", src);
	// Create a kernel that we will use to sharpen our image
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); // an approximation of second derivative, a quite strong kernel
	// do the laplacian filtering as it is
	// well, we need to convert everything in something more deeper then CV_8U
	// because the kernel has some negative values,
	// and we can expect in general to have a Laplacian image with negative values
	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
	// so the possible negative number will be truncated
	cv::Mat imgLaplacian;
	cv::filter2D(src, imgLaplacian, CV_32F, kernel);//������˹����ͻ��ͼ���еı�Ե��ϸ��
	cv::Mat sharp;
	src.convertTo(sharp, CV_32F);
	cv::Mat imgResult = sharp - imgLaplacian;//�������Ϊ��ͼ���еı�Ե������ͻ����ʾ
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	cv::imshow( "Laplace Filtered Image", imgLaplacian );
	cv::imshow("New Sharped Image", imgResult);
	// Create binary image from source image
	cv::Mat bw;
	cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
	cv::imshow("Grayy Image", bw);
	cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imshow("Binary Image", bw);
	// Perform the distance transform algorithm
	cv::Mat dist;
	cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	cv::imshow("Distance Transform Image", dist);
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
	// Dilate a bit the dist image
	cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
	cv::dilate(dist, dist, kernel1);
	cv::imshow("Peaks", dist);
	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	cv::Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
    std::vector<std::vector<cv::Point> > contours;
	findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
	}
	imshow("Markers", markers);
	// Draw the background marker
	cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
	cv::Mat markers8u;
	markers.convertTo(markers8u, CV_8U, 10);
	cv::imshow("Markers", markers8u);
	// Perform the watershed algorithm
	watershed(imgResult, markers);
	
	cv::Mat mark;
	markers.convertTo(mark, CV_8U);
	cv::imshow("Markers_v1", mark);
	cv::bitwise_not(mark, mark);
	imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
	// image looks like at that point
	// Generate random colors
	std::vector<cv::Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = cv::theRNG().uniform(0, 256);
		int g = cv::theRNG().uniform(0, 256);
		int r = cv::theRNG().uniform(0, 256);
		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	// Create the result image
	cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
			{
				dst.at<cv::Vec3b>(i, j) = colors[index - 1];
			}
		}
	}
	// Visualize the final image
	imshow("Final Result", dst);
	cv::waitKey(0);
	*/




    /*
	//distance transform
	cv::namedWindow("input", 0);
	cv::namedWindow("output", 0);
	cv::Mat input = cv::imread("3.jpg", cv::IMREAD_GRAYSCALE);

	cv::threshold(input, input, 200,255, cv::THRESH_BINARY);
	cv::imshow("input", input);
	cv::Mat dst;
	//ֻ���ܶ�ֵ��ͼ��ͳ�Ƶ��Ǿ�������0Ԫ���ж�Զ��L1.L2.C�ǲ�ͬ��ͳ�ƾ����ָ�꣩��
	//Ϊÿһ��0ֵԪ�ؾ�������Ķ��������ͬ�ı�ǩ
	cv::distanceTransform(input, dst, cv::DIST_C, cv::DIST_MASK_PRECISE, CV_8U);
	//ͨ����ӡ��dst����Ϊdst��MATͼ��
	//std::cout << dst << std::endl;
	cv::imshow("output", dst);
	*/

    /*
    //cv::blendLinear
	cv::namedWindow("LZP", 0);
	cv::namedWindow("HH", 0);
    cv::Mat LZP = cv::imread("11.jpg", cv::IMREAD_COLOR);
	cv::Mat HH = cv::imread("hh.jpg", cv::IMREAD_COLOR);
	cv::resize(LZP, LZP, HH.size());
	cv::namedWindow("OUTPUT", 0);
	cv::Mat weight1 = cv::Mat::ones(HH.size(), CV_32FC1);
	cv::Mat weight2(HH.size(),CV_32FC1,cv::Scalar(1));
	cv::Mat output;
	cv::blendLinear(LZP, HH, weight1, weight2, output);
	cv::imshow("LZP", output);
	cv::imshow("HH", output);
	cv::imshow("OUTPUT", output);
	*/
	

    
    /*
	//��ˮ���㷨
    cv::Mat img0 = cv::imread("22.jpg", cv::IMREAD_COLOR);
	std::cout << 123123123123 << std::endl;
	cv::Mat imgGray;
	img0.copyTo(img);
	cv::cvtColor(img, markerMask, cv::COLOR_BGR2GRAY);
	cv::cvtColor(markerMask, imgGray, cv::COLOR_GRAY2BGR);
	markerMask = cv::Scalar::all(0);
	cv::imshow("image", img);
	cv::setMouseCallback("image", onMouse, 0);
	for (;;)
	{
		char c = (char)cv::waitKey(0);
		if (c == 27)
			break;
		if (c == 'r')
		{
			markerMask = cv::Scalar::all(0);
			img0.copyTo(img);
			imshow("image", img);
		}
		if (c == 'w' || c == ' ')
		{
			int i, j, compCount = 0;
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			findContours(markerMask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
			if (contours.empty())
				continue;
			cv::Mat markers(markerMask.size(), CV_32S);
			markers = cv::Scalar::all(0);
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
				drawContours(markers, contours, idx, cv::Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
			if (compCount == 0)
				continue;
			std::vector<cv::Vec3b>colorTab;
			for (i = 0; i < compCount; i++)
			{
				int b = cv::theRNG().uniform(0, 255);
				int g = cv::theRNG().uniform(0, 255);
				int r = cv::theRNG().uniform(0, 255);
				colorTab.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			double t = (double)cv::getTickCount();
			watershed(img0, markers);
			t = (double)cv::getTickCount() - t;
			printf("execution time = %gms\n", t * 1000. / cv::getTickFrequency());
			cv::Mat wshed(markers.size(), CV_8UC3);
			// paint the watershed image
			for (i = 0; i < markers.rows; i++)
				for (j = 0; j < markers.cols; j++)
				{
					int index = markers.at<int>(i, j);
					if (index == -1)
						wshed.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
					else if (index <= 0 || index > compCount)
						wshed.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
					else
						wshed.at<cv::Vec3b>(i, j) = colorTab[index - 1];
				}
			wshed = wshed * 0.5 + imgGray * 0.5;
			imshow("watershed transform", wshed);
			cv::waitKey(0);
		}
	}
	*/


    /*
	//grabCut
	cv::namedWindow("input", 0);
	cv::Mat img = cv::imread("11.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("OUTPUT", 0);
	cv::Mat mask, bg, fg;
	cv::Mat result = img.clone();
	cv::Rect r = cv::selectROI(img);

	std::cout << 1231231 << std::endl;
	cv::grabCut(img, mask, r, bg, fg, 1, cv::GrabCutModes::GC_INIT_WITH_RECT);
	std::cout << 1231231 << std::endl;

	for (int i = 0; i < img.cols;i++)
	{
		for (int j = 0;j < img.rows;j++)
		{
			if ((int)mask.at<uchar>(cv::Point(i, j)) == 0)//0���������Եı���
			{
				result.at<cv::Vec3b>(cv::Point(i, j))[0] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[1] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[2] = 0;
			}
			else if ((int)mask.at<uchar>(cv::Point(i, j)) == 1)//0���������Եı���(ǰ��)
			{
				result.at<cv::Vec3b>(cv::Point(i, j))[0] = 255;
				result.at<cv::Vec3b>(cv::Point(i, j))[1] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[2] = 0;
			}
			else if ((int)mask.at<uchar>(cv::Point(i, j)) == 2)//2�����ſ��ܵı���
			{
				result.at<cv::Vec3b>(cv::Point(i, j))[0] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[1] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[2] = 255;
			}
			else if ((int)mask.at<uchar>(cv::Point(i, j)) == 3)//3�����ſ��ܵ�ǰ��
			{
				result.at<cv::Vec3b>(cv::Point(i, j))[0] = 0;
				result.at<cv::Vec3b>(cv::Point(i, j))[1] = 255;
				result.at<cv::Vec3b>(cv::Point(i, j))[2] = 255;
			}
		}
	}

	cv::imshow("input", img);
	cv::imshow("OUTPUT", result);

	cv::waitKey(0);
	*/


    /*
    //��ȡ��Ƶ���е�һ��ͼ��
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error: Failed to open camera" << std::endl;
		return -1;
	}

	// ��ȡ��һ֡ͼ��
	cv::Mat frame;
	cap >> frame;

	// ���ͼ���Ƿ��ȡ�ɹ�
	if (frame.empty()) {
		std::cerr << "Error: Failed to capture frame" << std::endl;
		return -1;
	}

	// ����ͼ���ļ�
	std::string filename = "first_frame.jpg";
	bool success = cv::imwrite(filename, frame);
	if (success) {
		std::cout << "First frame saved as " << filename << std::endl;
	}
	else {
		std::cerr << "Error: Failed to save frame" << std::endl;
		return -1;
	}

	// �ر�����ͷ
	cap.release();
	*/


    /*
	// ������ͷ
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error: Failed to open camera" << std::endl;
		return -1;
	}

	// ������Ƶ�������������VideoWriter����
	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	// ��ȡ����ͷ��֡��
	double fps = cap.get(cv::CAP_PROP_FPS);
	cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),fps, cv::Size(frame_width, frame_height));

	// ��ʼ¼����Ƶ
	std::cout << "Recording video..." << std::endl;
	double start_time = cv::getTickCount();
	while (true) {
		cv::Mat frame;
		cap >> frame; // ��ȡ֡

		// ���ͼ���Ƿ��ȡ�ɹ�
		if (frame.empty()) {
			std::cerr << "Error: Failed to capture frame" << std::endl;
			break;
		}

		// д��֡����Ƶ
		video.write(frame);

		// ���¼�Ƴ���5�룬�˳�ѭ��
		double current_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
		if (current_time >= 5.0) {
			std::cout << "Recording finished." << std::endl;
			break;
		}
	}

	// �ͷ���Դ
	cap.release();
	video.release();
	*/


    /*
    //floodfill��ˮ����
	cv::namedWindow("INPUT", 0);
	cv::namedWindow("OUTPUT", 0);
	cv::Mat img = cv::imread("22.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = img.clone();
	cv::Mat output,mask;
    cv::Rect r;

	cv::floodFill(dst, mask, cv::Point(1200, 1500), cv::Scalar(0), &r, cv::Scalar(0), cv::Scalar(50), cv::FLOODFILL_FIXED_RANGE);
	cv::imshow("INPUT", img);
	cv::imshow("OUTPUT", dst);
	cv::waitKey(0);
	*/




    /*
    //��������ֱ��ͼcalcHist
	cv::namedWindow("INPUT", 0);
	cv::namedWindow("OUTPUT", 0);
	cv::Mat img = cv::imread("22.jpg", cv::IMREAD_COLOR);
	cv::resize(img, img,cv::Size(1080, 1080));
	cv::Mat dst = img.clone();

    //Step 1 �趨ֱ��ͼ����
	float range[2] = { 0,255 };
	const float* histrange = { range };
	cv::Mat histout;

	const int histsize = 256;//����0-255
	const int channelnum = 0;
	//hisout �а������ش�С������ͳ����һ��ͼ������ص�������������һ��һ��256�е�mat
	cv::calcHist(&img, 1, &channelnum, cv::Mat(), histout, 1, &histsize, &histrange);

	std::cout << histout.size() << std::endl;
	histout = histout.t();
	std::cout << histout.size() << std::endl;
	//Step 2 ��ӡ��ÿһ�е�����
	int sum = 0;
	for (int k = 0; k < 256;k++)
	{
		sum += (int)histout.at<float>(cv::Point(k, 0));//����һ��ͼ�������ص�
		std::cout << (int)histout.at<float>(cv::Point(k, 0)) << std::endl;
	}
	std::cout << 1231231231<< std::endl;

	//Step 3
	int max = 0, index;
	for (int k = 0; k < 256;k++)
	{
		if((int)histout.at<float>(cv::Point(k, 0))>max)
		{ 
		     max = (int)histout.at<float>(cv::Point(k, 0));//����һ��ͼ���������ص�
		     index = k;
		}
	}
	double ratio = max / 1080;



	//Step 4����ֱ��ͼ
	cv::Mat graph = cv::Mat::zeros(cv::Size(1080,1080), CV_8UC1);
	int ref;
	for (int k = 0; k < 256;k++)
	{
		if (k !=0)
		{
			cv::line(graph, cv::Point((k - 1) * 2, 1079 - ref / ratio), 
				cv::Point(k * 2, 1079 - (int)histout.at<float>(cv::Point(k, 0))/ratio),
				cv::Scalar(255),1);
		}
		ref = (double)histout.at<float>(cv::Point(k, 0));
	}


	cv::imshow("INPUT", img);
	cv::imshow("OUTPUT", graph);
	cv::waitKey(0);
	*/


   /*
    //calcBackProject for segmentation
    int histsize = 2;
	cv::Mat img = cv::imread("22.jpg");
	//cv::cvtColor(img,img,cv::COLOR_BGR2HSV);

	cv::namedWindow("BackProj", 0);
	cv::namedWindow("Input", 0);
	cv::imshow("Input", img);

	float range[] = {0,255 };
	const float *histRange[] = { range };
	cv::Mat hist;const int channelNum = 2;
	calcHist(&img, 1, &channelNum, cv::Mat(), hist, 1, &histsize, histRange, true, false);
	normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
	cv::Mat backProject;cv::calcBackProject(&img, 1, &channelNum, hist, backProject, histRange, 1);
	cv::imshow("BackProj", backProject);
	cv::waitKey(0);
	*/


	/*
    //EMD����ֱ��ͼ֮������ƶ�
	cv::Mat tmpl = cv::imread("cat2.jpg", 1);
	//cv::cvtColor(tmpl, tmpl, CV_BGR2HSV);
	cv::Mat tmp2 = cv::imread("cat1.jpg",1);
	//cv::cvtColor(tmp2, tmp2, CV_BGR2HSV);
	int histsize = 256;
	cv::Mat histout1;
	cv::Mat histout2;
	float range[] = { 0,255 }; 
	const float *histRange[] = {range};
	const int channelNum = 2;
	calcHist(&tmpl, 1, &channelNum, cv::Mat(), histout1, 1, &histsize, histRange, true, false);

	calcHist(&tmp2, 1, &channelNum, cv::Mat(), histout2, 1, &histsize, histRange, true, false);

	std::cout << histout1.size() << std::endl;
	///creating signature steps
	cv::Mat temp1(cv::Size(3,256),CV_32FC1);
	cv::Mat temp2(cv::Size(3,256),CV_32FC1);
	std::cout << temp1.size() << std::endl;
	for (int i = 0;i < 256;i++)
	{
		temp1.at<float>(i, 0) = histout1.at<float>(i, 0);
		temp1.at<float>(i, 2) = (float)i;//������ϵ�������
		temp1.at<float>(i, 1) = (float)0;
	}
	for (int i = 0;i < 256;i++)
	{
		temp2.at<float>(i, 0) = histout2.at<float>(i, 0);
		temp2.at<float>(i, 2) = (float)i;
		temp2.at<float>(i, 1) = (float)0;
	}
	///EMD
	std::cout << std::to_string(cv::EMD(temp1, temp2, 2,cv::DIST_L1)) << std::endl;//Խ�ӽ�0����Լ������ֱ��ͼͳ����
	*/


    //threshflod() and findContours() and drawContours()  and rectangle(contours)
 //   cv::Mat img = cv::imread("xiantiao.png", cv::IMREAD_GRAYSCALE);
	//cv::namedWindow("IN", 0);
	//cv::namedWindow("OUTPUT", 0);
	//cv::namedWindow("mask", 0);


	//cv::Mat out;
	//cv::threshold(img, out, 100, 255, cv::THRESH_BINARY);

	//std::vector<std::vector<cv::Point>> contours;
	//std::vector<cv::Vec4i> hierarchy;

	//

	//cv::findContours(out, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	//std::cout << hierarchy[0][1] << 123412 << std::endl;
	//cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC3);

	//std::cout << contours[0]<<12312312312 << std::endl;
	//std::cout << contours[1] <<123412<< std::endl;
	//std::cout << contours[2] << 12313123151234<<std::endl;
	//cv::drawContours(mask, contours, -2, cv::Scalar(0, 255, 255), 5);

	//cv::Rect r = cv::boundingRect(contours[2]);

	////cv::rectangle(out, r, cv::Scalar(255,0, 0), 20);
	//cv::rectangle(out, r, cv::Scalar(255, 0, 0), 5);


	//cv::imshow("IN", img);
	//cv::imshow("OUTPUT", out);
	//cv::imshow("mask", mask);
	//cv::waitKey(0);


















	return 0;
}