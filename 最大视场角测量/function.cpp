#include"function.h"

int ImageBinarization(Mat src) {   /*对灰度图像二值化，自适应门限threshold*/
	int i, j, width, height, step, chanel, threshold;
	/*size是图像尺寸，svg是灰度直方图均值，va是方差*/
	float size, avg, va, maxVa, p, a, s;
	unsigned char *dataSrc;
	float histogram[256];

	width = src.rows;
	height = src.cols;
	dataSrc = (unsigned char *)src.data; //
	step = src.step / sizeof(char);
	chanel = src.channels();
	/*计算直方图并归一化histogram*/
	for (i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width*chanel; j++)
		{
			histogram[dataSrc[i*step + j] - '0' + 48]++;
		}
	}
	size = width * height;
	for (i = 0; i < 256; i++)
	{
		histogram[i] /= size;
	}
	/*计算灰度直方图中值和方差*/
	avg = 0;
	for (i = 0; i < 256; i++)
	{
		avg += i*histogram[i];
	}
	va = 0;
	for (i = 0; i < 256; i++)
	{
		va += fabs(i*i*histogram[i] - avg*avg);
	}
	/*利用加权最大方差求门限*/
	threshold = 20;
	maxVa = 0;
	p = a = s = 0;
	for (i = 0; i < 256; i++)
	{
		p += histogram[i];
		a += i*histogram[i];
		s = (avg*p - a)*(avg*p - a) / p / (1 - p);
		if (s > maxVa)
		{
			threshold = i;
			maxVa = s;
		}
	}
	///*二值化*/
	for (i = 0; i < height; i++)
	{
		for (j = 0; j<width*chanel; j++)
		{
			if (dataSrc[i*step + j] > threshold)
			{
				dataSrc[i*step + j] = 255;
			}
			else
			{
				dataSrc[i*step + j] = 0;
			}
		}
	}
	//返回合适阈值
	return threshold;
}

// 最大类间方差法 http://blog.csdn.net/guoyk1990/article/details/7606032
// 灰度直方图 http://blog.csdn.net/qq_20823641/article/details/51932798
// 对rows、cols、data、step、channels的理解 https://www.douban.com/note/265479171/
// OpenCV2:Mat属性type，depth，step http://www.tuicool.com/articles/eUbuYn
// 实现原理：http://blog.csdn.net/a153375250/article/details/50970104
// 七种阈值常见分割代码：http://blog.csdn.net/xw20084898/article/details/17564957

int getAdaptiveThreshold(Mat img) {  
    int T = 0;             // 阈值  
    int height = img.rows; // rows 是 行数 相当于 height 对应 .y
    int width  = img.cols; // cols 是 列数 相当于 width 对应 .x
    int step = img.step;  
    int channels  = img.channels();   // 通道，矩阵中的每一个矩阵元素拥有的值的个数
    uchar* data  = (uchar*)img.data;  // Mat对象中的一个指针，指向内存中存放矩阵数据的一块内存 (uchar* data) 
    double gSum0;      // 第一类灰度总值  
    double gSum1;      // 第二类灰度总值  
    double N0 = 0;     // 前景像素数  
    double N1 = 0;     // 背景像素数  
    double u0 = 0;     // 前景像素平均灰度  
    double u1 = 0;     // 背景像素平均灰度  
    double w0 = 0;     // 前景像素点数占整幅图像的比例为ω0  
    double w1 = 0;     // 背景像素点数占整幅图像的比例为ω1  
    double u = 0;      // 总平均灰度  
    double tempg = -1; // 临时类间方差  
    double g = -1;     // 类间方差  
    double Histogram[256]={0}; // = new double[256]; // 灰度直方图  
    double N = width * height; // 总像素数  

	// 计算直方图 
    for(int i=0;i<height;i++) { 
        for(int j=0;j<width;j++) {  
            double temp = data[i*step + j * 3] * 0.114 + data[i*step + j * 3+1] * 0.587 + data[i*step + j * 3+2] * 0.299;  
            temp = temp < 0 ? 0 : temp;  
            temp = temp > 255 ? 255 : temp;  
            Histogram[(int)temp]++;  
        }   
    }  
    // 计算阈值  
    // 在[0, 255] 中寻找一个阈值 i 使 g = w0*w1*(u0-u1)*(u0-u1) 值最大
    for (int i = 0;i<256;i++) {  
        gSum0 = 0;  
        gSum1 = 0;  
        N0 += Histogram[i];           
        N1 = N-N0;  
        if(0==N1)break; // 当出现前景无像素点时，跳出循环  
        w0 = N0/N;  
        w1 = 1-w0;  

		// 计算第一类灰度总值：[0, i]属于背景
        for (int j = 0;j<=i;j++) {  
            gSum0 += j*Histogram[j];  
        }  
        u0 = gSum0 / N0;  

		// 计算第二类灰度总值：[i+1, 255]属于目标
        for(int k = i+1;k<256;k++) {  
            gSum1 += k*Histogram[k];  
        }  
        u1 = gSum1/N1;  

        // 计算总平均灰度 u = w0*u0 + w1*u1; 
        g = w0*w1*(u0-u1)*(u0-u1);  
        if (tempg < g) {  
            tempg = g;  
            T = i;  
        }  
    }  
    return T;   
}  

// 迭代法求自适应阈值
// 七种阈值常见分割代码：http://blog.csdn.net/xw20084898/article/details/17564957 
// 迭代阈值法：http://blog.csdn.net/a361251388leaning/article/details/50198351
int IterationGetThreshold(Mat image) {
	int height = image.rows;
	int width = image.cols;
	int step = image.step;
	uchar *data =  uchar* data  = (uchar*)img.data;  // Mat对象中的一个指针，指向内存中存放矩阵数据的一块内存 (uchar* data) 
	double Histogram[256]={0}; // = new double[256]; // 灰度直方图  

	// 计算直方图
	for(int i = 0; i < height; ++i) {
		for(int j = 0; j< width; ++j) {
			double temp = data[i*step + j * 3] * 0.114 + data[i*step + j * 3+1] * 0.587 + data[i*step + j * 3+2] * 0.299; 
			temp = temp < 0 ? 0 : temp;  
            temp = temp > 255 ? 255 : temp; 
            Histogram[(int)temp]++；
		}
	}

	// 求取图像的平均灰度值作为图像的初始阈值
	int threshold = 0;
	for(int i = 0; i < 256; ++i) {
		threshold += Histogram[i] * i;
	}
	threshold /= height * width;

	int newThreashold = 0;
	double sum0 = 0;
	double sum1 = 0;
	double N0 = 0;
	double N1 = 0;

	while(threshold != newThreashold) {
		for(int i = 0; i < threshold; ++i) {
			sum0 += Histogram[i] * i;
			N0 += Histogram[i];
		}

		for(int j=threshold; j<256; ++j) {
			sum1 += Histogram[j] * j;
			N1 += Histogram[j];
		}

		int newThreashold = (sum0 / N0 + sum1 / N1) / 2;
	}

}

// 对图像进行：灰度 --> 二值化 --> 滤波
Mat preProcess(const char * image) {
	// imread imshow http://blog.csdn.net/poem_qianmo/article/details/20537737
    // IMREAD_UNCHANGED  = -1, // 8bit, color or not
    // IMREAD_GRAYSCALE  = 0,  // 8bit, gray
    // IMREAD_COLOR      = 1,  // ?, color
    // IMREAD_ANYDEPTH   = 2,  // any depth, ?
    // IMREAD_ANYCOLOR   = 4,  // ?, any color
    // IMREAD_LOAD_GDAL  = 8   // Use gdal driver
	Mat srcImage = imread(image);
	if ( !srcImage.data ) {
		return Mat();
	}
	// imshow("origin image", srcImage);

	Mat grayImage;
	Mat dstImage;

	// 创建与 srcImage 同类型和大小的矩阵
	dstImage.create(srcImage.size(), srcImage.type());
	// 将原始图像转化为灰度图像	 opecv3入门教程 P115
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// imshow("grayImage", grayImage);
	// threshold http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html
	int thresholdVal = getAdaptiveThreshold(grayImage);
	// std::cout<<thresholdVal<<std::endl;
	// threshold(grayImage, dstImage, thresholdVal, 255, 0);
	threshold(grayImage, dstImage, thresholdVal + 23, 255, 0);

	// threshold(grayImage, dstImage, 65, 255, 0);
	// 65（阈值大小） 0（阈值类型） 由 binaryZation 函数测试得来

	// 自适应阈值  // 经测试此函数不适用
	// adaptiveThreshold(grayImage, dstImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY, 3, 5);


	// imshow("binaryImage", dstImage);

	// 进行图像模糊以降噪，哪种滤波函数最适用，需经多种测试
	blur(dstImage, dstImage, Size(4,4));
	// imshow("blurImage", dstImage);

	// GaussianBlur(dstImage, dstImage, Size(3, 3), 0, 0);
	// imshow("GaussianBlurImage", dstImage);

	// boxFilter(dstImage, dstImage, -1, Size(4, 4));
	// imshow("boxFilterImage", dstImage);
	// 边缘检测
	// Canny(dstImage, dstImage, 3,9,3);

	return dstImage;
}

Mat getROI(Mat image, const char * srcImage) {
	// 截取图像清晰的一部分, 圆心所在位置一定要截取
    Mat imageROI = image(Rect(0, 180, image.cols * 0.5 - 3, image.rows * 0.5 - 12 - 180));

	//Point center = getCircleCenter(srcImage);
	//std::cout<<center<<std::endl;
	//std::cout<<image.rows<<" "<<image.cols<<std::endl;
	//Mat imageROI = image(Rect(0, 180, center.x - 4, center.y - 180));

	// Mat imageROI = image(Rect(0, 180, image.cols * 0.5 - 3, image.rows * 0.5 - 29 - 180));
	// std::cout<<imageROI.rows<<" "<<imageROI.cols<<std::endl;


	// 断点处是否可以通过膨胀操作连接：http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
	imageROI.at<uchar>(74, 90) = 255;
	imageROI.at<uchar>(75, 90) = 255;	
	imageROI.at<uchar>(76, 90) = 255;
	imageROI.at<uchar>(77, 90) = 255;
	imageROI.at<uchar>(78, 90) = 255;
	imageROI.at<uchar>(79, 90) = 255;
	imageROI.at<uchar>(80, 90) = 255;
	imageROI.at<uchar>(81, 90) = 255;
	imageROI.at<uchar>(82, 90) = 255;
	
	return imageROI;
}

// 轮廓查找和填充  http://www.tuicool.com/articles/IJBrUf
// findContours函数 http://blog.csdn.net/u012566751/article/details/54017882
// 同心圆检测 http://blog.csdn.net/u011853479/article/details/50405793
// Mat::zeros http://blog.csdn.net/giantchen547792075/article/details/7169255

int getCounterNum(Mat image) {
	std::vector<std::vector<Point> > contours; // 存放检测到的轮廓
	std::vector<Vec4i> hierarchy; // 每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]
	std::vector<std::vector<Point>>::const_iterator itContours;
	//srand((int)time(0));
	Mat src = image;
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

	Scalar color = Scalar(rand()&255, rand()&255, rand()&255);
	src = src > 100; // 作用是什么？
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE ); // 检测轮廓 
	itContours = contours.begin();
	int i = 0;
	for(; itContours != contours.end(); ++itContours) {
		color = Scalar(rand()&255, rand()&255, rand()&255);
		drawContours(dst, contours, i, color, FILLED); // 绘制轮廓
		i++;
	}
	imshow("contourImage", dst);
	return contours.size() - 1;
}
	

float getResult(const char *image) {
	Mat dstImage = preProcess(image);
	Mat imageROI = getROI(dstImage, image);
	imshow("imageROI", imageROI);
	
	float num2 = getPrecision(image);
	//std::cout<<"num2 is"<<num2<<std::endl;
    //float num2 = 0;
	int num1 = getCounterNum(imageROI);

	return num1 + num2;
}

/////////////////////////////////////////////////////////////////////////////////////
// 备注：
//		main函数通过调用 binaryZation 函数
//      确定 二值化函数（preProcess 中的 threshold函数） 所用到的合适的 1. 阈值类型 2. 阈值大小   

// 全局变量的定义  http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html
int threshold_value = 0;
int threshold_type = 3 ;
Mat srcImage, grayImage, dstImage;
char* window_name = "二值化参数测试器";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

// 自定义的阈值函数
void Threshold_Demo(int, void*) {
  /* 0: 二进制阈值
     1: 反二进制阈值
     2: 截断阈值
     3: 0阈值
     4: 反0阈值
   */
  int const max_BINARY_value = 255;
  threshold(grayImage, dstImage, threshold_value, max_BINARY_value, threshold_type);
  imshow(window_name, dstImage);
}

// 获取图像中某一点的像素值： http://blog.sina.com.cn/s/blog_6a2844750101at8y.html
// opecv中坐标系的认识： http://blog.csdn.net/liulina603/article/details/9376229
// 图像的遍历： http://blog.csdn.net/daoqinglin/article/details/23628125
float getPrecision(const char *image) {
	Mat srcImage = imread(image);
	if ( !srcImage.data ) {
		return 0;
	}
	// imshow("origin image", srcImage);

	Mat grayImage;
	Mat dstImage;

	// 创建与 srcImage 同类型和大小的矩阵
	dstImage.create(srcImage.rows, srcImage.cols, CV_8UC1);
	grayImage.create(srcImage.rows, srcImage.cols, CV_8UC1);
	// std::cout<<dstImage.rows<<" "<<dstImage.cols<<" "<<dstImage.channels()<<std::endl;

	// 将原始图像转化为灰度图像	 opecv3入门教程 P115
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// blur(grayImage, grayImage, Size(4,4));
	// imshow("grayImage", grayImage);
	// threshold http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html
	int thresholdVal = getAdaptiveThreshold(grayImage);
	// std::cout<<thresholdVal<<std::endl;
	// threshold(grayImage, dstImage, thresholdVal, 255, 0);
	threshold(grayImage, dstImage, thresholdVal + 23, 255, 0);
	imshow("threImage", dstImage);
	
	Point center = getCircleCenter(image);
	// std::cout<<"center is "<<center<<std::endl;

	int row = dstImage.rows;
	int col = dstImage.cols;
	int channels = dstImage.channels();
	////std::cout<<dstImage<<std::endl;
	//std::cout<<"row: "<<row<<" col: "<<col<<" channels: "<<channels<<std::endl;

	//for(int i=0; i<row; ++i) {
	//	uchar *data = dstImage.ptr<uchar>(i);// 获取第i行的首地址
	//	for(int j=0; j<col * channels; ++j) {
	//		std::cout<<i<<" "<<j<<" "<<(int)data[j]<<std::endl;
	//		waitKey(0);
	//	}
	//}


	//uchar *data = dstImage.ptr<uchar>(center.y);// 获取第i行的首地址
	//for(int j=0; j<col * channels; ++j) {
	//	std::cout<<" "<<" "<<(int)data[j];
	//}

   
	//std::cout<<"row is: "<<row<<"col is: "<<col<<std::endl;
	//for(int k=0; k<col; k+=2) {
	//	std::cout<<center.y << " " << k << " " <<dstImage.at<Vec2b>(center.y, k)<<std::endl;
	//}
	//for(int j=0; j<row; ++j) {
	//	for(int i=0; i<col; ++i) {
	//		std::cout<<grayImage.at<uchar>(i, j)<<std::endl;
	//	}
	//}

	// 
	uchar *data = dstImage.ptr<uchar>(center.y - 4);

	int j = 0;
	// 移动到最外层轮廓的边缘
	while(j < center.x && int(data[j]) == 0) {
		j++;	
	}

	// 获取最外层轮廓像素点的个数
	float num1 = 0;
	while(j < center.x && int(data[j]) != 0) {
		num1++;
		j++;
	}

	// 移动到倒数第二层轮廓的边缘
	while(j < center.x && int(data[j]) == 0) {
		j++;	
	}

	// 获取倒数第二层轮廓像素点的个数
	float num2 = 0;
	while(j < center.x && int(data[j]) != 0) {
		num2++;
		j++;
	}
	return num1/num2;



	std::cout<<"num1: "<<num1<<" "<<" num2 "<<num2<<std::endl;
	//waitKey(0);
	return num1/num2 < 1 ? num1/num2 : 1;
}

// 二值化函数
void binaryZation(const char * image) {
  // 读取一副图片，不改变图片本身的颜色类型
  srcImage = imread(image);
  imshow("origin image", srcImage);

  // 将图片转换成灰度图片
  cvtColor(srcImage, grayImage, COLOR_RGB2GRAY );

  // 创建一个窗口显示图片
  namedWindow( window_name, WINDOW_AUTOSIZE);

  // 创建滑动条来控制阈值 opencv3.0 P90
  // createTrackbar(滑动条的名称，所依附窗口的名称，滑块的位置，滑块可以达到的最大的值， 回调函数)
  int const max_type = 4;
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  int const max_value = 255;
  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );

  // 初始化自定义的阈值函数：回调函数， 滑块位置改变时被调用
  Threshold_Demo( 0, 0 );

  // 等待用户按键。如果是ESC健则退出等待过程。
  while(true) {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 ) { 
			break;
		}
   }
}

/////////////////////////////////////////////////////////////////////////////////////
// 备注：
//		测试：利用霍夫圆检测的可行性
//      霍夫圆变换：http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
//      HoughCircles 函数详解： http://blog.csdn.net/poem_qianmo/article/details/26977557
//      实现原理：http://blog.csdn.net/zhazhiqiang/article/details/51097439
Point getCircleCenter(const char *image) {
	Mat srcImage = imread(image);
	// cout<<srcImage.size()<<endl;
	// imshow("srcImage", srcImage);

	Mat grayImage;
	// 转化为灰度图
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// imshow("grayImage", grayImage);

	//threshold(grayImage, grayImage, 65, 255, 0);
	//imshow("binaryImage", grayImage);

	// 进行图像平滑处理
	GaussianBlur(grayImage, grayImage, Size(9, 9), 2, 2);

	//// 进行霍夫圆转换
	std::vector<Vec3f> circles;
	HoughCircles(grayImage, circles, HOUGH_GRADIENT, 1, 200, 100, 100, 0, 200);

	//Mat dstImage(srcImage.size(), srcImage.type());
	Point center;
	for(size_t i = 0; i< circles.size(); i++) {
		center = Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
		// int radius = cvRound(circles[i][2]);
		// std::cout<<"圆心"<<center<<"半径"<<radius<<std::endl;
		// 绘制圆心
		// Scalar color = Scalar(rand()&255, rand()&255, rand()&255);
		// circle(srcImage, center, 3, color, -1, 8, 0);
		// 绘制圆轮廓
		// circle(srcImage, center, radius, color, 3, 8, 0);
	}
	// imshow("dstImage", srcImage);
	return center;
}
	

	
/////////////////////////////////////////////////////////////////////////////////////
// 备注：
//		测试三种滤波函数
//  

int box_filter_value = 0;
int mean_blur_value = 0;
int gaussian_blur_value = 0;

Mat filter_srcImage;
Mat dst_image_1;
Mat dst_image_2;
Mat dst_image_3;


static void on_box_filter(int, void *) {
	boxFilter(filter_srcImage, dst_image_1, -1, Size(box_filter_value+1, box_filter_value+1));
	imshow("BoxFilter", dst_image_1);
}

static void on_mean_blur_filter(int, void *) {
	blur(filter_srcImage, dst_image_2, Size(mean_blur_value+1, mean_blur_value+1),Point(-1, -1));
	imshow("MeanBlurFilter", dst_image_2);
}

static void on_gaussian_blur_filter(int, void *) {
	GaussianBlur(filter_srcImage, dst_image_3, Size(gaussian_blur_value+1, gaussian_blur_value+1), 0, 0);
	imshow("GaussianBlurFilter", dst_image_3);
}


void filter(const char *image) {
	filter_srcImage = imread(image, 1);
	imshow("filter_srcImage", filter_srcImage);

    dst_image_1 = filter_srcImage.clone();
	dst_image_2 = filter_srcImage.clone();
	dst_image_3 = filter_srcImage.clone();

	//
	namedWindow("BoxFilter", 1);
	createTrackbar("BoxFilter", "BoxFilter", &box_filter_value, 40, on_box_filter);
	on_box_filter(int(), NULL);
	
	namedWindow("MeanBlurFilter", 1);
	createTrackbar("MeanBlurFilter", "MeanBlurFilter", &mean_blur_value, 40, on_mean_blur_filter);
	on_mean_blur_filter(int(), NULL);
	
	namedWindow("GaussianBlurFilter", 1);
	createTrackbar("GaussianBlurFilter", "GaussianBlurFilter", &gaussian_blur_value, 40, on_gaussian_blur_filter);
	on_gaussian_blur_filter(int(), NULL);

	while((char)(waitKey(1)) != 'q') {}
}
