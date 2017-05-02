#include"function.h"

int ImageBinarization(Mat src) {   /*�ԻҶ�ͼ���ֵ��������Ӧ����threshold*/
	int i, j, width, height, step, chanel, threshold;
	/*size��ͼ��ߴ磬svg�ǻҶ�ֱ��ͼ��ֵ��va�Ƿ���*/
	float size, avg, va, maxVa, p, a, s;
	unsigned char *dataSrc;
	float histogram[256];

	width = src.rows;
	height = src.cols;
	dataSrc = (unsigned char *)src.data; //
	step = src.step / sizeof(char);
	chanel = src.channels();
	/*����ֱ��ͼ����һ��histogram*/
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
	/*����Ҷ�ֱ��ͼ��ֵ�ͷ���*/
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
	/*���ü�Ȩ��󷽲�������*/
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
	///*��ֵ��*/
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
	//���غ�����ֵ
	return threshold;
}

// �����䷽� http://blog.csdn.net/guoyk1990/article/details/7606032
// �Ҷ�ֱ��ͼ http://blog.csdn.net/qq_20823641/article/details/51932798
// ��rows��cols��data��step��channels����� https://www.douban.com/note/265479171/
// OpenCV2:Mat����type��depth��step http://www.tuicool.com/articles/eUbuYn
// ʵ��ԭ��http://blog.csdn.net/a153375250/article/details/50970104
// ������ֵ�����ָ���룺http://blog.csdn.net/xw20084898/article/details/17564957

int getAdaptiveThreshold(Mat img) {  
    int T = 0;             // ��ֵ  
    int height = img.rows; // rows �� ���� �൱�� height ��Ӧ .y
    int width  = img.cols; // cols �� ���� �൱�� width ��Ӧ .x
    int step = img.step;  
    int channels  = img.channels();   // ͨ���������е�ÿһ������Ԫ��ӵ�е�ֵ�ĸ���
    uchar* data  = (uchar*)img.data;  // Mat�����е�һ��ָ�룬ָ���ڴ��д�ž������ݵ�һ���ڴ� (uchar* data) 
    double gSum0;      // ��һ��Ҷ���ֵ  
    double gSum1;      // �ڶ���Ҷ���ֵ  
    double N0 = 0;     // ǰ��������  
    double N1 = 0;     // ����������  
    double u0 = 0;     // ǰ������ƽ���Ҷ�  
    double u1 = 0;     // ��������ƽ���Ҷ�  
    double w0 = 0;     // ǰ�����ص���ռ����ͼ��ı���Ϊ��0  
    double w1 = 0;     // �������ص���ռ����ͼ��ı���Ϊ��1  
    double u = 0;      // ��ƽ���Ҷ�  
    double tempg = -1; // ��ʱ��䷽��  
    double g = -1;     // ��䷽��  
    double Histogram[256]={0}; // = new double[256]; // �Ҷ�ֱ��ͼ  
    double N = width * height; // ��������  

	// ����ֱ��ͼ 
    for(int i=0;i<height;i++) { 
        for(int j=0;j<width;j++) {  
            double temp = data[i*step + j * 3] * 0.114 + data[i*step + j * 3+1] * 0.587 + data[i*step + j * 3+2] * 0.299;  
            temp = temp < 0 ? 0 : temp;  
            temp = temp > 255 ? 255 : temp;  
            Histogram[(int)temp]++;  
        }   
    }  
    // ������ֵ  
    // ��[0, 255] ��Ѱ��һ����ֵ i ʹ g = w0*w1*(u0-u1)*(u0-u1) ֵ���
    for (int i = 0;i<256;i++) {  
        gSum0 = 0;  
        gSum1 = 0;  
        N0 += Histogram[i];           
        N1 = N-N0;  
        if(0==N1)break; // ������ǰ�������ص�ʱ������ѭ��  
        w0 = N0/N;  
        w1 = 1-w0;  

		// �����һ��Ҷ���ֵ��[0, i]���ڱ���
        for (int j = 0;j<=i;j++) {  
            gSum0 += j*Histogram[j];  
        }  
        u0 = gSum0 / N0;  

		// ����ڶ���Ҷ���ֵ��[i+1, 255]����Ŀ��
        for(int k = i+1;k<256;k++) {  
            gSum1 += k*Histogram[k];  
        }  
        u1 = gSum1/N1;  

        // ������ƽ���Ҷ� u = w0*u0 + w1*u1; 
        g = w0*w1*(u0-u1)*(u0-u1);  
        if (tempg < g) {  
            tempg = g;  
            T = i;  
        }  
    }  
    return T;   
}  

// ������������Ӧ��ֵ
// ������ֵ�����ָ���룺http://blog.csdn.net/xw20084898/article/details/17564957 
// ������ֵ����http://blog.csdn.net/a361251388leaning/article/details/50198351
int IterationGetThreshold(Mat image) {
	int height = image.rows;
	int width = image.cols;
	int step = image.step;
	uchar *data =  uchar* data  = (uchar*)img.data;  // Mat�����е�һ��ָ�룬ָ���ڴ��д�ž������ݵ�һ���ڴ� (uchar* data) 
	double Histogram[256]={0}; // = new double[256]; // �Ҷ�ֱ��ͼ  

	// ����ֱ��ͼ
	for(int i = 0; i < height; ++i) {
		for(int j = 0; j< width; ++j) {
			double temp = data[i*step + j * 3] * 0.114 + data[i*step + j * 3+1] * 0.587 + data[i*step + j * 3+2] * 0.299; 
			temp = temp < 0 ? 0 : temp;  
            temp = temp > 255 ? 255 : temp; 
            Histogram[(int)temp]++��
		}
	}

	// ��ȡͼ���ƽ���Ҷ�ֵ��Ϊͼ��ĳ�ʼ��ֵ
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

// ��ͼ����У��Ҷ� --> ��ֵ�� --> �˲�
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

	// ������ srcImage ͬ���ͺʹ�С�ľ���
	dstImage.create(srcImage.size(), srcImage.type());
	// ��ԭʼͼ��ת��Ϊ�Ҷ�ͼ��	 opecv3���Ž̳� P115
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// imshow("grayImage", grayImage);
	// threshold http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html
	int thresholdVal = getAdaptiveThreshold(grayImage);
	// std::cout<<thresholdVal<<std::endl;
	// threshold(grayImage, dstImage, thresholdVal, 255, 0);
	threshold(grayImage, dstImage, thresholdVal + 23, 255, 0);

	// threshold(grayImage, dstImage, 65, 255, 0);
	// 65����ֵ��С�� 0����ֵ���ͣ� �� binaryZation �������Ե���

	// ����Ӧ��ֵ  // �����Դ˺���������
	// adaptiveThreshold(grayImage, dstImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY, 3, 5);


	// imshow("binaryImage", dstImage);

	// ����ͼ��ģ���Խ��룬�����˲����������ã��辭���ֲ���
	blur(dstImage, dstImage, Size(4,4));
	// imshow("blurImage", dstImage);

	// GaussianBlur(dstImage, dstImage, Size(3, 3), 0, 0);
	// imshow("GaussianBlurImage", dstImage);

	// boxFilter(dstImage, dstImage, -1, Size(4, 4));
	// imshow("boxFilterImage", dstImage);
	// ��Ե���
	// Canny(dstImage, dstImage, 3,9,3);

	return dstImage;
}

Mat getROI(Mat image, const char * srcImage) {
	// ��ȡͼ��������һ����, Բ������λ��һ��Ҫ��ȡ
    Mat imageROI = image(Rect(0, 180, image.cols * 0.5 - 3, image.rows * 0.5 - 12 - 180));

	//Point center = getCircleCenter(srcImage);
	//std::cout<<center<<std::endl;
	//std::cout<<image.rows<<" "<<image.cols<<std::endl;
	//Mat imageROI = image(Rect(0, 180, center.x - 4, center.y - 180));

	// Mat imageROI = image(Rect(0, 180, image.cols * 0.5 - 3, image.rows * 0.5 - 29 - 180));
	// std::cout<<imageROI.rows<<" "<<imageROI.cols<<std::endl;


	// �ϵ㴦�Ƿ����ͨ�����Ͳ������ӣ�http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
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

// �������Һ����  http://www.tuicool.com/articles/IJBrUf
// findContours���� http://blog.csdn.net/u012566751/article/details/54017882
// ͬ��Բ��� http://blog.csdn.net/u011853479/article/details/50405793
// Mat::zeros http://blog.csdn.net/giantchen547792075/article/details/7169255

int getCounterNum(Mat image) {
	std::vector<std::vector<Point> > contours; // ��ż�⵽������
	std::vector<Vec4i> hierarchy; // ÿ������contours[i]��Ӧ4��hierarchyԪ��hierarchy[i][0] ~hierarchy[i][3]
	std::vector<std::vector<Point>>::const_iterator itContours;
	//srand((int)time(0));
	Mat src = image;
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

	Scalar color = Scalar(rand()&255, rand()&255, rand()&255);
	src = src > 100; // ������ʲô��
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE ); // ������� 
	itContours = contours.begin();
	int i = 0;
	for(; itContours != contours.end(); ++itContours) {
		color = Scalar(rand()&255, rand()&255, rand()&255);
		drawContours(dst, contours, i, color, FILLED); // ��������
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
// ��ע��
//		main����ͨ������ binaryZation ����
//      ȷ�� ��ֵ��������preProcess �е� threshold������ ���õ��ĺ��ʵ� 1. ��ֵ���� 2. ��ֵ��С   

// ȫ�ֱ����Ķ���  http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html
int threshold_value = 0;
int threshold_type = 3 ;
Mat srcImage, grayImage, dstImage;
char* window_name = "��ֵ������������";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

// �Զ������ֵ����
void Threshold_Demo(int, void*) {
  /* 0: ��������ֵ
     1: ����������ֵ
     2: �ض���ֵ
     3: 0��ֵ
     4: ��0��ֵ
   */
  int const max_BINARY_value = 255;
  threshold(grayImage, dstImage, threshold_value, max_BINARY_value, threshold_type);
  imshow(window_name, dstImage);
}

// ��ȡͼ����ĳһ�������ֵ�� http://blog.sina.com.cn/s/blog_6a2844750101at8y.html
// opecv������ϵ����ʶ�� http://blog.csdn.net/liulina603/article/details/9376229
// ͼ��ı����� http://blog.csdn.net/daoqinglin/article/details/23628125
float getPrecision(const char *image) {
	Mat srcImage = imread(image);
	if ( !srcImage.data ) {
		return 0;
	}
	// imshow("origin image", srcImage);

	Mat grayImage;
	Mat dstImage;

	// ������ srcImage ͬ���ͺʹ�С�ľ���
	dstImage.create(srcImage.rows, srcImage.cols, CV_8UC1);
	grayImage.create(srcImage.rows, srcImage.cols, CV_8UC1);
	// std::cout<<dstImage.rows<<" "<<dstImage.cols<<" "<<dstImage.channels()<<std::endl;

	// ��ԭʼͼ��ת��Ϊ�Ҷ�ͼ��	 opecv3���Ž̳� P115
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
	//	uchar *data = dstImage.ptr<uchar>(i);// ��ȡ��i�е��׵�ַ
	//	for(int j=0; j<col * channels; ++j) {
	//		std::cout<<i<<" "<<j<<" "<<(int)data[j]<<std::endl;
	//		waitKey(0);
	//	}
	//}


	//uchar *data = dstImage.ptr<uchar>(center.y);// ��ȡ��i�е��׵�ַ
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
	// �ƶ�������������ı�Ե
	while(j < center.x && int(data[j]) == 0) {
		j++;	
	}

	// ��ȡ������������ص�ĸ���
	float num1 = 0;
	while(j < center.x && int(data[j]) != 0) {
		num1++;
		j++;
	}

	// �ƶ��������ڶ��������ı�Ե
	while(j < center.x && int(data[j]) == 0) {
		j++;	
	}

	// ��ȡ�����ڶ����������ص�ĸ���
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

// ��ֵ������
void binaryZation(const char * image) {
  // ��ȡһ��ͼƬ�����ı�ͼƬ�������ɫ����
  srcImage = imread(image);
  imshow("origin image", srcImage);

  // ��ͼƬת���ɻҶ�ͼƬ
  cvtColor(srcImage, grayImage, COLOR_RGB2GRAY );

  // ����һ��������ʾͼƬ
  namedWindow( window_name, WINDOW_AUTOSIZE);

  // ������������������ֵ opencv3.0 P90
  // createTrackbar(�����������ƣ����������ڵ����ƣ������λ�ã�������Դﵽ������ֵ�� �ص�����)
  int const max_type = 4;
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  int const max_value = 255;
  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );

  // ��ʼ���Զ������ֵ�������ص������� ����λ�øı�ʱ������
  Threshold_Demo( 0, 0 );

  // �ȴ��û������������ESC�����˳��ȴ����̡�
  while(true) {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 ) { 
			break;
		}
   }
}

/////////////////////////////////////////////////////////////////////////////////////
// ��ע��
//		���ԣ����û���Բ���Ŀ�����
//      ����Բ�任��http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
//      HoughCircles ������⣺ http://blog.csdn.net/poem_qianmo/article/details/26977557
//      ʵ��ԭ��http://blog.csdn.net/zhazhiqiang/article/details/51097439
Point getCircleCenter(const char *image) {
	Mat srcImage = imread(image);
	// cout<<srcImage.size()<<endl;
	// imshow("srcImage", srcImage);

	Mat grayImage;
	// ת��Ϊ�Ҷ�ͼ
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// imshow("grayImage", grayImage);

	//threshold(grayImage, grayImage, 65, 255, 0);
	//imshow("binaryImage", grayImage);

	// ����ͼ��ƽ������
	GaussianBlur(grayImage, grayImage, Size(9, 9), 2, 2);

	//// ���л���Բת��
	std::vector<Vec3f> circles;
	HoughCircles(grayImage, circles, HOUGH_GRADIENT, 1, 200, 100, 100, 0, 200);

	//Mat dstImage(srcImage.size(), srcImage.type());
	Point center;
	for(size_t i = 0; i< circles.size(); i++) {
		center = Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
		// int radius = cvRound(circles[i][2]);
		// std::cout<<"Բ��"<<center<<"�뾶"<<radius<<std::endl;
		// ����Բ��
		// Scalar color = Scalar(rand()&255, rand()&255, rand()&255);
		// circle(srcImage, center, 3, color, -1, 8, 0);
		// ����Բ����
		// circle(srcImage, center, radius, color, 3, 8, 0);
	}
	// imshow("dstImage", srcImage);
	return center;
}
	

	
/////////////////////////////////////////////////////////////////////////////////////
// ��ע��
//		���������˲�����
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
