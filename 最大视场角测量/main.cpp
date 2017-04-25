#include"function.h"

void test1() {
	/*
		// ¶þÖµ»¯  --> ÂË²¨
		1. blur(dstImage, dstImage, Size(5,5));   
		2. blur(dstImage, dstImage, Size(4,4));  // more better
		3. GaussianBlur(dstImage, dstImage, Size(5, 5), 0, 0);
		4. GaussianBlur(dstImage, dstImage, Size(3, 3), 0, 0);
		5. boxFilter(dstImage, dstImage, -1, Size(5, 5));
		6. boxFilter(dstImage, dstImage, -1, Size(4, 4));
	*/

	//Mat dstImage = preProcess("20161128153556.JPG"); // good 32  32  33  33  33  32
    //Mat dstImage = preProcess("20161128153619.JPG"); // good 32  33  33  33  32  33
	//Mat dstImage = preProcess("zyh.JPG");            // good 32  32  33  33  33  32
	//Mat dstImage = preProcess("20161128153621.JPG"); // bad  32  32  32  32  32  32
	//Mat dstImage = preProcess("20161128153622.JPG"); // bad  31  32  32  32  31  32
	//Mat dstImage = preProcess("20161128153625.JPG"); // bad  33  33  32  32  33  33
	//Mat dstImage = preProcess("20161128153626.JPG"); // bad  31  32  32  32  31  32
	//Mat dstImage = preProcess("20161128153627.JPG"); // bad  32  32  31  33  32  32

	char *images[7] = { "20161128153556.JPG",
		                "20161128153619.JPG", 
					    "20161128153612.JPG",
					    "20161128153621.JPG",
					    "20161128153622.JPG",
					    "20161128153625.JPG",
					    "20161128153626.JPG",
						//"20161128153627.JPG" 
					   };

	for(int i=0; i<7; ++i) {
		std::cout<<getResult(images[i])<<std::endl;

		waitKey(0);
	}
}


void test2() {
	//binaryZation("zyh.JPG");
	binaryZation("2.JPG");
	waitKey(1000000000000);
}

void test3() {
	std::cout<<getCircleCenter("zyh.JPG")<<std::endl;
    //hough("c.jpg");
}

void test4() {
	filter("zyh.JPG");
}

void test5() {
	Mat originImage = imread("c.jpg");
	std::cout<<ImageBinarization(originImage)<<std::endl;
	/*imshow("originImage", originImage);

	Mat grayImage;
	Mat dstImage;
	cvtColor(originImage, grayImage, COLOR_BGR2GRAY);
	imshow("grayImage", grayImage);

	threshold(grayImage, dstImage, 222, 255, 0);
	imshow("binaryImage", dstImage);


    dstImage = imageFilling(dstImage);
	imshow("dst", dstImage);
	waitKey(0);*/
}

void test6() {
	//std::cout<<getPrecision("20161128153619.JPG")<<std::endl;
	//std::cout<<getPrecision("mohu.jpg")<<std::endl;
	//std::cout<<getPrecision("zyh.JPG")<<std::endl;
	//std::cout<<getPrecision("20161128153619.JPG")<<std::endl;
	std::cout<<getPrecision("test.JPG")<<std::endl;
}

void test7() {
	std::cout<<getResult("mohu.jpg")<<std::endl;
	waitKey(0);
}
int main(int argc, char **argv) {
	//test1();
	//test3();
	test6();
	return 0;
}



