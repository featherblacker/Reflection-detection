
#include<stdlib.h>
#include<stdio.h>
#include<opencv2\opencv.hpp>
#include<io.h>
#include<string>

using namespace std;
using namespace cv;

class readfile {
public:
	readfile(String file) {
		bool ans = true;
		String window_name = "Demo";
		Mat img = imread(file);
		resize(img, img, Size(300, 600));
		int height = img.rows;
		int width = img.cols;

		medianBlur(img, img, 7);

		Mat absY;
		Mat absX;
		Sobel(img, absY, CV_16S, 0, 2);
		convertScaleAbs(absY, absY);
		Sobel(img, absX, CV_16S, 2, 0);
		convertScaleAbs(absX, absX);


		cvtColor(absY, absY, COLOR_BGR2GRAY);
		cvtColor(absX, absX, COLOR_BGR2GRAY);

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (absX.at<uchar>(i, j) > 40) {
					absY.at<uchar>(i, j) = 0;
				}
			}
		}

		double ret;
		threshold(absY, absY, 90, 255, THRESH_BINARY);
		GaussianBlur(absY, absY, Size(3, 3), 0);

		/*int a = 3;
		int b = 11;
		Mat kernal[a][b];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 11; j++) {
				kernal[i][j] = 0;
			}
		}
		morphologyEx(absY, absY, MORPH_CLOSE,kernal);*/

		medianBlur(absY, absY, 5);
		threshold(absY, absY, 30, 255, THRESH_BINARY);

		/*Mat contours, hierarchy;*/
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(absY, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		vector<vector<Point> > a;
		int area;
		vector<Rect> boundRect(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			area = contourArea(contours[i]);

			boundRect[i] = boundingRect(Mat(contours[i]));
			//
			if (area < 30) {
				for (int m = 0; m < boundRect[i].height; m++) {
					for (int n = 0; n < boundRect[i].width; n++) {
						absY.at<uchar>(boundRect[i].y + m, boundRect[i].x + n) = 0;
					}
				}
			}
			else {
				a.push_back(contours[i]);
			}
		}

		area = 0;
		vector<vector<Point> > b;
		vector<Rect> boundRect2(a.size());

		for (int i = 0; i < a.size(); i++) {
			area += contourArea(a[i]);
		}
		if (a.size() != 0) {
			area /= a.size();
			for (int j = 0; j < a.size(); j++) {
				boundRect2[j] = boundingRect(Mat(a[j]));
				int l = contourArea(a[j]);
				if (l < area) {
					for (int m = 0; m < boundRect2[j].height; m++) {
						for (int n = 0; n < boundRect2[j].width; n++) {
							absY.at<uchar>(boundRect2[j].y + m, boundRect2[j].x + n) = 0;
						}
					}
				}
				else {
					b.push_back(a[j]);
				}
			}
		}

		int s = 0;
		double k;
		vector<vector<Point> > p;
		vector<Rect> boundRect3(a.size());
		for (int i = 0; i < b.size(); i++) {
			boundRect3[i] = boundingRect(Mat(b[i]));
			k = boundRect3[i].height / boundRect3[i].width;
			if ((boundRect3[i].x > 0.05 * width) && (boundRect3[i].x + boundRect3[i].width < 0.95 * width) && (boundRect3[i].y > 0.2 * height) && (boundRect3[i].y + boundRect3[i].height < 0.8 * height)) {
				if (k < 0.5) {
					p.push_back(b[i]);
					s += boundRect3[i].width;
				}
				else {
					for (int m = 0; m < boundRect3[i].height; m++) {
						for (int n = 0; n < boundRect3[i].width; n++) {
							absY.at<uchar>(boundRect3[i].y + m, boundRect3[i].x + n) = 0;
						}
					}
				}
			}
			else {
				for (int m = 0; m < boundRect3[i].height; m++) {
					for (int n = 0; n < boundRect3[i].width; n++) {
						absY.at<uchar>(boundRect3[i].y + m, boundRect3[i].x + n) = 0;
					}
				}
			}
		}

		vector<vector<Point> > c;
		vector<Rect> boundRect4(a.size());


		if (p.size() != 0) {
			s /= p.size();
			for (int i = 0; i < p.size(); i++) {
				boundRect4[i] = boundingRect(Mat(p[i]));
				if (0.5 * s < boundRect4[i].width < 2 * s) {
					c.push_back(p[i]);
					rectangle(img, boundRect4[i].tl(), boundRect4[i].br(), (255, 255, 0));
				}
			}
		}
		if (c.size() > 12) {
			ans = false;
		}

		Mat black = Mat::zeros(absY.size(), CV_8UC1);

		vector<Rect> boundRect5(c.size());

		for (int i = 0; i < c.size(); i++) {
			boundRect5[i] = boundingRect(Mat(c[i]));
			for (int m = 0; m < boundRect5[i].width / 2; m++) {
				//cout << boundRect5[i].y << " " << boundRect5[i].x << " " << boundRect5[i].height << " " << boundRect5[i].width << endl;
				black.at<uchar>(boundRect5[i].y, boundRect5[i].x + m) = 255;
			}
		}

		/*	imshow("black", black);
			imshow(window_name, absY)*/;

		bool sol = false;
		int sizeh = height / 4;
		for (int i = height / 5; i < 4 * height / 5 - sizeh; i += sizeh / 4) {
			for (int j = width / 5; j < 4 * width / 5; j++) {
				int changecolor = 0;
				for (int m = i; m < i + sizeh - 1; m++) {
					if (black.at<uchar>(m, j) != black.at<uchar>(m + 1, j)) {
						changecolor += 1;
					}
					if (changecolor >= 6) {
						sol = true;
					}if (changecolor > 16) {
						ans = false;
					}
				}
			}
		}
		ans = sol && ans;
	}
	bool show() {
		return ans;
	}
	private:
		bool ans = true;
};
void main() {
	vector<String> files;
	vector<String> images;
	String IMG_PATH = "E:\\new\\img6\\Yes\\*.jpg";
	glob(IMG_PATH, files);
	size_t count = files.size();
	for (int i = 0; i < count; i++) {
		readfile File(files[i]);
		File.show();
		waitKey(0);
	}
}