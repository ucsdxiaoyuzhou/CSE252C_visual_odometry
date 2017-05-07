#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat non_max_sup(int n, int margin, unsigned char tau, const cv::Mat &image)  {
    std::cout << int(tau) << std::endl;
	int width = image.cols, height = image.rows;
	cv::Mat image_nms = cv::Mat::zeros(height, width, CV_8U);
	for (int i = n + margin; i < width - n - margin; i = i + n + 1) {
		for (int j = n + margin; j < height - n - margin; j = j + n + 1) {
			int max_i = i, max_j = j;
			unsigned char max_val = image.at<unsigned char>(j, i);
			unsigned char curr_val;
			for (int i_1 = i; i_1 < i + n + 1; i_1++) {
                for (int j_1 = j; j_1 < j + n + 1; j_1++) {
                    curr_val = image.at<unsigned char>(j_1, i_1);
                    if (curr_val > max_val) {
                        max_i = i_1;
                        max_j = j_1;
                        max_val = curr_val;
                    }
                }
            }
            bool failed = false;
            for (int i_2 = max_i - n; i_2 < std::min(max_i + n, width - margin) + 1; i_2 ++) {
                for (int j_2 = max_j - n; j_2 < std::min(max_j + n, height - margin) + 1; j_2 ++) {
                    curr_val = image.at<unsigned char>(j_2, i_2);
                    if (curr_val > max_val && 
                    	(i_2 < i || i_2 > i + n || j_2 < j || j_2 > j + n)) {
                        failed = true;
                        break;
                    }
                }
                if (failed)
                    break;
            }

            if (max_val >= tau && !failed)
                image_nms.at<unsigned char>(max_j, max_i) = image.at<unsigned char>(max_j, max_i);
		}
	}
	return image_nms;
}

int main(int argc, char** argv) {
	cv::Mat image, image_nms;
	image = cv::imread(argv[1], 0);
	cv::imshow("Corner Image", image);
	image_nms = non_max_sup(3, 5, 2, image);
	cv::imshow("NMS Corner Image", image_nms);
	cv::waitKey(0);
}