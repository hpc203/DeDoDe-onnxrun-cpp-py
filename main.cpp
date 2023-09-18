#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/features2d.hpp>

//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class DeDoDeRunner_end2end
{
public:
	DeDoDeRunner_end2end(string model_path);
	void detect(Mat image_a, Mat image_b, vector<cv::KeyPoint>& points_A, vector<cv::KeyPoint>& points_B);
private:
	const int inpWidth = 256;
	const int inpHeight = 256;
	const float mean_[3] = { 0.485, 0.456, 0.406 };
	const float std_[3] = { 0.229, 0.224, 0.225 };
	vector<float> input_images;
	void preprocess(Mat image_a, Mat image_b);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "cv::KeyPoints detect and match");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

DeDoDeRunner_end2end::DeDoDeRunner_end2end(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
}

void DeDoDeRunner_end2end::preprocess(Mat image_a, Mat image_b)
{
	Mat dstimg;
	cvtColor(image_a, dstimg, COLOR_BGR2RGB);
	Size target_size = Size(this->inpWidth, this->inpHeight);
	resize(dstimg, dstimg, target_size, INTER_LINEAR);
	this->input_images.resize(2 * target_size.area() * 3);
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				this->input_images[c * target_size.area() + i * this->inpWidth + j] = (pix / 255.0 - this->mean_[c]) / this->std_[c];
			}
		}
	}

	cvtColor(image_b, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, target_size, INTER_LINEAR);
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				this->input_images[(3 + c) * target_size.area() + i * this->inpWidth + j] = (pix / 255.0 - this->mean_[c]) / this->std_[c];
			}
		}
	}
}


void DeDoDeRunner_end2end::detect(Mat image_a, Mat image_b, vector<cv::KeyPoint>& points_A, vector<cv::KeyPoint>& points_B)
{
	this->preprocess(image_a, image_b);
	array<int64_t, 4> input_shape_{ 2, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_images.data(), input_images.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	///Postprocessing
	const float* matches_A = ort_outputs[0].GetTensorMutableData<float>();
	const float* matches_B = ort_outputs[1].GetTensorMutableData<float>();
	int num_points = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[0];
	///cout << "tensor total element = " << ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() << endl;
	points_A.resize(num_points);
	for (int i = 0; i < num_points; i++)
	{
		points_A[i].pt.x = (matches_A[i * 2] + 1) * 0.5 * image_a.cols;
		points_A[i].pt.y = (matches_A[i * 2 + 1] + 1) * 0.5 * image_a.rows;
		points_A[i].size = 1.f;
	}

	num_points = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape()[0];
	points_B.resize(num_points);
	for (int i = 0; i < num_points; i++)
	{
		points_B[i].pt.x = (matches_B[i * 2] + 1) * 0.5 * image_b.cols;
		points_B[i].pt.y = (matches_B[i * 2 + 1] + 1) * 0.5 * image_b.rows;
		points_B[i].size = 1.f;
	}
}

int main()
{
	DeDoDeRunner_end2end mynet("weights/dedode_end2end_1024.onnx");
	string imgpath_a = "images/im_A.jpg";
	string imgpath_b = "images/im_B.jpg";
	Mat image_a = imread(imgpath_a);
	Mat image_b = imread(imgpath_b);

	vector<cv::KeyPoint> points_A;
	vector<cv::KeyPoint> points_B;
	mynet.detect(image_a, image_b, points_A, points_B);

	//匹配结果放在matches里面
	const int num_points = points_A.size();
	vector<DMatch> matches(num_points);
	for (int i = 0; i < num_points; i++)
	{
		matches[i] = DMatch(i, i, 0.f);
	}

	//按照匹配关系将图画出来，背景图为match_img
	Mat match_img;
	drawMatches(image_a, points_A, image_b, points_B, matches, match_img);

	//-- Show detected matches
	static const string kWinName = "Image Matches in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, match_img);
	waitKey(0);
	destroyAllWindows();
}