// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/paddleocr sdk.h"
#include <string>
#include <numeric>
#include <vector>

void PPOCRSDK::Init(bool det, bool rec, bool use_angle_cls,
					const std::string &det_model_dir,
					const std::string &rec_model_dir,
					const std::string &cls_model_dir,
					const std::string &rec_char_dict_path)
{

	bool use_gpu = false;
	int gpu_id = 0;
	int gpu_mem = 4000;
	int cpu_math_library_num_threads = 1;
	bool enable_mkldnn = true;

	int max_side_len = 960;

	double det_db_thresh = 0.3;
	double det_db_box_thresh = 0.5;
	double det_db_unclip_ratio = 2.0;
	std::string det_db_score_mode = "slow";
	bool use_dilation = false;

	bool visualize = true;
	bool use_tensorrt = false;
	std::string precision = "fp16";

	// cls setting
	double cls_thresh = 0.9;
	int cls_batch_num = 1;

	// rec setting
	int rec_batch_num = 6;
	int rec_img_h = 32;
	int rec_img_w = 320;

	if (det)
	{
		this->detector_ = new PaddleOCR::DBDetector(
			det_model_dir, use_gpu, gpu_id, gpu_mem,
			cpu_math_library_num_threads, enable_mkldnn, max_side_len,
			det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
			det_db_score_mode, use_dilation, use_tensorrt,
			precision);
	}

	if (use_angle_cls)
	{
		this->classifier_ = new PaddleOCR::Classifier(
			cls_model_dir, use_gpu, gpu_id, gpu_mem,
			cpu_math_library_num_threads, enable_mkldnn, cls_thresh,
			use_tensorrt, precision, cls_batch_num);
	}
	if (rec)
	{
		this->recognizer_ = new PaddleOCR::CRNNRecognizer(
			rec_model_dir, use_gpu, gpu_id, gpu_mem,
			cpu_math_library_num_threads, enable_mkldnn, rec_char_dict_path,
			use_tensorrt, precision, rec_batch_num,
			rec_img_h, rec_img_w);
	}
};

void PPOCRSDK::det(cv::Mat img, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
				   std::vector<double> &times)
{
	std::vector<std::vector<std::vector<int>>> boxes;
	std::vector<double> det_times;

	this->detector_->Run(img, boxes, det_times);

	for (int i = 0; i < boxes.size(); i++)
	{
		PaddleOCR::OCRPredictResult res;
		res.box = boxes[i];
		ocr_results.push_back(res);
	}

	times[0] += det_times[0];
	times[1] += det_times[1];
	times[2] += det_times[2];
}

void PPOCRSDK::rec(std::vector<cv::Mat> img_list,
				   std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
				   std::vector<double> &times)
{
	std::vector<std::string> rec_texts(img_list.size(), "");
	std::vector<float> rec_text_scores(img_list.size(), 0);
	std::vector<double> rec_times;
	this->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
	// output rec results
	for (int i = 0; i < rec_texts.size(); i++)
	{
		ocr_results[i].text = rec_texts[i];
		ocr_results[i].score = rec_text_scores[i];
	}
	times[0] += rec_times[0];
	times[1] += rec_times[1];
	times[2] += rec_times[2];
}

void PPOCRSDK::cls(std::vector<cv::Mat> img_list,
				   std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
				   std::vector<double> &times)
{
	std::vector<int> cls_labels(img_list.size(), 0);
	std::vector<float> cls_scores(img_list.size(), 0);
	std::vector<double> cls_times;
	this->classifier_->Run(img_list, cls_labels, cls_scores, cls_times);
	// output cls results
	for (int i = 0; i < cls_labels.size(); i++)
	{
		ocr_results[i].cls_label = cls_labels[i];
		ocr_results[i].cls_score = cls_scores[i];
	}
	times[0] += cls_times[0];
	times[1] += cls_times[1];
	times[2] += cls_times[2];
}

std::vector<std::vector<PaddleOCR::OCRPredictResult>>
PPOCRSDK::ocr(const cv::Mat& img, bool det, bool rec,
			  bool cls)
{
	std::vector<double> time_info_det = {0, 0, 0};
	std::vector<double> time_info_rec = {0, 0, 0};
	std::vector<double> time_info_cls = {0, 0, 0};
	std::vector<std::vector<PaddleOCR::OCRPredictResult>> ocr_results;

	std::vector<cv::Mat> img_list;

	if (!det)
	{
		std::vector<PaddleOCR::OCRPredictResult> ocr_result;


		img_list.push_back(img);
		PaddleOCR::OCRPredictResult res;
		ocr_result.push_back(res);

		if (cls && this->classifier_ != nullptr)
		{
			this->cls(img_list, ocr_result, time_info_cls);
			for (int i = 0; i < img_list.size(); i++)
			{
				if (ocr_result[i].cls_label % 2 == 1 &&
					ocr_result[i].cls_score > this->classifier_->cls_thresh)
				{
					cv::rotate(img_list[i], img_list[i], 1);
				}
			}
		}
		if (rec)
		{
			this->rec(img_list, ocr_result, time_info_rec);
		}
		for (int i = 0; i < img_list.size(); ++i)
		{
			std::vector<PaddleOCR::OCRPredictResult> ocr_result_tmp;
			ocr_result_tmp.push_back(ocr_result[i]);
			ocr_results.push_back(ocr_result_tmp);
		}
	}
	 else
	 {
		std::vector<PaddleOCR::OCRPredictResult> ocr_result;

		if (!img.data)
		{
			std::cerr << "[ERROR] input image data error!" << endl;
			exit(1);
		}
		// det
		this->det(img, ocr_result, time_info_det);
		// crop image
		std::vector<cv::Mat> img_list;
		for (int j = 0; j < ocr_result.size(); j++)
		{
			cv::Mat crop_img;
			crop_img = PaddleOCR::Utility::GetRotateCropImage(img, ocr_result[j].box);
			img_list.push_back(crop_img);
		}

		// cls
		if (cls && this->classifier_ != nullptr)
		{
			this->cls(img_list, ocr_result, time_info_cls);
			for (int i = 0; i < img_list.size(); i++)
			{
				if (ocr_result[i].cls_label % 2 == 1 &&
					ocr_result[i].cls_score > this->classifier_->cls_thresh)
				{
					cv::rotate(img_list[i], img_list[i], 1);
				}
			}
		}
		// rec
		if (rec)
		{
			this->rec(img_list, ocr_result, time_info_rec);
		}
		ocr_results.push_back(ocr_result);
	 }

	return ocr_results;
} // namespace PaddleOCR

// std::vector<std::vector<PaddleOCR::OCRPredictResult>>
// PPOCRSDK::ocr(std::vector<cv::String> cv_all_img_names, bool det, bool rec,
//            bool cls) {
//   std::vector<double> time_info_det = {0, 0, 0};
//   std::vector<double> time_info_rec = {0, 0, 0};
//   std::vector<double> time_info_cls = {0, 0, 0};
//   std::vector<std::vector<PaddleOCR::OCRPredictResult>> ocr_results;

//   if (!det) {
//     std::vector<PaddleOCR::OCRPredictResult> ocr_result;
//     // read image
//     std::vector<cv::Mat> img_list;
//     for (int i = 0; i < cv_all_img_names.size(); ++i) {
//       cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
//       if (!srcimg.data) {
//         std::cerr << "[ERROR] image read failed! image path: "
//                   << cv_all_img_names[i] << endl;
//         exit(1);
//       }
//       img_list.push_back(srcimg);
//       PaddleOCR::OCRPredictResult res;
//       ocr_result.push_back(res);
//     }
//     if (cls && this->classifier_ != nullptr) {
//       this->cls(img_list, ocr_result, time_info_cls);
//       for (int i = 0; i < img_list.size(); i++) {
//         if (ocr_result[i].cls_label % 2 == 1 &&
//             ocr_result[i].cls_score > this->classifier_->cls_thresh) {
//           cv::rotate(img_list[i], img_list[i], 1);
//         }
//       }
//     }
//     if (rec) {
//       this->rec(img_list, ocr_result, time_info_rec);
//     }
//     for (int i = 0; i < cv_all_img_names.size(); ++i) {
//       std::vector<PaddleOCR::OCRPredictResult> ocr_result_tmp;
//       ocr_result_tmp.push_back(ocr_result[i]);
//       ocr_results.push_back(ocr_result_tmp);
//     }
//   } else {

//     for (int i = 0; i < cv_all_img_names.size(); ++i) {
//       std::vector<PaddleOCR::OCRPredictResult> ocr_result;

//       cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
//       if (!srcimg.data) {
//         std::cerr << "[ERROR] image read failed! image path: "
//                   << cv_all_img_names[i] << endl;
//         exit(1);
//       }
//       // det
//       this->det(srcimg, ocr_result, time_info_det);
//       // crop image
//       std::vector<cv::Mat> img_list;
//       for (int j = 0; j < ocr_result.size(); j++) {
//         cv::Mat crop_img;
//         crop_img = PaddleOCR::Utility::GetRotateCropImage(srcimg, ocr_result[j].box);
//         img_list.push_back(crop_img);
//       }

//       // cls
//       if (cls && this->classifier_ != nullptr) {
//         this->cls(img_list, ocr_result, time_info_cls);
//         for (int i = 0; i < img_list.size(); i++) {
//           if (ocr_result[i].cls_label % 2 == 1 &&
//               ocr_result[i].cls_score > this->classifier_->cls_thresh) {
//             cv::rotate(img_list[i], img_list[i], 1);
//           }
//         }
//       }
//       // rec
//       if (rec) {
//         this->rec(img_list, ocr_result, time_info_rec);
//       }
//       ocr_results.push_back(ocr_result);
//     }
//   }

//   return ocr_results;
// } // namespace PaddleOCR

PPOCRSDK::~PPOCRSDK()
{
	if (this->detector_ != nullptr)
	{
		delete this->detector_;
	}
	if (this->classifier_ != nullptr)
	{
		delete this->classifier_;
	}
	if (this->recognizer_ != nullptr)
	{
		delete this->recognizer_;
	}
};
