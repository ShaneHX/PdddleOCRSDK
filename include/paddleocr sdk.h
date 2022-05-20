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

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/preprocess_op.h>
#include <include/utility.h>

using namespace paddle_infer;

class PPOCRSDK
{

public:
	PPOCRSDK() = default;
	~PPOCRSDK();

	void Init(bool det, bool rec, bool use_angle_cls,
			  const std::string &det_model_dir,
			  const std::string &rec_model_dir,
			  const std::string &cls_model_dir,
			  const std::string &rec_char_dict_path);

	std::vector<std::vector<PaddleOCR::OCRPredictResult>>
	ocr(const cv::Mat &img, bool det, bool rec,
		bool cls);

private:
	PaddleOCR::DBDetector *detector_ = nullptr;
	PaddleOCR::Classifier *classifier_ = nullptr;
	PaddleOCR::CRNNRecognizer *recognizer_ = nullptr;

	void det(cv::Mat img, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
			 std::vector<double> &times);
	void rec(std::vector<cv::Mat> img_list,
			 std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
			 std::vector<double> &times);
	void cls(std::vector<cv::Mat> img_list,
			 std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
			 std::vector<double> &times);
	void log(std::vector<double> &det_times, std::vector<double> &rec_times,
			 std::vector<double> &cls_times, int img_num);
};
