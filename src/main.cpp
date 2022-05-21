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
//#include "opencv2/core.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgproc.hpp"
//#include <iostream>
//#include <vector>
//
//#include <include/args.h>
//#include <include/paddleocr.h>
//
//using namespace PaddleOCR;
//
//void check_params() {
//  if (FLAGS_det) {
//    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
//      std::cout << "Usage[det]: ./ppocr "
//                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
//                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
//      exit(1);
//    }
//  }
//  if (FLAGS_rec) {
//    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
//      std::cout << "Usage[rec]: ./ppocr "
//                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
//                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
//      exit(1);
//    }
//  }
//  if (FLAGS_cls && FLAGS_use_angle_cls) {
//    if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty()) {
//      std::cout << "Usage[cls]: ./ppocr "
//                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
//                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
//      exit(1);
//    }
//  }
//  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
//      FLAGS_precision != "int8") {
//    cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. " << endl;
//    exit(1);
//  }
//}
//
//int main(int argc, char **argv) {
//  // Parsing command-line
//  google::ParseCommandLineFlags(&argc, &argv, true);
//  check_params();
//
//  if (!Utility::PathExists(FLAGS_image_dir)) {
//    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
//              << endl;
//    exit(1);
//  }
//
//  std::vector<cv::String> cv_all_img_names;
//  cv::glob(FLAGS_image_dir, cv_all_img_names);
//  std::cout << "total images num: " << cv_all_img_names.size() << endl;
//
//  PPOCR ocr = PPOCR();
//
//  std::vector<std::vector<OCRPredictResult>> ocr_results =
//      ocr.ocr(cv_all_img_names, FLAGS_det, FLAGS_rec, FLAGS_cls);
//
//  for (int i = 0; i < cv_all_img_names.size(); ++i) {
//    if (FLAGS_benchmark) {
//      cout << cv_all_img_names[i] << '\t';
//      for (int n = 0; n < ocr_results[i].size(); n++) {
//        for (int m = 0; m < ocr_results[i][n].box.size(); m++) {
//          cout << ocr_results[i][n].box[m][0] << ' '
//               << ocr_results[i][n].box[m][1] << ' ';
//        }
//      }
//      cout << endl;
//    } else {
//      cout << cv_all_img_names[i] << "\n";
//      Utility::print_result(ocr_results[i]);
//      if (FLAGS_visualize && FLAGS_det) {
//        cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
//        if (!srcimg.data) {
//          std::cerr << "[ERROR] image read failed! image path: "
//                    << cv_all_img_names[i] << endl;
//          exit(1);
//        }
//        std::string file_name = Utility::basename(cv_all_img_names[i]);
//
//        Utility::VisualizeBboxes(srcimg, ocr_results[i],
//                                 FLAGS_output + "/" + file_name);
//      }
//      cout << "***************************" << endl;
//    }
//  }
//}


#include<Windows.h>
#include "include/ocr_sdk_api.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    cv::Mat game_window_img = cv::imread("D:\\WorkSpace\\GodCreatorV2\\test\\work_dir\\dataset\\1653044764636.jpg");

    cv::Mat weapon_region_img = cv::Mat(game_window_img, cv::Rect(2175, 820, 385, 550 + 70));


    char det_path[] = "D:\\WorkSpace\\HyperFps\\models\\en_PP-OCRv3_det_slim_infer";
    char rec_path[] = "D:\\WorkSpace\\HyperFps\\models\\en_PP-OCRv3_rec_slim_infer";
    char cls_path[] = "D:\\WorkSpace\\HyperFps\\models\\ch_ppocr_mobile_v2.0_cls_slim_infer";
    char key_path[] = "D:\\WorkSpace\\HyperFps\\models\\dict\\en_dict.txt";


    InitialSDK(true, true, true, det_path, rec_path, cls_path, key_path);
    LpOCRResult lpocrreult;
    SetConsoleOutputCP(CP_UTF8);



    std::vector<cv::Mat> weapon_region_vec;

    const int weapon_region_w = weapon_region_img.cols;
    const int weapon_region_h = weapon_region_img.rows;


    LARGE_INTEGER timeStart;	//开始时间
    LARGE_INTEGER timeEnd;		//结束时间

    LARGE_INTEGER frequency;	//计时器频率
    QueryPerformanceFrequency(&frequency);
    double quadpart = (double)frequency.QuadPart;//计时器频率


 //   cv::imshow("src_weapon_region", weapon_region_img);
 //   cv::waitKey();
 //   QueryPerformanceCounter(&timeStart);
    //OCRRun(weapon_region_img, &lpocrreult, true, true, false);
    //QueryPerformanceCounter(&timeEnd);
 //   double time_cost = double((timeEnd.QuadPart - timeStart.QuadPart) / quadpart);
 //   D_INFO("time cost: %f", time_cost);
 //   cv::waitKey();

    for (int i = 0; i < 5; i++)
    {

        int start_y = weapon_region_h - (i + 1) * 110 - 70;
        std::cout << "start_y:" << start_y << std::endl;
        cv::Mat s_weapon_region = cv::Mat(weapon_region_img, cv::Rect(0, start_y, weapon_region_w, 110));
        cv::Mat s_region_idx_img = cv::Mat(s_weapon_region, cv::Rect(s_weapon_region.cols - 20, 5, 20, 40));
        cv::Mat show_idx_img;
        //cv::resize(s_region_idx_img, show_idx_img, cv::Size(100, 100));
        //QueryPerformanceCounter(&timeStart);
        //auto a = OCRRun(show_idx_img, &lpocrreult, true, true, false);
        //QueryPerformanceCounter(&timeEnd);

        cv::resize(s_region_idx_img, show_idx_img, cv::Size(100, 100));
        QueryPerformanceCounter(&timeStart);
        auto a = OCRRun(weapon_region_img, &lpocrreult, true, true, false);
        QueryPerformanceCounter(&timeEnd);

        double time_cost = double((timeEnd.QuadPart - timeStart.QuadPart) / quadpart);
        std::cout << "time_cost:" << time_cost << std::endl;

        FreeDetectResult(lpocrreult);
        cv::imshow(std::to_string(i) + "src", s_weapon_region);
        cv::imshow(std::to_string(i), show_idx_img);
        s_weapon_region.release();
        s_region_idx_img.release();
        show_idx_img.release();


    }








    cv::imshow("test", weapon_region_img);



    cv::waitKey();

}

