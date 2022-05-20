
#include "ocr_sdk_api.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

#include<Windows.h>

#include <iostream>

#pragma comment (lib,"PaddleOCRSDK.lib")
extern "C" {


    __declspec(dllimport) void InitialSDK(bool det, bool rec, bool use_angle_cls,
        char* det_model_dir,
        char* rec_model_dir,
        char* cls_model_dir,
        char* rec_char_dict_path);
    __declspec(dllimport) int OCRRun(cv::Mat& img, LpOCRResult* resptr, bool det, bool rec, bool use_angle_cls);
    __declspec(dllimport) void FreeDetectResult(LpOCRResult pOCRResult);
}

int main()
{
    char det_path[] = "D:\\WorkSpace\\paddle-ocrsharp\\PaddleOCRLib\\inference\\ch_PP-OCRv3_det_infer";
    char rec_path[] = "D:\\WorkSpace\\paddle-ocrsharp\\PaddleOCRLib\\inference\\ch_PP-OCRv3_rec_infer";
    char cls_path[] = "D:\\WorkSpace\\paddle-ocrsharp\\PaddleOCRLib\\inference\\ch_ppocr_mobile_v2.0_cls_infer";
    char key_path[] = "D:\\WorkSpace\\paddle-ocrsharp\\PaddleOCRLib\\inference\\ppocr_keys.txt";


    LARGE_INTEGER timeStart;	//开始时间
    LARGE_INTEGER timeEnd;		//结束时间

    LARGE_INTEGER frequency;	//计时器频率
    QueryPerformanceFrequency(&frequency);
    double quadpart = (double)frequency.QuadPart;//计时器频率

    InitialSDK(true, true, true, det_path, rec_path, cls_path, key_path);


    //cv::Mat img = cv::imread("D:\\WorkSpace\\PaddleOCRDemo\\test_img\\1.png");
    cv::Mat img = cv::imread("D:\\WorkSpace\\PaddleOCRDemo\\test_img\\2.jpg");
    LpOCRResult lpocrreult;
    SetConsoleOutputCP(CP_UTF8);
    DWORD det_num = 0;
    double total_time = 0.0;
    for (int i = 0; i < 10000; i++)
    {
        QueryPerformanceCounter(&timeStart);
        OCRRun(img, &lpocrreult, true, true, true);
        QueryPerformanceCounter(&timeEnd);
        det_num++;


        //得到两个时间的耗时
        total_time += (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;

        FreeDetectResult(lpocrreult);
        std::cout << "time cost:" << total_time / det_num << std::endl;

        Sleep(100);

    }

}