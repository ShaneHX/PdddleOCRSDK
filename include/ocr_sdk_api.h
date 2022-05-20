#pragma once
#ifndef _CVI_RE_
#define _CVI_RE_

#ifdef CVI_EXPORTS
#define CVI_API __declspec(dllexport)
#else
#define CVI_API __declspec(dllimport)
#endif

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "paddle_api.h"
#include "paddle_inference_api.h"

struct _OCRTextPoint
{
    int x;
    int y;
    _OCRTextPoint() : x(0), y(0)
    {
    }
};

struct _OCRText
{
    int textLen;
    char *ptext;

    _OCRTextPoint points[4];

    float score;

    float cls_score;
    int cls_label;

    _OCRText()
    {
        textLen = 0;
        ptext = nullptr;
        score = 0.0f;

        cls_score = 0.0f;
        cls_label = -1;
    }
};

typedef struct _OCRResult
{

    int textCount;
    _OCRText *pOCRText;
} OCRResult, *LpOCRResult;

#define DLLEXPORT __declspec(dllexport)
#ifdef __cplusplus

extern "C"
{
#endif

    // DLLEXPORT char* PaddleOCRText(cv::Mat& img);

    // DLLEXPORT int PaddleOCRTextRect(cv::Mat& , OCRTextRect*);
    DLLEXPORT void InitialSDK(bool det, bool rec, bool use_angle_cls, 
                            char* det_model_dir, 
                            char* rec_model_dir, 
                            char* cls_model_dir, 
                            char* rec_char_dict_path);
    DLLEXPORT int OCRRun(cv::Mat& img, LpOCRResult* resptr, bool det, bool rec, bool use_angle_cls);
    DLLEXPORT void FreeDetectResult(LpOCRResult pOCRResult);

#ifdef __cplusplus
}
#endif

#endif