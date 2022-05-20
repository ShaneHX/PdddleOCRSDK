#include <include/ocr_sdk_api.h>
#include <include/paddleocr sdk.h>
#include <memory>
#include <include/args.h>

std::unique_ptr<PPOCRSDK> ocr_sdk;

DLLEXPORT void InitialSDK(bool det, bool rec, bool use_angle_cls, char* det_model_dir, char* rec_model_dir, char* cls_model_dir, char* rec_char_dict_path)
{

	ocr_sdk = std::make_unique<PPOCRSDK>();
	ocr_sdk->Init(det, rec, use_angle_cls,
		det_model_dir,
		rec_model_dir,
		cls_model_dir,
		rec_char_dict_path);

}
DLLEXPORT int OCRRun(cv::Mat &img, LpOCRResult *resptr, bool det, bool rec, bool use_angle_cls)
{
	std::vector<PaddleOCR::OCRPredictResult> ocr_result = ocr_sdk->ocr(img, det, rec, use_angle_cls)[0];


    LpOCRResult ocr_res_ptr = new _OCRResult();
    *resptr = ocr_res_ptr;
    ocr_res_ptr->textCount = ocr_result.size();
    ocr_res_ptr->pOCRText = new _OCRText[ocr_res_ptr->textCount];


    for (int i = 0; i < ocr_result.size(); i++) {
       // det
       std::vector<std::vector<int>> boxes = ocr_result[i].box;
       if (boxes.size() > 0) {

           for (int n = 0; n < boxes.size(); n++) {
			   ocr_res_ptr->pOCRText[i].points->x = boxes[n][0];
			   ocr_res_ptr->pOCRText[i].points->y = boxes[n][1];
           }
       }
       // rec
       if (ocr_result[i].score != -1.0) {
		   ocr_res_ptr->pOCRText[i].score = ocr_result[i].score;
		   ocr_res_ptr->pOCRText[i].textLen = ocr_result[i].text.length();
		   ocr_res_ptr->pOCRText[i].ptext = new char[ocr_result[i].text.length()];
		   if (ocr_res_ptr->pOCRText[i].ptext != nullptr
		   && ocr_res_ptr->pOCRText[i].textLen >= 0)
		   {
				memcpy(ocr_res_ptr->pOCRText[i].ptext, ocr_result[i].text.c_str(), ocr_res_ptr->pOCRText[i].textLen);
		   }


       }

       // cls
       if (ocr_result[i].cls_label != -1) {
           ocr_res_ptr->pOCRText[i].cls_label = ocr_result[i].cls_label;
           ocr_res_ptr->pOCRText[i].score = ocr_result[i].cls_score;
        
       }

    }




	cout << "---------------------ocr_output----------------" << "\n";
	PaddleOCR::Utility::print_result(ocr_result);

}

DLLEXPORT void FreeDetectResult(LpOCRResult pOCRResult)
{
    if (pOCRResult == nullptr)  return;
    if (pOCRResult->textCount == 0) return;

    for (int i = 0; i < pOCRResult->textCount; i++)
    {
        delete[] pOCRResult->pOCRText[i].ptext;
        pOCRResult->pOCRText[i].ptext = nullptr;
    }
    delete[] pOCRResult->pOCRText;
    pOCRResult->pOCRText = nullptr;

    delete pOCRResult;
    pOCRResult = nullptr;
}


