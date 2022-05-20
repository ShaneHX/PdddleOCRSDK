# PaddleOCR封装成dll笔记

# 1️⃣ dll导出函数编写

这部分主要实现了三个函数， 具体实现参考源码：

```cpp
DLLEXPORT void InitialSDK(bool det, bool rec, bool use_angle_cls, 
                            char* det_model_dir, 
                            char* rec_model_dir, 
                            char* cls_model_dir, 
                            char* rec_char_dict_path);

DLLEXPORT int OCRRun(cv::Mat& img, LpOCRResult* resptr, bool det, bool rec, bool use_angle_cls);

DLLEXPORT void FreeDetectResult(LpOCRResult pOCRResult);
```

- `InitialSDK` : 初始化SDK，加载各种模型。
- `OCRRun`: 单张图片的OCR识别函数。
- `FreeDetectResult`: 当通过OCRRun获取到识别结果后，用完识别结果需要释放这部分内存，防止内存泄露。

# 2️⃣ 编写main.cpp，测试dll导出函数

- 可以在ppocr先建cpp文件，定义main函数，或者直接把ppocr中`main.cpp`的的main函数注释掉，添加自己的main函数。
    
    ```cpp
    #include<Windows.h>
    #include "include/ocr_sdk_api.h"
    
    #include "opencv2/core.hpp"
    #include "opencv2/imgcodecs.hpp"
    #include "opencv2/imgproc.hpp"
    #include <iostream>
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
    
    		//初始化SDK,加载模型
    	  InitialSDK(true, true, true, det_path, rec_path, cls_path, key_path);
    
        //cv::Mat img = cv::imread("D:\\WorkSpace\\PaddleOCRDemo\\test_img\\1.png");
        cv::Mat img = cv::imread("D:\\WorkSpace\\PaddleOCRDemo\\test_img\\2.jpg");
    
        LpOCRResult lpocrreult;
        SetConsoleOutputCP(CP_UTF8);  // 这个函数很重要，如果程序vs输出窗口打印的中文乱码的话，就要加上这个函数
    
        DWORD det_num = 0;
        double total_time = 0.0;
    
        for(int i =0; i < 10000; i++)  //运行10000次， 测试内存泄露
        {
            QueryPerformanceCounter(&timeStart);
    
            OCRRun(img, &lpocrreult);  //ocr识别函数
    
            QueryPerformanceCounter(&timeEnd);
            det_num++;
    
            //得到两个时间的耗时
            total_time += (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    
            FreeDetectResult(lpocrreult);  // 释放ocr输出结构体，防止内存泄露。
            std::cout << "time cost:" << total_time/ det_num << std::endl;
            Sleep(100);
    
        }
    
    }
    ```
    
- ⚠️注意点：
    - 编译程序只能用release模式，因为paddle_inference库内用的是所有的库都是release版本，用debug会报错。
    - 测试你写的mian函数时，记得把ppocr设置成启动项目。
    - 注意依赖：运行main函数的时候，需要将paddleocr依赖的dll库都放置在ppocr.exe的目录下。 参考[依赖](https://www.notion.so/PaddleOCR-dll-751dc24e26f7421598bc8c8c0f1611ca)

# 3️⃣CMakeLists配置

写好导出函数之后，需要配置CMakeLists, 代码如下：

```makefile
set(BUILD_SHARED_LIBS ON)
add_library(PaddleOCRSDK SHARED ${SRCS})
target_link_libraries(PaddleOCRSDK ${DEPS})
```

这段代码添加的位置是原先`CMakeLists.txt`中生成可执行文件的命令下方，如下：

```makefile
# 原来的
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(${DEMO_NAME} ${SRCS})
target_link_libraries(${DEMO_NAME} ${DEPS})

# 现在添加的
set(BUILD_SHARED_LIBS ON)
add_library(PaddleOCRExport SHARED ${SRCS}) 
target_link_libraries(PaddleOCRExport ${DEPS})
```

⚠️注意：add_library的时候一定要是SHARED格式的，不能是STATIC，不然后面调用对应的函数，会报找不到dll里面的一些函数。

# 4️⃣ 调用生成的dll

在调用PaddleOCRSDK的工程中引入`PaddleOCRSDK.dll`的代码如下：

```cpp
#include "ocr_sdk_api.h" //一定要有这个头文件，或者把这个头文件里面的LpOCRResult 结构体拿出来

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
```

然后就可以编写函数调用这三个函数了。

⚠️注意点：

- 一定要先include `"opencv2/core.hpp"`和`"opencv2/imgcodecs.hpp"`，然后再include `<Windows.h>`。不然std::min会报错。
- 注意PaddleOCRSDK的依赖问题。
    - 编译调用程序的时候，`PaddleOCRSDK.lib`，`PaddleOCRSDK.dll`和 `PaddleOCRSDK.exp`需要在visual studio的工程下面。
    - 编译通过后，运行对应生成的exe，需要把paddle ocr的依赖和PaddleOCRSDK的依赖都放到exe目录下面。

# 🧰 依赖

- PaddleOCR依赖
    
    使用或者编译的PaddleOCR相关的程序时一般会依赖一些如下文件：
    
    ```cpp
    // 这几个是编译的时候，自动生成的。
    libiomp5md.dll
    mkldnn.dll
    mklml.dll
    
    // 这个是paddle_inference包里面找
    paddle_inference.dll
    
    //这两个是要重paddle_inference 的third_party里面找
    onnxruntime.dll
    paddle2onnx.dll
    ```
    
- PaddleOCR SDK依赖
    
    使用或者编译的PaddleOCRSDK相关的程序时，一般会依赖一些如下文件：
    
    ```cpp
    PaddleOCRSDK.lib
    PaddleOCRSDK.dll
    PaddleOCRSDK.exp
    ```