#include <Windows.h>
#include "string"

#include "plate_sample.hpp"


typedef void (*FUNADDR)(void**);

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Please design config file and image name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string image_name = argv[2];

    HINSTANCE dllDemo = ::LoadLibrary("plate_recognize.dll");
	DWORD dwErrNO = GetLastError();
	if (dwErrNO) {
		std::cout << "dll open errno : " + std::to_string(dwErrNO) << std::endl;
	}

	if (dllDemo) {
		printf("load dll demo\n");
		FUNADDR getclass;
		getclass = (FUNADDR)GetProcAddress(dllDemo, "GetPlateObject");

        void* general_pointor;
		getclass(&general_pointor);
		PlateSample* plate_instance = (PlateSample*)general_pointor;
        plate_instance->Init(config_file);
        for(int i = 0; i < 1000; i++)
        {
            DetectionResult detect_result;
            RecognizeResult recog_result;
            plate_instance->Run(image_name, detect_result, recog_result);
            std::cout<<"图片 " <<image_name<<" 的识别结果是: \n";
            for(int i=0; i < recog_result.RegOuts.size(); i++)
            {
                for (int j=0; j<recog_result.RegOuts[i].reg_out.size(); j++)
                {
                    std::cout<<recog_result.RegOuts[i].reg_out[j];
                }
                std::cout<<"\n";
            }
            std::cout<<"识别完成!"<<std::endl;
        }

    }
    return 0;
}