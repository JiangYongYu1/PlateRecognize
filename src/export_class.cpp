#include "plate.hpp"

extern "C" _declspec(dllexport) bool GetPlateObject(void **_RtObject)
{
    Plate *pMan = new Plate();
    *_RtObject = (void *)pMan;
    return true;
}

extern "C" _declspec(dllexport) bool ReleaseCtLungObject(void *RtObject)
{
    delete RtObject;
    RtObject = NULL;
    return true;
}

