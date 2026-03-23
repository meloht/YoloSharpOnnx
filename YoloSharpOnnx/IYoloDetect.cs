using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public interface IYoloDetect: IDisposable
    {
        List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig);

    }
}
