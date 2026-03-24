using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public interface IYoloDetect: IDisposable
    {
        List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig);

        YoloResult<DetectionResult> RunDetect(Mat inputImage, YoloConfiguration yoloConfig);

    }
}
