using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Utils;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal interface IDetPreprocess
    {
        PreDetectResult PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer);
    }
}
