using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IDetPreprocess
    {
        PreDetectResult PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer);
    }
}
