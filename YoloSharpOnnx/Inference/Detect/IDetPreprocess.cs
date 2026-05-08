using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IDetPreprocess
    {
        PreResult PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer, InterpolationFlags interpolationFlags);
    }
}
