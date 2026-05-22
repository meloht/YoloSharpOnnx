using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Classify
{
    internal interface IClsPreprocess
    {
        void PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer);
    }
}
