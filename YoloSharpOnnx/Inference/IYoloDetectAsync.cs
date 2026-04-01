using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public interface IYoloDetectAsync: IBatchDetect
    {

        PreResultBatch PreprocessImageChannel(string imagePath, InterpolationFlags interpolationFlags);
        PreResultBatch PreprocessImageChannel(Mat img, string imagePath, InterpolationFlags interpolationFlags);

        void InitBufferPool(int batchPoolSize);

        int BufferPoolUsedCount { get; }

    }
}
