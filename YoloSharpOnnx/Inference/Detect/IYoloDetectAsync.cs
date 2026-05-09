using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IYoloDetectAsync
    {

        PreDetectResultBatch PreprocessImageChannel(string imagePath);
        PreDetectResultBatch PreprocessImageChannel(Mat img, string imagePath);

        void InitBufferPool(int batchPoolSize);

        int BufferPoolUsedCount { get; }

    }
}
