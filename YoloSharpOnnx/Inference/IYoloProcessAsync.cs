using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference
{
    internal interface IYoloProcessAsync<TBatchPreResult, TResult> : IRunBatch<TResult, TBatchPreResult>
    {
        TBatchPreResult PreprocessImageChannel(string imagePath);
        TBatchPreResult PreprocessImageChannel(Mat img, string imagePath);

        void InitBufferPool(int batchPoolSize);

        int BufferPoolUsedCount { get; }
    }
}
