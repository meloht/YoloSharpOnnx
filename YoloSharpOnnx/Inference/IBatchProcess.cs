using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public interface IBatchProcess<TResult, TBatchPreResult, TBatchResult> :IRunBatch<TResult, TBatchPreResult>
    {
        TBatchPreResult GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath);

        TBatchResult BuildBatchResult(string imagePath, List<TResult> results, long timestamp);


    }
}
