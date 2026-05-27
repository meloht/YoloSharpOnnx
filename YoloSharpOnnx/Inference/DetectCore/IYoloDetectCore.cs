using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal interface IYoloDetectCore<TResult, TBatchResult> : IDisposable where TResult : IYoloSummary<TResult>
    {
        List<TResult> Run(Mat inputImage);

        YoloResult<TResult> RunWithTime(Mat inputImage);

        void DrawDetections(Mat inputImage, List<TResult> list);

        TBatchResult[] BatchRunPostSync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction);

        Task<TBatchResult[]> BatchRunAsyncPostSync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction);

        IAsyncEnumerable<TBatchResult> BatchRunForeachSync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch, TResult> GetYoloProcessAsync();
    }
}
