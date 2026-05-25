using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    internal interface IYoloSegment : IDisposable
    {
        List<SegResult> Run(Mat inputImage);

        YoloResult<SegResult> RunWithTime(Mat inputImage);

        void DrawSegments(Mat inputImage, List<SegResult> list);

        SegBatchResult[] BatchRun(List<string> listImg, IBatchProcessCallback<SegBatchResult> processCallback, Action<SegBatchResult> receiveAction);

        Task<SegBatchResult[]> BatchRunAsync(List<string> listImg, IBatchProcessCallback<SegBatchResult> processCallback, Action<SegBatchResult> receiveAction);

        IAsyncEnumerable<SegBatchResult> BatchRunForeachAsync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch, SegResult> GetYoloProcessAsync();


    }
}
