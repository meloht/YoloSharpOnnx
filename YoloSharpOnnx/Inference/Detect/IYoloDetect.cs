using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IYoloDetect : IDisposable
    {
        List<DetectionResult> Run(Mat inputImage);

        YoloResult<DetectionResult> RunWithTime(Mat inputImage);

        void DrawDetections(Mat inputImage, List<DetectionResult> list);

        DetectionBatchResult[] BatchRun(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        Task<DetectionBatchResult[]> BatchRunAsync(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        IAsyncEnumerable<DetectionBatchResult> BatchRunForeachAsync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch> GetYoloProcessAsync();

        IRunBatch<DetectionResult, PreDetectResultBatch> GetRunBatch();

    }
}
