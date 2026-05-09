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

        DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> listImg);


        IYoloDetectAsync GetYoloDetectAsync();

        IRunBatch<DetectionResult, PreDetectResultBatch> GetRunBatch();

    }
}
