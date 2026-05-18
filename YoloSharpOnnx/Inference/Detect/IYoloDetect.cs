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

        DetectionBatchResult[] BatchRunPostSync(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        Task<DetectionBatchResult[]> BatchRunAsyncPostSync(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction);

        IAsyncEnumerable<DetectionBatchResult> BatchRunForeachSync(List<string> listImg);


        IYoloProcessAsync<PreDetectResultBatch> GetYoloProcessAsync();

        IRunBatch<DetectionResult, PreDetectResultBatch> GetRunBatch();

    }
}
