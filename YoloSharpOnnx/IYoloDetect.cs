using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public interface IYoloDetect : IDisposable
    {
        List<DetectionResult> Run(Mat inputImage, YoloConfig yoloConfig);

        YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfig yoloConfig);

        void DrawDetections(Mat inputImage, List<DetectionResult> list);

        DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, int batchPoolSize, YoloConfig yoloConfig);

        Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, int batchPoolSize, YoloConfig yoloConfig);

        event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

    }
}
