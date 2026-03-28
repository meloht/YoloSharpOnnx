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
        List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig);

        YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfiguration yoloConfig);

        void DrawDetections(Mat inputImage, List<DetectionResult> list);

        DetectionBatchResult[] BatchDetect(List<string> listImg, int batchPoolSize, YoloConfiguration yoloConfig);

        Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, int batchPoolSize, YoloConfiguration yoloConfig);

        event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

    }
}
