using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;

        public YoloConfiguration YoloConfiguration { get; set; }

        public YoloSharp(YoloConfiguration yoloConfig, IExecutionProvider executionProvider)
        {
            YoloConfiguration = yoloConfig;
            _yoloDetect = executionProvider.CreateYoloDetect();
        }

        public YoloSharp(IExecutionProvider executionProvider)
        {
            YoloConfiguration = YoloConfiguration.Default;
            _yoloDetect = executionProvider.CreateYoloDetect();
        }

        public YoloSharp(float confidence, float iou, IExecutionProvider executionProvider)
        {
            YoloConfiguration = new YoloConfiguration(confidence, iou);
            _yoloDetect = executionProvider.CreateYoloDetect();
        }

        public YoloSharp(float confidence, float iou, InterpolationFlags resizeAlgorithm, IExecutionProvider executionProvider)
        {
            YoloConfiguration = new YoloConfiguration(confidence, iou, resizeAlgorithm);
            _yoloDetect = executionProvider.CreateYoloDetect();
        }

        public List<DetectionResult> RunDetect(string path)
        {
            using (Mat img = Cv2.ImRead(path))
            {
                return _yoloDetect.Run(img, YoloConfiguration);
            }
        }

        public YoloResult<DetectionResult> RunDetectWithTime(string path)
        {
            using (Mat img = Cv2.ImRead(path))
            {
                return _yoloDetect.RunDetect(img, YoloConfiguration);
            }
        }

        public void DrawDetections(Mat inputImage, List<DetectionResult> list)
        {
            _yoloDetect.DrawDetections(inputImage, list);
        }

        public void Dispose()
        {
            _yoloDetect?.Dispose();

        }
    }
}
