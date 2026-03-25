using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;

        public event EventHandler<BatchDetectionResultEventArgs> BatchDetectCompleted;

        public YoloConfiguration YoloConfiguration { get; set; }

        public YoloSharp(YoloConfiguration yoloConfig, IExecutionProvider executionProvider)
        {
            YoloConfiguration = yoloConfig;
            InitDetector(executionProvider);
        }

        public YoloSharp(IExecutionProvider executionProvider)
        {
            YoloConfiguration = YoloConfiguration.Default;
            InitDetector(executionProvider);
        }

        public YoloSharp(float confidence, float iou, IExecutionProvider executionProvider)
        {
            YoloConfiguration = new YoloConfiguration(confidence, iou);
            InitDetector(executionProvider);
        }

        public YoloSharp(float confidence, float iou, InterpolationFlags resizeAlgorithm, IExecutionProvider executionProvider)
        {
            YoloConfiguration = new YoloConfiguration(confidence, iou, resizeAlgorithm);
            InitDetector(executionProvider);
        }
        private void InitDetector(IExecutionProvider executionProvider)
        {
            _yoloDetect = executionProvider.CreateYoloDetect();
            _yoloDetect.BatchDetectCompleted += _yoloDetect_BatchDetectCompleted;
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
                return _yoloDetect.RunWithTime(img, YoloConfiguration);
            }
        }

        public void RunBatchDetect(string path, int batchSize = 50)
        {

            var files = Directory.GetFiles(path);

            _yoloDetect.BatchDetect(files, batchSize, YoloConfiguration);
        }

        private void _yoloDetect_BatchDetectCompleted(object? sender, BatchDetectionResultEventArgs e)
        {
            BatchDetectCompleted?.Invoke(sender, e);
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
