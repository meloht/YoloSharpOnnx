using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;

        public event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

        public YoloConfig YoloConfiguration { get; set; }

        #region Constructor


        public YoloSharp(IExecutionProvider executionProvider) : this(new YoloConfig(), executionProvider)
        {

        }
        public YoloSharp(YoloConfig yoloConfig, IExecutionProvider executionProvider)
        {
            YoloConfiguration = yoloConfig;
            _yoloDetect = executionProvider.CreateYoloDetect();
            _yoloDetect.BatchDetectItemCompleted += YoloDetect_BatchDetectItemCompleted;
        }

        public YoloSharp(float confidence, float iou, IExecutionProvider executionProvider)
            : this(confidence, iou, InterpolationFlags.Linear, executionProvider)
        {

        }

        public YoloSharp(float confidence, float iou, InterpolationFlags resizeAlgorithm, IExecutionProvider executionProvider)
            : this(new YoloConfig(confidence, iou, resizeAlgorithm), executionProvider)
        {

        }



        #endregion

        #region Synchronous
        public List<DetectionResult> RunDetect(string imagePath)
        {
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.Run(img, YoloConfiguration);
            }
        }

        public List<DetectionResult> RunDetect(Mat img)
        {
            return _yoloDetect.Run(img, YoloConfiguration);
        }

        public YoloResult<DetectionResult> RunDetectWithTime(string imagePath)
        {
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.RunWithTime(img, YoloConfiguration);
            }
        }
        public YoloResult<DetectionResult> RunDetectWithTime(Mat img)
        {
            return _yoloDetect.RunWithTime(img, YoloConfiguration);
        }
        #endregion

        #region Asynchronous

        public IYoloAsync CreateAsyncChannel()
        {
            return new YoloChannelAsync(YoloConfiguration, _yoloDetect.GetYoloDetectAsync());
        }


        #endregion


        #region BatchDetect

        public DetectionBatchResult[] RunBatchDetect(string imgDir)
        {
            return RunBatchDetect(imgDir, null, null);
        }
        public DetectionBatchResult[] RunBatchDetect(string imgDir, Action<DetectionBatchResult> receiveAction)
        {
            return RunBatchDetect(imgDir, null, receiveAction);
        }
        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback processCallback)
        {
            return RunBatchDetect(imgDir, processCallback, null);
        }

        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration.BatchPoolSize, YoloConfiguration);

            return _yoloDetect.BatchDetect(files, processCallback, receiveAction, YoloConfiguration);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration.BatchPoolSize, YoloConfiguration);

            return await _yoloDetect.BatchDetectAsync(files, null, null, YoloConfiguration);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback processCallback)
        {
            return await RunBatchDetectAsync(imgDir, processCallback, null);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, Action<DetectionBatchResult> receiveAction)
        {
            return await RunBatchDetectAsync(imgDir, null, receiveAction);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration.BatchPoolSize, YoloConfiguration);

            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction, YoloConfiguration);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloDetect.BatchDetectAsync(files, null, null, YoloConfiguration);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback processCallback)
        {
            return await RunBatchDetectAsync(images, processCallback, null);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, Action<DetectionBatchResult> receiveAction)
        {
            return await RunBatchDetectAsync(images, null, receiveAction);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction, YoloConfiguration);
        }


        public DetectionBatchResult[] RunBatchDetect(List<string> images)
        {
            return RunBatchDetect(images, null, null);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, Action<DetectionBatchResult> receiveAction)
        {
            return RunBatchDetect(images, null, receiveAction);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback processCallback)
        {
            return RunBatchDetect(images, processCallback, null);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchDetect(files, processCallback, receiveAction, YoloConfiguration);
        }



        private void YoloDetect_BatchDetectItemCompleted(object? sender, DetectionBatchResult e)
        {
            BatchDetectItemCompleted?.Invoke(sender, e);
        }

        #endregion

        #region DrawDetections


        public void DrawDetections(Mat inputImage, List<DetectionResult> list)
        {
            _yoloDetect.DrawDetections(inputImage, list);
        }
        public void DrawDetectionsAndSave(Mat inputImage, List<DetectionResult> list, string saveFileName)
        {
            _yoloDetect.DrawDetections(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }


        public void DrawDetections(string inputImage, List<DetectionResult> list)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloDetect.DrawDetections(img, list);
        }
        public void DrawDetectionsAndSave(string inputImage, List<DetectionResult> list, string saveFileName)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloDetect.DrawDetections(img, list);
            Cv2.ImWrite(saveFileName, img);
        }

        #endregion



        #region Validation


        #endregion


        public void Dispose()
        {
            _yoloDetect?.Dispose();
        }

    }
}
