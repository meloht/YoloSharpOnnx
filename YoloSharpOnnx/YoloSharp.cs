using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;

        public event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

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
            _yoloDetect.BatchDetectItemCompleted += YoloDetect_BatchDetectItemCompleted;
        }

        public List<DetectionResult> RunDetect(string imagePath)
        {
            ValidationImagePath(imagePath);
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
            ValidationImagePath(imagePath);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.RunWithTime(img, YoloConfiguration);
            }
        }
        public YoloResult<DetectionResult> RunDetectWithTime(Mat img)
        {
            return _yoloDetect.RunWithTime(img, YoloConfiguration);
        }


        #region BatchDetect

        public DetectionBatchResult[] RunBatchDetect(string imgDir, int batchPoolSize = 20)
        {
            return RunBatchDetect(imgDir, null, null, batchPoolSize);
        }
        public DetectionBatchResult[] RunBatchDetect(string imgDir, Action<DetectionBatchResult> receiveAction, int batchPoolSize = 20)
        {
            return RunBatchDetect(imgDir, null, receiveAction, batchPoolSize);
        }
        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback processCallback, int batchPoolSize = 20)
        {
            return RunBatchDetect(imgDir, processCallback, null, batchPoolSize);
        }

        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null, int batchPoolSize = 20)
        {
            var files = ValidationImageBatch(imgDir, batchPoolSize);

            return _yoloDetect.BatchDetect(files, processCallback, receiveAction, batchPoolSize, YoloConfiguration);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, int batchPoolSize = 20)
        {
            var files = ValidationImageBatch(imgDir, batchPoolSize);

            return await _yoloDetect.BatchDetectAsync(files, null, null, batchPoolSize, YoloConfiguration);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback processCallback, int batchPoolSize = 20)
        {
            return await RunBatchDetectAsync(imgDir, processCallback, null, batchPoolSize);
        }

        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, Action<DetectionBatchResult> receiveAction, int batchPoolSize = 20)
        {
            return await RunBatchDetectAsync(imgDir, null, receiveAction, batchPoolSize);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null, int batchPoolSize = 20)
        {
            var files = ValidationImageBatch(imgDir, batchPoolSize);

            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction, batchPoolSize, YoloConfiguration);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, int batchPoolSize = 20)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            ValidationImageListPath(files);
            return await _yoloDetect.BatchDetectAsync(files, null, null, batchPoolSize, YoloConfiguration);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback processCallback, int batchPoolSize = 20)
        {
            return await RunBatchDetectAsync(images, processCallback, null, batchPoolSize);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, Action<DetectionBatchResult> receiveAction, int batchPoolSize = 20)
        {
            return await RunBatchDetectAsync(images, null, receiveAction, batchPoolSize);
        }
        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction=null, int batchPoolSize = 20)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            ValidationImageListPath(files);
            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction, batchPoolSize, YoloConfiguration);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, Action<DetectionBatchResult> receiveAction, int batchPoolSize = 20)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            ValidationImageListPath(files);
            return _yoloDetect.BatchDetect(files, null, receiveAction, batchPoolSize, YoloConfiguration);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback processCallback, int batchPoolSize = 20)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            ValidationImageListPath(files);
            return _yoloDetect.BatchDetect(files, processCallback, null, batchPoolSize, YoloConfiguration);
        }

        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback processCallback = null, Action<DetectionBatchResult> receiveAction = null, int batchPoolSize = 20)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            ValidationImageListPath(files);
            return _yoloDetect.BatchDetect(files, processCallback, receiveAction, batchPoolSize, YoloConfiguration);
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
            ValidationImagePath(inputImage);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloDetect.DrawDetections(img, list);
        }
        public void DrawDetectionsAndSave(string inputImage, List<DetectionResult> list, string saveFileName)
        {
            ValidationImagePath(inputImage);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloDetect.DrawDetections(img, list);
            Cv2.ImWrite(saveFileName, img);
        }

        #endregion



        #region Validation
        private void ValidationImagePath(string imagePath)
        {
            if (string.IsNullOrWhiteSpace(imagePath))
            {
                throw new ArgumentNullException($"imagePath is null or empty");
            }
            if (!File.Exists(imagePath))
            {
                throw new DirectoryNotFoundException($"{imagePath} file is not found");
            }
            string ext = Path.GetExtension(imagePath);
            if (YoloConfiguration.ImageExtsBatch.AsSpan().IndexOf(ext) == -1)
            {
                throw new ArgumentNullException($"imagePath is not a image for ext({string.Join(',', YoloConfiguration.ImageExtsBatch)})");
            }
        }

        private List<string> ValidationImageBatch(string imgDir, int batchSize)
        {
            if (string.IsNullOrWhiteSpace(imgDir))
            {
                throw new ArgumentNullException($"imgDir is null or empty");
            }
            if (!Directory.Exists(imgDir))
            {
                throw new DirectoryNotFoundException($"{imgDir} directory not found");
            }
            if (batchSize <= 0)
            {
                throw new ArgumentNullException("batchSize must be greater than zero");
            }

            var files = YoloUtils.GetFilesFromDirectory(imgDir, YoloConfiguration.ImageExtsBatch);
            if (files.Count == 0)
            {
                throw new ArgumentNullException($"there no any images in the directory for image ext({string.Join(',', YoloConfiguration.ImageExtsBatch)})");
            }
            return files;
        }

        private void ValidationImageListPath(List<string> list)
        {
            if (list == null || list.Count == 0)
            {
                throw new ArgumentNullException($"images is invalid for image ext({string.Join(',', YoloConfiguration.ImageExtsBatch)})");
            }

        }

        #endregion


        public void Dispose()
        {
            _yoloDetect?.Dispose();
        }

    }
}
