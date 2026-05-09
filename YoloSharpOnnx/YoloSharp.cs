using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;
        private IYoloClassify _yoloClassify;
        private bool disposedValue;


        public YoloConfig YoloConfiguration { get; set; }

        #region Constructor


        public YoloSharp(IExecutionProvider executionProvider) : this(new YoloConfig(), executionProvider)
        {

        }
        public YoloSharp(YoloConfig yoloConfig, IExecutionProvider executionProvider)
        {
            YoloConfiguration = yoloConfig;
            executionProvider.SetYoloConfiguration(yoloConfig);
            _yoloDetect = executionProvider.CreateYoloDetect();
            _yoloClassify = executionProvider.CreateYoloClassify();
          
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
                return _yoloDetect.Run(img);
            }
        }

        public List<DetectionResult> RunDetect(Mat img)
        {
            return _yoloDetect.Run(img);
        }

        public YoloResult<DetectionResult> RunDetectWithTime(string imagePath)
        {
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.RunWithTime(img);
            }
        }
        public YoloResult<DetectionResult> RunDetectWithTime(Mat img)
        {
            return _yoloDetect.RunWithTime(img);
        }


        public List<ClsResult> RunClassify(string imagePath)
        {
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.Run(img);
            }
        }
        public List<ClsResult> RunClassify(Mat img)
        {
            return _yoloClassify.Run(img);
        }

        public YoloResult<ClsResult> RunClassifyWithTime(string imagePath)
        {
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.RunWithTime(img);
            }
        }
        public YoloResult<ClsResult> RunClassifyWithTime(Mat img)
        {
            return _yoloClassify.RunWithTime(img);
        }

        #endregion

        #region Asynchronous

        public IYoloAsync CreateAsyncChannel()
        {
            return new YoloChannelAsync(YoloConfiguration, _yoloDetect.GetYoloDetectAsync(), _yoloDetect.GetRunBatch());
        }


        #endregion


        #region BatchDetect

        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloDetect.BatchDetect(files, processCallback, receiveAction);
        }
        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchDetect(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloDetect.BatchDetectAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> images)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchDetectForeachAsync(files);
        }



        #endregion


        #region BatchCls

        public ClsBatchResult[] RunBatchCls(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloClassify.BatchCls(files, processCallback, receiveAction);
        }
        public ClsBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchCls(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return await _yoloClassify.BatchClsAsync(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchClsAsync(List<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloClassify.BatchClsAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(List<string> images)
        {
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchClsForeachAsync(files);
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


        public void DrawClassification(Mat inputImage, List<ClsResult> list)
        {
            _yoloClassify.DrawClassification(inputImage, list);
        }
        public void DrawClassificationAndSave(Mat inputImage, List<ClsResult> list, string saveFileName)
        {
            _yoloClassify.DrawClassification(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }


        public void DrawClassification(string inputImage, List<ClsResult> list)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloClassify.DrawClassification(img, list);
        }
        public void DrawClassificationAndSave(string inputImage, List<ClsResult> list, string saveFileName)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloClassify.DrawClassification(img, list);
            Cv2.ImWrite(saveFileName, img);
        }

        #endregion



        #region Validation


        #endregion


        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                _yoloDetect?.Dispose();
                _yoloClassify?.Dispose();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloSharp()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

    }
}
