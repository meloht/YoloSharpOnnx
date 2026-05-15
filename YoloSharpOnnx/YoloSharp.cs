using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetect _yoloDetect;
        private IYoloClassify _yoloClassify;
        private IYoloSegment _yoloSegment;
        private ModelType _currentModelType;

        private bool disposedValue;


        public YoloConfig YoloConfiguration { get; set; }
        public ModelType CurrentModelType { get { return _currentModelType; } }

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
            _yoloSegment = executionProvider.CreateYoloSegment();
            _currentModelType = executionProvider.CurrentModelType;

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

        #region Synchronous detect

        public List<DetectionResult> RunDetect(string imagePath)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.Run(img);
            }
        }

        public List<DetectionResult> RunDetect(Mat img)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            return _yoloDetect.Run(img);
        }

        public YoloResult<DetectionResult> RunDetectWithTime(string imagePath)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.RunWithTime(img);
            }
        }
        public YoloResult<DetectionResult> RunDetectWithTime(Mat img)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            return _yoloDetect.RunWithTime(img);
        }

        #endregion


        #region Synchronous classify

        public List<ClsResult> RunClassify(string imagePath)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.Run(img);
            }
        }
        public List<ClsResult> RunClassify(Mat img)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            return _yoloClassify.Run(img);
        }

        public YoloResult<ClsResult> RunClassifyWithTime(string imagePath)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.RunWithTime(img);
            }
        }
        public YoloResult<ClsResult> RunClassifyWithTime(Mat img)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            return _yoloClassify.RunWithTime(img);
        }

        #endregion

        #region Synchronous segment
        public List<SegResult> RunSegment(string imagePath)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloSegment.Run(img);
            }
        }
        public List<SegResult> RunSegment(Mat img)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            return _yoloSegment.Run(img);
        }
        public YoloResult<SegResult> RunSegmentWithTime(string imagePath)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloSegment.RunWithTime(img);
            }
        }
        public YoloResult<SegResult> RunSegmentWithTime(Mat img)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            return _yoloSegment.RunWithTime(img);
        }

        #endregion


        #region Asynchronous


        public IYoloAsync CreateAsyncChannel()
        {
            return new YoloAsync(_yoloDetect, _yoloClassify, _yoloSegment, YoloConfiguration, _currentModelType);
        }
        #endregion


        #region BatchDetect

        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloDetect.BatchRun(files, processCallback, receiveAction);
        }
        public DetectionBatchResult[] RunBatchDetect(List<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchRun(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloDetect.BatchRunAsync(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(List<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloDetect.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> images)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchRunForeachAsync(files);
        }



        #endregion


        #region BatchCls

        public ClsBatchResult[] RunBatchCls(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloClassify.BatchRun(files, processCallback, receiveAction);
        }
        public ClsBatchResult[] RunBatchCls(List<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchRun(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchClsAsync(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return await _yoloClassify.BatchRunAsync(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchClsAsync(List<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloClassify.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(List<string> images)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchRunForeachAsync(files);
        }



        #endregion

        #region BatchSegment

        public SegBatchResult[] RunBatchSegment(string imgDir, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloSegment.BatchRun(files, processCallback, receiveAction);
        }
        public SegBatchResult[] RunBatchSegment(List<string> images, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloSegment.BatchRun(files, processCallback, receiveAction);
        }


        public async Task<SegBatchResult[]> RunBatchSegmentAsync(string imgDir, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloSegment.BatchRunAsync(files, processCallback, receiveAction);
        }


        public async Task<SegBatchResult[]> RunBatchSegmentAsync(List<string> images, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloSegment.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<SegBatchResult> BatchSegmentForeachAsync(List<string> images)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloSegment.BatchRunForeachAsync(files);
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


        public void DrawSegment(Mat inputImage, List<SegResult> list)
        {
            _yoloSegment.DrawSegments(inputImage, list);
        }
        public void DrawSegment(Mat inputImage, List<SegResult> list, string saveFileName)
        {
            _yoloSegment.DrawSegments(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }

        public void DrawSegmentAndSave(Mat inputImage, List<SegResult> list, string saveFileName)
        {
            _yoloSegment.DrawSegments(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }

        public void DrawSegment(string inputImage, List<SegResult> list)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloSegment.DrawSegments(img, list);
        }
        public void DrawSegmentAndSave(string inputImage, List<SegResult> list, string saveFileName)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloSegment.DrawSegments(img, list);
            Cv2.ImWrite(saveFileName, img);
        }

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
                _yoloSegment?.Dispose();
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
