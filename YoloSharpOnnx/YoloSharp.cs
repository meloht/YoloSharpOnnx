using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.Utils;


namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IYoloDetectCore<DetectionResult, DetectionBatchResult> _yoloDetect;
        private IYoloClassify _yoloClassify;
        private IYoloSegment _yoloSegment;
        private IYoloDetectCore<PoseResult, PoseBatchResult> _yoloPose;
        private IYoloDetectCore<ObbResult, ObbBatchResult> _yoloObb;
        private YoloTaskType _currentTaskType;

        private bool disposedValue;


        public YoloConfig YoloConfiguration { get; set; }
        public YoloTaskType CurrentTaskType { get { return _currentTaskType; } }

        #region Constructor


        public YoloSharp(ExecutionProvider executionProvider) : this(new YoloConfig(), executionProvider)
        {

        }
        public YoloSharp(YoloConfig yoloConfig, ExecutionProvider executionProvider)
        {
            YoloConfiguration = yoloConfig;
            executionProvider.SetYoloConfiguration(yoloConfig);
            _yoloDetect = executionProvider.CreateYoloDetect();
            _yoloClassify = executionProvider.CreateYoloClassify();
            _yoloSegment = executionProvider.CreateYoloSegment();
            _yoloPose = executionProvider.CreateYoloPose();
            _yoloObb = executionProvider.CreateYoloObb();
            _currentTaskType = executionProvider.CurrentTaskType;

        }

        public YoloSharp(float confidence, float iou, ExecutionProvider executionProvider)
            : this(confidence, iou, InterpolationFlags.Linear, executionProvider)
        {

        }

        public YoloSharp(float confidence, float iou, InterpolationFlags resizeAlgorithm, ExecutionProvider executionProvider)
            : this(new YoloConfig(confidence, iou, resizeAlgorithm), executionProvider)
        {

        }



        #endregion

        #region Synchronous detect

        public List<DetectionResult> RunDetect(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.Run(img);
            }
        }

        public List<DetectionResult> RunDetect(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            return _yoloDetect.Run(img);
        }

        public YoloResult<DetectionResult> RunDetectWithTime(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloDetect.RunWithTime(img);
            }
        }
        public YoloResult<DetectionResult> RunDetectWithTime(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            return _yoloDetect.RunWithTime(img);
        }

        #endregion

        #region Synchronous obb
        public List<ObbResult> RunObbDetect(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloObb.Run(img);
            }
        }

        public List<ObbResult> RunObbDetect(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            return _yoloObb.Run(img);
        }

        public YoloResult<ObbResult> RunObbDetectWithTime(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloObb.RunWithTime(img);
            }
        }
        public YoloResult<ObbResult> RunObbDetectWithTime(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            return _yoloObb.RunWithTime(img);
        }
        #endregion

        #region Synchronous classify

        public List<ClsResult> RunClassify(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.Run(img);
            }
        }
        public List<ClsResult> RunClassify(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            return _yoloClassify.Run(img);
        }

        public YoloResult<ClsResult> RunClassifyWithTime(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloClassify.RunWithTime(img);
            }
        }
        public YoloResult<ClsResult> RunClassifyWithTime(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            return _yoloClassify.RunWithTime(img);
        }

        #endregion

        #region Synchronous segment
        public List<SegResult> RunSegment(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloSegment.Run(img);
            }
        }
        public List<SegResult> RunSegment(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            return _yoloSegment.Run(img);
        }
        public YoloResult<SegResult> RunSegmentWithTime(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloSegment.RunWithTime(img);
            }
        }
        public YoloResult<SegResult> RunSegmentWithTime(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            return _yoloSegment.RunWithTime(img);
        }

        #endregion

        #region Synchronous pose
        public List<PoseResult> RunPose(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloPose.Run(img);
            }
        }
        public List<PoseResult> RunPose(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            return _yoloPose.Run(img);
        }
        public YoloResult<PoseResult> RunPoseWithTime(string imagePath)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            YoloValidation.ValidationImagePath(imagePath, YoloConfiguration);
            using (Mat img = Cv2.ImRead(imagePath))
            {
                return _yoloPose.RunWithTime(img);
            }
        }
        public YoloResult<PoseResult> RunPoseWithTime(Mat img)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            return _yoloPose.RunWithTime(img);
        }

        #endregion

        #region Asynchronous


        public IYoloAsync CreateAsyncChannel()
        {
            ThrowIfDisposed();
            return new YoloAsync(_yoloDetect, _yoloClassify, _yoloSegment, _yoloPose, _yoloObb, YoloConfiguration, _currentTaskType);
        }
        #endregion

        #region BatchDetect

        public DetectionBatchResult[] RunBatchDetect(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloDetect.BatchRunPostSync(files, processCallback, receiveAction);
        }
        public DetectionBatchResult[] RunBatchDetect(IReadOnlyList<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchRunPostSync(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(string imgDir, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloDetect.BatchRunAsyncPostSync(files, processCallback, receiveAction);
        }


        public async Task<DetectionBatchResult[]> RunBatchDetectAsync(IReadOnlyList<string> images, IBatchProcessCallback<DetectionBatchResult> processCallback = null, Action<DetectionBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloDetect.BatchRunAsyncPostSync(files, processCallback, receiveAction);
        }

        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(string imgDir)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloDetect.BatchRunForeachSync(files);
        }

        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(IReadOnlyList<string> images)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationDetectModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloDetect.BatchRunForeachSync(files);
        }



        #endregion

        #region BatchCls

        public ClsBatchResult[] RunBatchCls(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloClassify.BatchRunPostSync(files, processCallback, receiveAction);
        }
        public ClsBatchResult[] RunBatchCls(IReadOnlyList<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchRunPostSync(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchClsAsync(string imgDir, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return await _yoloClassify.BatchRunAsyncPostSync(files, processCallback, receiveAction);
        }


        public async Task<ClsBatchResult[]> RunBatchClsAsync(IReadOnlyList<string> images, IBatchProcessCallback<ClsBatchResult> processCallback = null, Action<ClsBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloClassify.BatchRunAsyncPostSync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(string imgDir)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloClassify.BatchRunForeachSync(files);
        }
        public IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(IReadOnlyList<string> images)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationClsModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloClassify.BatchRunForeachSync(files);
        }



        #endregion

        #region BatchSegment

        public SegBatchResult[] RunBatchSegment(string imgDir, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloSegment.BatchRun(files, processCallback, receiveAction);
        }
        public SegBatchResult[] RunBatchSegment(IReadOnlyList<string> images, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloSegment.BatchRun(files, processCallback, receiveAction);
        }


        public async Task<SegBatchResult[]> RunBatchSegmentAsync(string imgDir, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloSegment.BatchRunAsync(files, processCallback, receiveAction);
        }

        public async Task<SegBatchResult[]> RunBatchSegmentAsync(IReadOnlyList<string> images, IBatchProcessCallback<SegBatchResult> processCallback = null, Action<SegBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloSegment.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<SegBatchResult> BatchSegmentForeachAsync(string imgDir)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloSegment.BatchRunForeachAsync(files);
        }
        public IAsyncEnumerable<SegBatchResult> BatchSegmentForeachAsync(IReadOnlyList<string> images)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationSegModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloSegment.BatchRunForeachAsync(files);
        }

        #endregion

        #region BatchPose
        public PoseBatchResult[] RunBatchPose(string imgDir, IBatchProcessCallback<PoseBatchResult> processCallback = null, Action<PoseBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloPose.BatchRun(files, processCallback, receiveAction);
        }
        public PoseBatchResult[] RunBatchPose(IReadOnlyList<string> images, IBatchProcessCallback<PoseBatchResult> processCallback = null, Action<PoseBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloPose.BatchRun(files, processCallback, receiveAction);
        }


        public async Task<PoseBatchResult[]> RunBatchPoseAsync(string imgDir, IBatchProcessCallback<PoseBatchResult> processCallback = null, Action<PoseBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloPose.BatchRunAsync(files, processCallback, receiveAction);
        }

        public async Task<PoseBatchResult[]> RunBatchPoseAsync(IReadOnlyList<string> images, IBatchProcessCallback<PoseBatchResult> processCallback = null, Action<PoseBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloPose.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<PoseBatchResult> BatchPoseForeachAsync(string imgDir)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloPose.BatchRunForeachAsync(files);
        }
        public IAsyncEnumerable<PoseBatchResult> BatchPoseForeachAsync(IReadOnlyList<string> images)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationPoseModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloPose.BatchRunForeachAsync(files);
        }

        #endregion

        #region BatchObb
        public ObbBatchResult[] RunBatchObbDetect(string imgDir, IBatchProcessCallback<ObbBatchResult> processCallback = null, Action<ObbBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return _yoloObb.BatchRun(files, processCallback, receiveAction);
        }
        public ObbBatchResult[] RunBatchObbDetect(IReadOnlyList<string> images, IBatchProcessCallback<ObbBatchResult> processCallback = null, Action<ObbBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloObb.BatchRun(files, processCallback, receiveAction);
        }

        public async Task<ObbBatchResult[]> RunBatchObbDetectAsync(string imgDir, IBatchProcessCallback<ObbBatchResult> processCallback = null, Action<ObbBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);

            return await _yoloObb.BatchRunAsync(files, processCallback, receiveAction);
        }

        public async Task<ObbBatchResult[]> RunBatchObbDetectAsync(IReadOnlyList<string> images, IBatchProcessCallback<ObbBatchResult> processCallback = null, Action<ObbBatchResult> receiveAction = null)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return await _yoloObb.BatchRunAsync(files, processCallback, receiveAction);
        }
        public IAsyncEnumerable<ObbBatchResult> BatchObbDetectForeachAsync(string imgDir)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloValidation.ValidationImageBatch(imgDir, YoloConfiguration);
            return _yoloObb.BatchRunForeachAsync(files);
        }
        public IAsyncEnumerable<ObbBatchResult> BatchObbDetectForeachAsync(IReadOnlyList<string> images)
        {
            ThrowIfDisposed();
            YoloValidation.ValidationObbModelType(_currentTaskType);
            var files = YoloUtils.GetFilesFromListPaths(images, YoloConfiguration.ImageExtsBatch);
            YoloValidation.ValidationImageListPath(files, YoloConfiguration);
            return _yoloObb.BatchRunForeachAsync(files);
        }
        #endregion

        #region DrawResults

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

        #region DrawClassification
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

        #region DrawSegment
        public void DrawSegment(Mat inputImage, List<SegResult> list)
        {
            _yoloSegment.DrawSegments(inputImage, list);
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

        #region DrawPose
        public void DrawPose(Mat inputImage, List<PoseResult> list)
        {
            _yoloPose.DrawDetections(inputImage, list);
        }

        public void DrawPoseAndSave(Mat inputImage, List<PoseResult> list, string saveFileName)
        {
            _yoloPose.DrawDetections(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }

        public void DrawPose(string inputImage, List<PoseResult> list)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloPose.DrawDetections(img, list);
        }
        public void DrawPoseAndSave(string inputImage, List<PoseResult> list, string saveFileName)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloPose.DrawDetections(img, list);
            Cv2.ImWrite(saveFileName, img);
        }
        #endregion

        #region DrawObb

        public void DrawObb(Mat inputImage, List<ObbResult> list)
        {
            _yoloObb.DrawDetections(inputImage, list);
        }

        public void DrawObbAndSave(Mat inputImage, List<ObbResult> list, string saveFileName)
        {
            _yoloObb.DrawDetections(inputImage, list);
            Cv2.ImWrite(saveFileName, inputImage);
        }

        public void DrawObb(string inputImage, List<ObbResult> list)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloObb.DrawDetections(img, list);
        }
        public void DrawObbAndSave(string inputImage, List<ObbResult> list, string saveFileName)
        {
            YoloValidation.ValidationImagePath(inputImage, YoloConfiguration);
            using Mat img = Cv2.ImRead(inputImage);
            _yoloObb.DrawDetections(img, list);
            Cv2.ImWrite(saveFileName, img);
        }

        #endregion

        #endregion

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(disposedValue, this);
        }
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
                _yoloPose?.Dispose();
                _yoloObb?.Dispose();
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
