using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;

namespace YoloSharpOnnx.Inference
{
    internal class YoloAsync : IYoloAsync
    {
        private Lazy<IYoloTaskAsync<DetectionResult, DetectAsyncResult>> _yoloDetectAsync;
        private Lazy<IYoloTaskAsync<ClsResult, ClsAsyncResult>> _yoloClsAsync;
        private Lazy<IYoloTaskAsync<SegResult, SegAsyncResult>> _yoloSegAsync;
        private Lazy<IYoloTaskAsync<PoseResult, PoseAsyncResult>> _yoloPoseAsync;
        private Lazy<IYoloTaskAsync<ObbResult, ObbAsyncResult>> _yoloObbAsync;

        private IYoloDetectCore<DetectionResult, DetectionBatchResult> _yoloDetect;
        private IYoloClassify _yoloClassify;
        private IYoloSegment _yoloSegment;
        private IYoloDetectCore<PoseResult, PoseBatchResult> _yoloPose;
        private IYoloDetectCore<ObbResult, ObbBatchResult> _yoloObb;

        private YoloConfig _yoloConfig;
        private ModelType _currentModelType;

        public YoloAsync(IYoloDetectCore<DetectionResult, DetectionBatchResult> yoloDetect, IYoloClassify yoloClassify, IYoloSegment yoloSegment, IYoloDetectCore<PoseResult, PoseBatchResult> yoloPose, IYoloDetectCore<ObbResult, ObbBatchResult> yoloObb, YoloConfig yoloConfig, ModelType modelType)
        {
            _currentModelType = modelType;
            _yoloDetect = yoloDetect;
            _yoloClassify = yoloClassify;
            _yoloSegment = yoloSegment;
            _yoloPose = yoloPose;
            _yoloObb = yoloObb;

            _yoloConfig = yoloConfig;

            if (_yoloDetect != null)
            {
                _yoloDetectAsync = new Lazy<IYoloTaskAsync<DetectionResult, DetectAsyncResult>>(() => new YoloChannelAsync<DetectionResult, PreDetectResultBatch, DetectAsyncResult>(_yoloConfig, _yoloDetect.GetYoloProcessAsync()));
            }
            if (_yoloClassify != null)
            {
                _yoloClsAsync = new Lazy<IYoloTaskAsync<ClsResult, ClsAsyncResult>>(() => new YoloChannelAsync<ClsResult, PreClsResultBatch, ClsAsyncResult>(_yoloConfig, _yoloClassify.GetYoloProcessAsync()));
            }
            if (_yoloSegment != null)
            {
                _yoloSegAsync = new Lazy<IYoloTaskAsync<SegResult, SegAsyncResult>>(() => new YoloChannelAsync<SegResult, PreDetectResultBatch, SegAsyncResult>(_yoloConfig, _yoloSegment.GetYoloProcessAsync()));
            }
            if (_yoloPose != null)
            {
                _yoloPoseAsync = new Lazy<IYoloTaskAsync<PoseResult, PoseAsyncResult>>(() => new YoloChannelAsync<PoseResult, PreDetectResultBatch, PoseAsyncResult>(_yoloConfig, _yoloPose.GetYoloProcessAsync()));
            }
            if (_yoloObb != null)
            {
                _yoloObbAsync = new Lazy<IYoloTaskAsync<ObbResult, ObbAsyncResult>>(() => new YoloChannelAsync<ObbResult, PreDetectResultBatch, ObbAsyncResult>(_yoloConfig, _yoloObb.GetYoloProcessAsync()));
            }

        }
        public void Dispose()
        {
            DisposeLazy(_yoloClsAsync);
            _yoloClsAsync = null;
            DisposeLazy(_yoloDetectAsync);
            _yoloDetectAsync = null;

            DisposeLazy(_yoloSegAsync);
            _yoloSegAsync = null;
            DisposeLazy(_yoloPoseAsync);
            _yoloPoseAsync = null;

            DisposeLazy(_yoloObbAsync);
            _yoloObbAsync = null;
        }

        private void DisposeLazy<T>(Lazy<T> lazy) where T : IDisposable
        {
            if (lazy != null && lazy.IsValueCreated)
            {
                lazy.Value.Dispose();
            }
        }

        public Task<List<DetectionResult>> RunDetectAsync(string inputImage)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            return _yoloDetectAsync.Value.RunAsync(inputImage);
        }

        public Task<List<DetectionResult>> RunDetectAsync(Mat img)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            return _yoloDetectAsync.Value.RunAsync(img);
        }
        public Task RunDetectAsync(Mat img, Guid guid, IBatchProcessCallback<DetectAsyncResult> callback, Action<DetectAsyncResult> receiveAction)
        {
            YoloValidation.ValidationDetectModelType(_currentModelType);
            return _yoloDetectAsync.Value.RunAsync(img, guid, callback, receiveAction);
        }

        public Task<List<ClsResult>> RunClassifyAsync(string inputImage)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            return _yoloClsAsync.Value.RunAsync(inputImage);
        }

        public Task<List<ClsResult>> RunClassifyAsync(Mat img)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            return _yoloClsAsync.Value.RunAsync(img);
        }

        public Task RunClassifyAsync(Mat img, Guid guid, IBatchProcessCallback<ClsAsyncResult> callback, Action<ClsAsyncResult> receiveAction)
        {
            YoloValidation.ValidationClsModelType(_currentModelType);
            return _yoloClsAsync.Value.RunAsync(img, guid, callback, receiveAction);
        }

        public Task<List<SegResult>> RunSegmentAsync(string inputImage)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            return _yoloSegAsync.Value.RunAsync(inputImage);
        }

        public Task<List<SegResult>> RunSegmentAsync(Mat img)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            return _yoloSegAsync.Value.RunAsync(img);
        }

        public Task RunSegmentAsync(Mat img, Guid guid, IBatchProcessCallback<SegAsyncResult> callback, Action<SegAsyncResult> receiveAction)
        {
            YoloValidation.ValidationSegModelType(_currentModelType);
            return _yoloSegAsync.Value.RunAsync(img, guid, callback, receiveAction);
        }

        public Task<List<PoseResult>> RunPoseAsync(string inputImage)
        {
            YoloValidation.ValidationPoseModelType(_currentModelType);
            return _yoloPoseAsync.Value.RunAsync(inputImage);
        }

        public Task<List<PoseResult>> RunPoseAsync(Mat img)
        {
            YoloValidation.ValidationPoseModelType(_currentModelType);
            return _yoloPoseAsync.Value.RunAsync(img);
        }

        public Task RunPoseAsync(Mat img, Guid guid, IBatchProcessCallback<PoseAsyncResult> callback, Action<PoseAsyncResult> receiveAction)
        {
            YoloValidation.ValidationPoseModelType(_currentModelType);
            return _yoloPoseAsync.Value.RunAsync(img, guid, callback, receiveAction);
        }

        public Task<List<ObbResult>> RunObbDetectAsync(string inputImage)
        {
            YoloValidation.ValidationObbModelType(_currentModelType);
            return _yoloObbAsync.Value.RunAsync(inputImage);
        }

        public Task<List<ObbResult>> RunObbDetectAsync(Mat img)
        {
            YoloValidation.ValidationObbModelType(_currentModelType);
            return _yoloObbAsync.Value.RunAsync(img);
        }

        public Task RunObbDetectAsync(Mat img, Guid guid, IBatchProcessCallback<ObbAsyncResult> callback, Action<ObbAsyncResult> receiveAction)
        {
            YoloValidation.ValidationObbModelType(_currentModelType);
            return _yoloObbAsync.Value.RunAsync(img, guid, callback, receiveAction);
        }

        public Task CompleteAndCloseAsyncChannel()
        {
           if(_yoloDetectAsync != null)
           {
               return _yoloDetectAsync.Value.CompleteAndCloseAsyncChannel();
           }
           else if(_yoloClsAsync != null)
           {
               return _yoloClsAsync.Value.CompleteAndCloseAsyncChannel();
           }
           else if(_yoloSegAsync != null)
           {
               return _yoloSegAsync.Value.CompleteAndCloseAsyncChannel();
           }
           else if(_yoloPoseAsync != null)
           {
               return _yoloPoseAsync.Value.CompleteAndCloseAsyncChannel();
           }
           else if(_yoloObbAsync != null)
           {
               return _yoloObbAsync.Value.CompleteAndCloseAsyncChannel();
           }
           throw new NotImplementedException();
        }
    }
}
