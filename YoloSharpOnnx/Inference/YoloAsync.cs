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
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;

namespace YoloSharpOnnx.Inference
{
    internal class YoloAsync : IYoloAsync
    {
        private Lazy<IYoloTaskAsync<DetectionResult>> _yoloDetectAsync;
        private Lazy<IYoloTaskAsync<ClsResult>> _yoloClsAsync;
        private Lazy<IYoloTaskAsync<SegResult>> _yoloSegAsync;
        private Lazy<IYoloTaskAsync<PoseResult>> _yoloPoseAsync;
        private Lazy<IYoloTaskAsync<ObbResult>> _yoloObbAsync;

        private IYoloDetect _yoloDetect;
        private IYoloClassify _yoloClassify;
        private IYoloSegment _yoloSegment;
        private IYoloPose _yoloPose;
        private IYoloObb _yoloObb;

        private YoloConfig _yoloConfig;
        private ModelType _currentModelType;

        public YoloAsync(IYoloDetect yoloDetect, IYoloClassify yoloClassify, IYoloSegment yoloSegment, IYoloPose yoloPose, IYoloObb yoloObb, YoloConfig yoloConfig, ModelType modelType)
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
                _yoloDetectAsync = new Lazy<IYoloTaskAsync<DetectionResult>>(() => new YoloChannelAsync<DetectionResult, PreDetectResultBatch, PreDetectChannelData>(_yoloConfig, _yoloDetect.GetYoloProcessAsync()));
            }
            if (_yoloClassify != null)
            {
                _yoloClsAsync = new Lazy<IYoloTaskAsync<ClsResult>>(() => new YoloChannelAsync<ClsResult, PreClsResultBatch, PreClsChannelData>(_yoloConfig, _yoloClassify.GetYoloProcessAsync()));
            }
            if (_yoloSegment != null)
            {
                _yoloSegAsync = new Lazy<IYoloTaskAsync<SegResult>>(() => new YoloChannelAsync<SegResult, PreDetectResultBatch, PreDetectChannelData>(_yoloConfig, _yoloSegment.GetYoloProcessAsync()));
            }
            if (_yoloPose != null)
            {
                _yoloPoseAsync = new Lazy<IYoloTaskAsync<PoseResult>>(() => new YoloChannelAsync<PoseResult, PreDetectResultBatch, PreDetectChannelData>(_yoloConfig, _yoloPose.GetYoloProcessAsync()));
            }
            if (_yoloObb != null)
            {
                _yoloObbAsync = new Lazy<IYoloTaskAsync<ObbResult>>(() => new YoloChannelAsync<ObbResult, PreDetectResultBatch, PreDetectChannelData>(_yoloConfig, _yoloObb.GetYoloProcessAsync()));
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

        public Task<List<ObbResult>> RunObbAsync(string inputImage)
        {
            YoloValidation.ValidationObbModelType(_currentModelType);
            return _yoloObbAsync.Value.RunAsync(inputImage);
        }

        public Task<List<ObbResult>> RunObbAsync(Mat img)
        {
            YoloValidation.ValidationObbModelType(_currentModelType);
            return _yoloObbAsync.Value.RunAsync(img);
        }
    }
}
