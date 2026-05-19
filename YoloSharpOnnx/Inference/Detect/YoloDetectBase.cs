using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx.Inference.Detect
{
    public abstract class YoloDetectBase : OnnxInferenceCore<DetectionResult, PreDetectResultBatch, DetectionBatchResult>, IYoloProcessAsync<PreDetectResultBatch, DetectionResult>
    {
        protected readonly IDetPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;
        private bool disposedValue;

        protected abstract void DisposedSub();
        protected abstract OrtValue RunInferenceBatch(PreDetectResultBatch preResult);

        public YoloDetectBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
        }

        protected PreDetectResult PreprocessTime(Mat inputImage, SpeedResult speed)
        {
            _stopwatch.Restart();

            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;

            return preRes;
        }


        protected List<DetectionResult> PostProcessTime(OrtValue output0, PreDetectResult preDetect, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var res = _postprocess.PostProcessSync(output0, preDetect);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return res;
        }
        protected Task BatchPostProcess(DetectionBatchResult[] batchResults, int idx, OrtValue output0, string imagePath, long startTime, PreDetectResult preDetect,
            IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            return Task.Run(() =>
             {
                 using (output0)
                 {
                     var result = _postprocess.PostProcessAsync(output0, preDetect);
                     batchResults[idx] = BuildBatchResult(imagePath, result, startTime);
                 }

                 _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);
             });

        }
        protected override DetectionBatchResult PostprocessChannel(InferModel inferModel)
        {
            try
            {
                using (inferModel.Output0)
                {
                    var res = _postprocess.PostProcessSync(inferModel.Output0, inferModel.PreDetectResult);
                    return BuildBatchResult(inferModel.ImagePath, res, inferModel.StartTime);
                }
            }
            finally
            {
                _inferModelPool.Value.Return(inferModel);
            }
        }

        protected override List<DetectionResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                using var output = RunInferenceBatch(preResult);
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcessSync(output, preResult.PreResult);
                return result;
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
                _preResultPool.Value.Return(preResult);
            }

        }
        protected override Task RunBatchInfer(DetectionBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime,
            IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var output = RunInferenceBatch(item);
                isReturn = true;
                // 后处理
                return BatchPostProcess(batchResults, idx, output, item.ImagePath, startTime, item.PreResult, processCallback, receiveAction);
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(item.Data);
                }
                _preResultPool.Value.Return(item);
            }
        }

        protected override InferModel RunInfer(PreDetectResultBatch preResult, long startTime)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var ortValue = RunInferenceBatch(preResult);
                isReturn = true;

                var data = _inferModelPool.Value.Rent();
                data.Initialize(ortValue, null, preResult.ImagePath, startTime, preResult.PreResult);
                return data;
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
                _preResultPool.Value.Return(preResult);
            }
        }

        protected override PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            var batchData = _preResultPool.Value.Rent();
            batchData.Initialize(preRes, imagePath, imageBatchData);
            return batchData;
        }

        private static DetectionBatchResult BuildBatchResult(string imagePath, List<DetectionResult> results, long timestamp)
        {
            return new DetectionBatchResult(imagePath, results, timestamp);
        }

        public List<DetectionResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }
        protected override DetectionBatchResult PostprocessModel(PreDetectResultBatch preResult, long startTime)
        {
            var res = BuildBatchResult(preResult.ImagePath, null, startTime);
            res.Results = RunBatchInfer(preResult);
            return res;
        }

        public IYoloProcessAsync<PreDetectResultBatch, DetectionResult> GetYoloProcessAsync()
        {
            return this;
        }

        public void DrawDetections(Mat inputImage, List<DetectionResult> list)
        {
            foreach (var item in list)
            {
                YoloUtils.DrawDetections(inputImage, item.Box, item.Confidence, item.ClassName, _onnxModel.ColorPalette[item.ClassId]);
            }
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
                DisposeCore();
                DisposedSub();
                _postprocess.Dispose();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloDetectBase()
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
