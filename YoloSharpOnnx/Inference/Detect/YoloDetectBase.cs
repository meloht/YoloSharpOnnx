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
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx.Inference.Detect
{
    public abstract class YoloDetectBase : OnnxInferenceCore<DetectionResult, PreDetectResultBatch, DetectionBatchResult>,
        IBatchProcess<DetectionResult, PreDetectResultBatch, DetectionBatchResult>, IYoloProcessAsync<PreDetectResultBatch>
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
            InitBatchProcess(this);
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
        protected Task BatchPostProcess(DetectionBatchResult[] batchResults, int idx, OrtValue output0, PreDetectResultBatch item, long startTime, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            return Task.Run(() =>
             {
                 using (output0)
                 {
                     var result = _postprocess.PostProcessAsync(output0, item.PreResult);
                     batchResults[idx] = BuildBatchResult(item, result, startTime);
                 }

                 _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);
             });

        }
        protected override DetectionBatchResult PostprocessChannel(InferModel<PreDetectResultBatch> inferModel)
        {
            using (inferModel.Output0)
            {
                var res = _postprocess.PostProcessSync(inferModel.Output0, inferModel.TBatchPreResult.PreResult);
                return BuildBatchResult(inferModel.TBatchPreResult, res, inferModel.StartTime);
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
            }

        }
        protected override Task RunBatchInfer(DetectionBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var output = RunInferenceBatch(item);
                isReturn = true;
                // 后处理
                return BatchPostProcess(batchResults, idx, output, item, startTime, processCallback, receiveAction);
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(item.Data);
                }
            }
        }

        protected override InferModel<PreDetectResultBatch> RunInfer(PreDetectResultBatch preResult, long startTime)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var ortValue = RunInferenceBatch(preResult);
                isReturn = true;

                return new InferModel<PreDetectResultBatch>(ortValue, null, preResult, startTime);
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
            }
        }

        public PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            return new PreDetectResultBatch(preRes, imagePath, imageBatchData);
        }

        public DetectionBatchResult BuildBatchResult(PreDetectResultBatch batchPreResult, List<DetectionResult> results, long timestamp)
        {
            return new DetectionBatchResult(batchPreResult.ImagePath, results, timestamp);
        }

        public List<DetectionResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        public IYoloProcessAsync<PreDetectResultBatch> GetYoloProcessAsync()
        {
            return this;
        }
        public IRunBatch<DetectionResult, PreDetectResultBatch> GetRunBatch()
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
