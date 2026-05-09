using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public class YoloClsOrtVal : YoloClsBase, IYoloClassify, IBatchProcess<ClsResult, PreClsResultBatch, ClsBatchResult>
    {
        private bool disposedValue;

        public YoloClsOrtVal(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
            InitBatchProcess(this);
            Warmup();
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
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloClsOrtVal()
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

        private void Warmup()
        {
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
        }
        public List<ClsResult> Run(Mat inputImage)
        {
            // 预处理图像
            _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = _postprocess.PostProcess(output0);
            return result;
        }

        public YoloResult<ClsResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
             _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 后处理
            var res = _postprocess.PostProcess(output0);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<ClsResult>(res, speed);
        }

        public PreClsResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            _preprocess.PreprocessImage(inputImage, imageBatchData.ResizedImg, imageBatchData.FixedBuffer, _config.ResizeAlgorithm);
            return new PreClsResultBatch(imagePath, imageBatchData);
        }

        public ClsBatchResult BuildBatchResult(PreClsResultBatch batchPreResult, List<ClsResult> results, long timestamp)
        {
            return new ClsBatchResult(batchPreResult.ImagePath, results, timestamp);
        }

        public List<ClsResult> RunBatch(PreClsResultBatch preResult)
        {
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            _matPool.Return(preResult.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0);

            return result;
        }

        public ClsBatchResult[] BatchCls(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            return BatchDetectBase(listImg, processCallback, receiveAction);
        }

        public async Task<ClsBatchResult[]> BatchClsAsync(List<string> listImg, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            return await BatchDetectBaseAsync(listImg, processCallback, receiveAction);
        }

        public IAsyncEnumerable<ClsBatchResult> BatchClsForeachAsync(List<string> listImg)
        {
            return BatchDetectBaseForeachAsync(listImg);
        }
    }
}
