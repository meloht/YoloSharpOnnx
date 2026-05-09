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
    public class YoloClsOrtVal : YoloClsBase, IYoloClassify
    {
        private bool disposedValue;

        public YoloClsOrtVal(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
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
            return RunCore(inputImage);
        }

        public YoloResult<ClsResult> RunWithTime(Mat inputImage)
        {
            return RunWithTimeCore(inputImage);
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

        protected override OrtValue RunInference()
        {
            var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            return outputs[0];
        }

        protected override void AfterInference(OrtValue ortValue)
        {
            ortValue.Dispose();
            ortValue = null;
        }

        protected override List<ClsResult> RunBatchInfer(PreClsResultBatch preResult)
        {
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            _matPool.Return(preResult.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0);

            return result;
        }
    }
}
