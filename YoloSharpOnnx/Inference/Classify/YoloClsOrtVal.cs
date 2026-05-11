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
    public class YoloClsOrtVal : YoloClsBase
    {

        public YoloClsOrtVal(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
            Warmup();
        }

        protected override void DisposedSub()
        {

        }

        private void Warmup()
        {
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
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
