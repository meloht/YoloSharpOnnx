using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectOrtVal : YoloDetectBase
    {
        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
           : base(session, options, postprocess, preprocess, onnxModel, config)
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

        protected override List<DetectionResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            _matPool.Return(preResult.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0, preResult.PreResult, _config);

            return result;
        }
    }
}
