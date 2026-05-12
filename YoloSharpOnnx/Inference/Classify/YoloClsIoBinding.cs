using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
    public class YoloClsIoBinding : YoloClsBase
    {

        private OrtIoBinding _binding;
        protected OrtValue _outputOrtValue;
        public YoloClsIoBinding(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
            _binding = _session.CreateIoBinding();

            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            _onnxModel.OutputShape, _outputFixedBuffer.Address, _onnxModel.OutputSizeInBytes);

            Warmup();
        }

        protected override void DisposedSub()
        {
            _binding.Dispose();
            _outputOrtValue.Dispose();
        }

        private void Warmup()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName0, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
        }
       

        protected override OrtValue RunInference()
        {

            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName0, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

            return _outputOrtValue;

        }

        protected override void AfterInference(OrtValue ortValue)
        {

        }

        protected override List<ClsResult> RunBatchInfer(PreClsResultBatch preResult)
        {

            _binding.BindInput(_onnxModel.InputName, preResult.Data.InputOrtValue);
            _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            using var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            using var output = results[0];
            _matPool.Return(preResult.Data);
            // 后处理
            var result = _postprocess.PostProcess(output);

            return result;
        }
    }
}
