using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectIoBinding : YoloDetectBase
    {
        private OrtIoBinding _binding;
        protected OrtValue _outputOrtValue;
       

        public YoloDetectIoBinding(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
          : base(session, options, postprocess, preprocess, onnxModel, config)
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

            //_binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            //_binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

            //using var results = _session.RunWithBoundResults(_runOptions, _binding);
            //_binding.SynchronizeBoundOutputs();
            //using var output = results[0];

            return _outputOrtValue;
        }

        protected override void AfterInference(OrtValue ortValue)
        {
           
        }

        protected override List<DetectionResult> RunBatchInfer(PreDetectResultBatch preResult)
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
            var result = _postprocess.PostProcess(output, preResult.PreResult, _config);

            return result;
        }


    }
}
