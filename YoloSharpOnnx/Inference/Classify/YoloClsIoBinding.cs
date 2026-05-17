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
    public class YoloClsIoBinding : YoloClsBase, IYoloClassify
    {
        private readonly OrtIoBinding _binding;
        private readonly OrtValue _outputOrtValue;
        private readonly FixedBuffer _outputFixedBuffer;
        public YoloClsIoBinding(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config, postprocess, preprocess)
        {
            _binding = _session.CreateIoBinding();
            _outputFixedBuffer = new FixedBuffer(_onnxModel.OutputShapeSize0);
            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            _onnxModel.OutputShape0, _outputFixedBuffer.Address, _onnxModel.OutputSizeInBytes0);

            RunInference();
        }

        protected override void DisposedSub()
        {
            _binding.Dispose();
            _outputOrtValue.Dispose();
            _outputFixedBuffer.Dispose();
        }

        private  void RunInference()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName0, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

        }
        public List<ClsResult> Run(Mat inputImage)
        {
            // 预处理图像
            _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);
            // 执行推理
            RunInference();
            // 后处理
            return _postprocess.PostProcess(_outputOrtValue);
        }

        public YoloResult<ClsResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();

            // 预处理图像
            PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            RunInference();
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(_outputOrtValue, speed);
            return new YoloResult<ClsResult>(res, speed);
        }

        protected override List<ClsResult> RunBatchInfer(PreClsResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                _binding.BindInput(_onnxModel.InputName, preResult.Data.InputOrtValue);
                _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
                _binding.SynchronizeBoundInputs();

                // 执行推理
                using var results = _session.RunWithBoundResults(_runOptions, _binding);
                _binding.SynchronizeBoundOutputs();
                using var output = results[0];
                _matPool.Return(preResult.Data);
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcess(output);

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

        protected override void RunBatchInfer(ClsBatchResult[] batchResults, int idx, PreClsResultBatch item, long startTime, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                _binding.BindInput(_onnxModel.InputName, item.Data.InputOrtValue);
                _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
                _binding.SynchronizeBoundInputs();

                // 执行推理
                var results = _session.RunWithBoundResults(_runOptions, _binding);
                _binding.SynchronizeBoundOutputs();
               
                _matPool.Return(item.Data);
                isReturn = true;

                // 后处理
                Task.Run(() =>
                {
                    BatchPostProcess(batchResults, idx, results[0], item, startTime, processCallback, receiveAction);
                });

            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(item.Data);
                }
            }
        }
    }
}
