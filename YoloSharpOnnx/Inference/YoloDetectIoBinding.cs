using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class YoloDetectIoBinding : YoloDetectBase, IYoloDetect
    {
        private OrtIoBinding _binding;
        private OrtValue _inputOrtValue;
        private readonly OrtSafeMemoryHandle _inputNativeAllocation;

        public YoloDetectIoBinding(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
          : base(session, options, postprocess, onnxModel)
        {
            var inputSizeInBytes = _onnxModel.InputShapeSize * sizeof(float);
            nint allocPtrIn = Marshal.AllocHGlobal((int)inputSizeInBytes);
            _inputNativeAllocation = new OrtSafeMemoryHandle(allocPtrIn);

            _binding = _session.CreateIoBinding();

            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
                _onnxModel.InputShape, _inputNativeAllocation.Handle, inputSizeInBytes);

        }


        private void PopulateNativeBuffer<T>(IntPtr buffer, T[] elements)
        {
            Span<T> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<T>(buffer.ToPointer(), elements.Length);
            }
            elements.CopyTo(bufferSpan);
        }
        public void Dispose()
        {
            _session.Dispose();
            _binding.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
            _inputOrtValue.Dispose();
            _inputNativeAllocation.Dispose();
        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            // 预处理图像
            var preRes = Preprocess(inputImage, _inputBuffer, yoloConfig.ResizeAlgorithm);
            PopulateNativeBuffer(_inputNativeAllocation.Handle, _inputBuffer);
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            using var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            using var output = results[0];
            // 后处理
            var result = _postprocess.PostProcess(output, preRes, yoloConfig);
            return result;

        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfiguration yoloConfig)
        {
            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = Preprocess(inputImage, yoloConfig.ResizeAlgorithm);
            PopulateNativeBuffer(_inputNativeAllocation.Handle, _inputBuffer);
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();


            // 执行推理
            using var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            using var output = results[0];

            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 后处理
            var res = _postprocess.PostProcess(output, preRes, yoloConfig);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }
    }
}
