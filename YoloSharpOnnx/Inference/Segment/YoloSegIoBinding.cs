using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class YoloSegIoBinding : YoloSegBase, IYoloSegment
    {
        private readonly OrtIoBinding _binding;
        private readonly OrtValue _outputOrtValue0;
        private readonly OrtValue _outputOrtValue1;
        private readonly FixedBuffer _outputFixedBuffer0;
        private readonly FixedBuffer _outputFixedBuffer1;

        public YoloSegIoBinding(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
            _binding = _session.CreateIoBinding();

            _outputFixedBuffer0 = new FixedBuffer(_onnxModel.OutputShapeSize0);
            _outputFixedBuffer1 = new FixedBuffer(_onnxModel.OutputShapeSize1);

            _outputOrtValue0 = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            _onnxModel.OutputShape0, _outputFixedBuffer0.Address, _onnxModel.OutputSizeInBytes0);

            _outputOrtValue1 = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            _onnxModel.OutputShape1, _outputFixedBuffer1.Address, _onnxModel.OutputSizeInBytes1);

            RunInference();
        }

        protected override void DisposedSub()
        {
            _binding.Dispose();
            _outputOrtValue0.Dispose();
            _outputOrtValue1.Dispose();
            _outputFixedBuffer0.Dispose();
            _outputFixedBuffer1.Dispose();
        }

        private void RunInference()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName0, _outputOrtValue0);
            _binding.BindOutput(_onnxModel.OutputName1, _outputOrtValue1);
            _binding.SynchronizeBoundInputs();

            //_binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            //_binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

            //using var results = _session.RunWithBoundResults(_runOptions, _binding);
            //_binding.SynchronizeBoundOutputs();
            //using var output = results[0];

        }

        public List<SegResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage,_resizedImg, _inputFixedBuffer);

            // 执行推理
            RunInference();

            // 后处理
            return _postprocess.PostProcessSync(_outputOrtValue0, _outputOrtValue1, preRes);
        }

        public YoloResult<SegResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();
            // 预处理图像
            var preRes = PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            RunInference();
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(_outputOrtValue0, _outputOrtValue1, preRes, speed);
            return new YoloResult<SegResult>(res, speed);
        }

        protected override IDisposableReadOnlyCollection<OrtValue> RunInferenceBatch(PreDetectResultBatch preResult)
        {
            _binding.BindInput(_onnxModel.InputName, preResult.Data.InputOrtValue);
            _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            _matPool.Return(preResult.Data);
            return results;
        }
       


    }
}
