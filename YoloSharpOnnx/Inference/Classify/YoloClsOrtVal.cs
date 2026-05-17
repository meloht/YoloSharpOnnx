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
        }
        public List<ClsResult> Run(Mat inputImage)
        {
            // 预处理图像
            _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var ortValue = outputs[0];

            // 后处理
            return _postprocess.PostProcess(ortValue);
        }


        public YoloResult<ClsResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();

            // 预处理图像
            PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var ortValue = outputs[0];
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(ortValue, speed);
            return new YoloResult<ClsResult>(res, speed);
        }


        protected override List<ClsResult> RunBatchInfer(PreClsResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                using var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
                using var output0 = outputs[0];
                _matPool.Return(preResult.Data);
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcess(output0);
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
                // 执行推理
                var outputs = _session.Run(_runOptions, _session.InputNames, [item.Data.InputOrtValue], _session.OutputNames);
                _matPool.Return(item.Data);
                isReturn = true;

                // 后处理
                Task.Run(() =>
                {
                    BatchPostProcess(batchResults, idx, outputs[0], item, startTime, processCallback, receiveAction);
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
