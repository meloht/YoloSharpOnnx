using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx.Inference.Segment
{
    public abstract class YoloSegBase : OnnxInferenceCore<SegResult, PreDetectResultBatch, SegBatchResult>, IYoloProcessAsync<PreDetectResultBatch, SegResult>
    {
        protected readonly ISegPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;
        private bool disposedValue;

        protected abstract void DisposedSub();
        protected abstract IDisposableReadOnlyCollection<OrtValue> RunInferenceBatch(PreDetectResultBatch preResult);

        protected YoloSegBase(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
        }

        private static SegBatchResult BuildBatchResult(string imagePath, List<SegResult> results, long timestamp)
        {
            return new SegBatchResult(imagePath, results, timestamp);
        }

        protected override PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            var data = _preResultPool.Value.Rent();
            data.Initialize(preRes, imagePath, imageBatchData);
            return data;
        }

        public List<SegResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        protected override SegBatchResult PostprocessModel(PreDetectResultBatch preResult, long startTime)
        {
            var res = BuildBatchResult(preResult.ImagePath, null, startTime);
            res.Results = RunBatchInfer(preResult);
            return res;
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
                DisposedSub();
                _postprocess.Dispose();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloSegBase()
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


        protected PreDetectResult PreprocessTime(Mat inputImage, SpeedResult speed)
        {
            _stopwatch.Restart();

            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;

            return preRes;
        }


        protected List<SegResult> PostProcessTime(OrtValue output0, OrtValue output1, PreDetectResult preDetect, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var res = _postprocess.PostProcessSync(output0, output1, preDetect);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return res;
        }
        protected Task BatchPostProcess(SegBatchResult[] batchResults, int idx, OrtValue output0, OrtValue output1, string imagePath, PreDetectResult preDetect, long startTime,
            IBatchProcessCallback<SegBatchResult> processCallback, Action<SegBatchResult> receiveAction)
        {
            return Task.Run(() =>
              {
                  using (output0)
                  using (output1)
                  {
                      var result = _postprocess.PostProcessAsync(output0, output1, preDetect);
                      batchResults[idx] = BuildBatchResult(imagePath, result, startTime);
                  }
                  _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);
              });

        }
        protected override SegBatchResult PostprocessChannel(InferModel inferModel)
        {
            try
            {
                using (inferModel.Output0)
                using (inferModel.Output1)
                {
                    var res = _postprocess.PostProcessSync(inferModel.Output0, inferModel.Output1, inferModel.PreDetectResult);
                    return BuildBatchResult(inferModel.ImagePath, res, inferModel.StartTime);
                }

            }
            finally
            {
                _inferModelPool.Value.Return(inferModel);
            }

        }

        protected override InferModel RunInfer(PreDetectResultBatch preResult, long startTime)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var results = RunInferenceBatch(preResult);
                isReturn = true;
                var data = _inferModelPool.Value.Rent();
                data.Initialize(results[0], results[1], preResult.ImagePath, startTime, preResult.PreResult);
                return data;

            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
                _preResultPool.Value.Return(preResult);
            }
        }
        protected override List<SegResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var results = RunInferenceBatch(preResult);
                using var output0 = results[0];
                using var output1 = results[1];
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcessSync(output0, output1, preResult.PreResult);

                return result;
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
                _preResultPool.Value.Return(preResult);
            }

        }

        protected override Task RunBatchInfer(SegBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime, IBatchProcessCallback<SegBatchResult> processCallback, Action<SegBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var results = RunInferenceBatch(item);
                isReturn = true;
                // 后处理
                var res = BatchPostProcess(batchResults, idx, results[0], results[1], item.ImagePath, item.PreResult, startTime, processCallback, receiveAction);

                return res;
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(item.Data);
                }
                _preResultPool.Value.Return(item);
            }
        }

        public void DrawSegments(Mat inputImage, List<SegResult> list)
        {
            foreach (var item in list)
            {
                YoloUtils.DrawDetections(inputImage, item.Box, item.Confidence, item.ClassName, _onnxModel.ColorPalette[item.ClassId]);
                DrawTransparentMask(inputImage, item.Mask, item.Box, _onnxModel.ColorPalette[item.ClassId]);
            }
        }

        /// <summary>
        /// 在原图上绘制半透明实例分割区域（推荐）
        /// </summary>
        /// <param name="image">原图（BGR）</param>
        /// <param name="binaryMask">二值mask（CV_8UC1，0/255）</param>
        /// <param name="color">显示颜色（BGR）</param>
        /// <param name="alpha">透明度：0~1，推荐 0.3~0.6</param>
        public static void DrawTransparentMask(Mat image, Mat binaryMask, Rect box, Scalar color, double alpha = 0.4)
        {
            Rect validRect = new Rect(
                Math.Max(0, box.X),
                Math.Max(0, box.Y),
                Math.Min(box.Width, image.Width - box.X),
                Math.Min(box.Height, image.Height - box.Y));

            using Mat colorMat = new Mat(validRect.Size, MatType.CV_8UC3, color);

            using Mat roi = new Mat(image, validRect);

            using Mat blended = new Mat();
            Cv2.AddWeighted(roi, alpha, colorMat, alpha, 0, blended);

            blended.CopyTo(roi, binaryMask);

        }


        public IYoloProcessAsync<PreDetectResultBatch, SegResult> GetYoloProcessAsync()
        {
            return this;
        }

    }
}
