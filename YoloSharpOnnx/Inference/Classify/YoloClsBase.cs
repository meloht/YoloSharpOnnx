using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    internal abstract class YoloClsBase : OnnxInferenceCore<ClsResult, PreClsResultBatch, ClsBatchResult>, IYoloProcessAsync<PreClsResultBatch, ClsResult>
    {
        protected readonly IClsPostprocess _postprocess;
        protected readonly IClsPreprocess _preprocess;
        private bool disposedValue;


        protected abstract void DisposedSub();
        protected abstract OrtValue RunInferenceBatch(PreClsResultBatch preResult);
        public YoloClsBase(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;

        }
        protected void PreprocessTime(Mat inputImage, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 预处理图像
            _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;

        }


        protected List<ClsResult> PostProcessTime(OrtValue output0, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var result = _postprocess.PostProcess(output0);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return result;
        }
        protected Task BatchPostProcess(ClsBatchResult[] batchResults, int idx, OrtValue output0, string imagePath, long startTime, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            return Task.Run(() =>
              {
                  using (output0)
                  {
                      var result = _postprocess.PostProcess(output0);
                      batchResults[idx] = BuildBatchResult(imagePath, result, startTime);
                  }

                  _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);

              });

        }
        protected override ClsBatchResult PostprocessChannel(InferModel inferModel)
        {
            try
            {
                using (inferModel.Output0)
                {
                    var res = _postprocess.PostProcess(inferModel.Output0);
                    return BuildBatchResult(inferModel.ImagePath, res, inferModel.StartTime);
                }
            }
            finally
            {

                _inferModelPool.Value.Return(inferModel);

            }

        }

        protected override List<ClsResult> RunBatchInfer(PreClsResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                using var output0 = RunInferenceBatch(preResult);
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
                _preResultPool.Value.Return(preResult);
            }

        }

        protected override Task RunBatchInfer(ClsBatchResult[] batchResults, int idx, PreClsResultBatch item, long startTime,
            IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var ortValue = RunInferenceBatch(item);
                isReturn = true;
                // 后处理
                return BatchPostProcess(batchResults, idx, ortValue, item.ImagePath, startTime, processCallback, receiveAction);
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

        protected override InferModel RunInfer(PreClsResultBatch preResult, long startTime)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var ortValue = RunInferenceBatch(preResult);
                isReturn = true;

                var inferModel = _inferModelPool.Value.Rent();
                inferModel.Initialize(ortValue, null, preResult.ImagePath, startTime, new PreDetectResult());
                return inferModel;
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

        protected override PreClsResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            PreClsResultBatch preClsResult = _preResultPool.Value.Rent();
            preClsResult.Initialize(imagePath, imageBatchData);
            return preClsResult;
        }

        private static ClsBatchResult BuildBatchResult(string imagePath, List<ClsResult> results, long timestamp)
        {
            return new ClsBatchResult(imagePath, results, timestamp);
        }

        public List<ClsResult> RunBatch(PreClsResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }
        protected override ClsBatchResult PostprocessModel(PreClsResultBatch preResult, long startTime)
        {
            var res = BuildBatchResult(preResult.ImagePath, null, startTime);
            res.Results = RunBatchInfer(preResult);
            return res;
        }

        public IYoloProcessAsync<PreClsResultBatch, ClsResult> GetYoloProcessAsync()
        {
            return this;
        }

        public void DrawClassification(Mat img, List<ClsResult> results)
        {
            if (results == null || results.Count == 0)
                return;

            int x = 10;
            int yStart = 30;
            int lineGap = 5;

            var font = HersheyFonts.HersheySimplex;
            double fontScale = 0.8;
            int thickness = 1;

            // ===== 1. 生成所有文本 =====
            string[] texts = results.Select(r => $"{r.ClassName} {r.Confidence:0.00}").ToArray();

            // ===== 2. 计算最大宽度 & 总高度 =====
            int maxWidth = 0;
            int totalHeight = 0;

            foreach (var text in texts)
            {
                var size = Cv2.GetTextSize(text, font, fontScale, thickness, out int baseline);
                maxWidth = Math.Max(maxWidth, size.Width);
                totalHeight += size.Height + lineGap;
            }

            totalHeight -= lineGap; // 去掉最后一个 gap

            // ===== 3. 绘制整体半透明背景=====
            var rect = new Rect(
                x - 5,
                yStart - Cv2.GetTextSize(texts[0], font, fontScale, thickness, out _).Height - 5,
                maxWidth + 10,
                totalHeight + 10
            );

            YoloUtils.DrawTransparentRect(img, rect, Scalar.Black, 0.5);

            // ===== 4. 逐行绘制文本 =====
            int y = yStart;

            foreach (var text in texts)
            {
                var size = Cv2.GetTextSize(text, font, fontScale, thickness, out _);

                Cv2.PutText(img, text,
                    new OpenCvSharp.Point(x, y),
                    font, fontScale,
                    Scalar.White,
                    thickness,
                    LineTypes.AntiAlias);

                y += size.Height + lineGap;
            }
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
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloClsBase()
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
    }
}
