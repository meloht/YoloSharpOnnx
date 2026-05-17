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
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public abstract class YoloClsBase : OnnxInferenceCore<ClsResult, PreClsResultBatch, ClsBatchResult>, 
        IBatchProcess<ClsResult, PreClsResultBatch, ClsBatchResult>, IYoloProcessAsync<PreClsResultBatch>
    {
        protected readonly IClsPostprocess _postprocess;
        protected readonly IClsPreprocess _preprocess;
        private bool disposedValue;

       
        protected abstract void DisposedSub();
        public YoloClsBase(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
            InitBatchProcess(this);
        }
        protected void PreprocessTime(Mat inputImage, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 预处理图像
            _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;

        }


        protected List<ClsResult> PostProcessTime(OrtValue output0,  SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var result = _postprocess.PostProcess(output0);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return result;
        }
        protected void BatchPostProcess(ClsBatchResult[] batchResults, int idx, OrtValue output0, PreClsResultBatch item, long startTime, IBatchProcessCallback<ClsBatchResult> processCallback, Action<ClsBatchResult> receiveAction)
        {
            using (output0)
            {
                var result = _postprocess.PostProcess(output0);
                batchResults[idx] = BuildBatchResult(item, result, startTime);
            }

            _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);
        }

        public PreClsResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            _preprocess.PreprocessImage(inputImage, imageBatchData.ResizedImg, imageBatchData.FixedBuffer);
            return new PreClsResultBatch(imagePath, imageBatchData);
        }

        public ClsBatchResult BuildBatchResult(PreClsResultBatch batchPreResult, List<ClsResult> results, long timestamp)
        {
            return new ClsBatchResult(batchPreResult.ImagePath, results, timestamp);
        }

        public List<ClsResult> RunBatch(PreClsResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        public IRunBatch<ClsResult, PreClsResultBatch> GetRunBatch()
        {
            return this;
        }
        public IYoloProcessAsync<PreClsResultBatch> GetYoloProcessAsync()
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

            DrawTransparentRect(img, rect, Scalar.Black, 0.5);

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

        private static void DrawTransparentRect(Mat img, Rect rect, Scalar color, double alpha)
        {
            rect = rect.Intersect(new Rect(0, 0, img.Width, img.Height));
            if (rect.Width <= 0 || rect.Height <= 0) return;

            using var roi = new Mat(img, rect);
            using var overlay = new Mat(roi.Size(), roi.Type(), color);

            Cv2.AddWeighted(overlay, alpha, roi, 1 - alpha, 0, roi);
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
