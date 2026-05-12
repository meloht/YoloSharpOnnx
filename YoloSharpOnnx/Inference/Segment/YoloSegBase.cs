using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public abstract class YoloSegBase : OnnxInferenceCore<SegResult, PreDetectResultBatch, SegBatchResult>,
        IBatchProcess<SegResult, PreDetectResultBatch, SegBatchResult>, IYoloProcessAsync<PreDetectResultBatch>
    {
        protected readonly ISegPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;
        private bool disposedValue;

        protected abstract void DisposedSub();

        protected YoloSegBase(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
            InitBatchProcess(this);

        }

        public SegBatchResult BuildBatchResult(PreDetectResultBatch batchPreResult, List<SegResult> results, long timestamp)
        {
            return new SegBatchResult(batchPreResult.ImagePath, results, timestamp);
        }

        public PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizedImg, imageBatchData.FixedBuffer);
            return new PreDetectResultBatch(preRes, imagePath, imageBatchData);
        }

        public List<SegResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
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
            var res = _postprocess.PostProcess(output0, output1, preDetect);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return res;
        }


        public void DrawSegments(Mat inputImage, List<SegResult> list)
        {
            foreach (var res in list)
            {
                // 随机颜色
                var color = _onnxModel.ColorPalette[res.ClassId];

                // 画框
                Cv2.Rectangle(inputImage, res.Box, color, 2);

                // 掩码叠加（半透明）
                using Mat maskColor = new Mat();
                Cv2.CvtColor(res.Mask, maskColor, ColorConversionCodes.GRAY2BGR);
                maskColor.ConvertTo(maskColor, MatType.CV_8UC3, 255);
                Cv2.AddWeighted(inputImage, 0.7, maskColor, 0.3, 0, inputImage);

                // 文字
                string label = $"cls:{res.ClassId} {res.Confidence:F2}";
                Cv2.PutText(inputImage, label, new Point(res.Box.X, res.Box.Y - 5),
                    HersheyFonts.HersheySimplex, 0.5, color, 1);

            }
        }

        public IYoloProcessAsync<PreDetectResultBatch> GetYoloProcessAsync()
        {
            return this;
        }

        public IRunBatch<SegResult, PreDetectResultBatch> GetRunBatch()
        {
            return this;
        }
    }
}
