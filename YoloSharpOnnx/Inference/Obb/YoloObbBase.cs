using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;
using static System.Formats.Asn1.AsnWriter;

namespace YoloSharpOnnx.Inference.Obb
{
    internal abstract class YoloObbBase : OnnxInferenceCore<ObbResult, PreDetectResultBatch, ObbBatchResult>, IYoloProcessAsync<PreDetectResultBatch, ObbResult>
    {
        protected readonly IObbPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;
        private bool disposedValue;

        protected abstract void DisposedSub();
        protected abstract OrtValue RunInferenceBatch(PreDetectResultBatch preResult);

        protected YoloObbBase(InferenceSession session, SessionOptions options, IObbPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config) : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
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
        // ~YoloObbBase()
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

        private static ObbBatchResult BuildBatchResult(string imagePath, List<ObbResult> results, long timestamp)
        {
            return new ObbBatchResult(imagePath, results, timestamp);
        }

        protected override PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            var data = _preResultPool.Value.Rent();
            data.Initialize(preRes, imagePath, imageBatchData);
            return data;
        }

        public List<ObbResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        protected override ObbBatchResult PostprocessModel(PreDetectResultBatch preResult, long startTime)
        {
            var res = BuildBatchResult(preResult.ImagePath, null, startTime);
            res.Results = RunBatchInfer(preResult);
            return res;
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


        protected List<ObbResult> PostProcessTime(OrtValue output0, PreDetectResult preDetect, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var res = _postprocess.PostProcessSync(output0, preDetect);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return res;
        }
        protected Task BatchPostProcess(ObbBatchResult[] batchResults, int idx, OrtValue output0, string imagePath, PreDetectResult preDetect, long startTime,
            IBatchProcessCallback<ObbBatchResult> processCallback, Action<ObbBatchResult> receiveAction)
        {
            return Task.Run(() =>
            {
                using (output0)
                {
                    var result = _postprocess.PostProcessAsync(output0, preDetect);
                    batchResults[idx] = BuildBatchResult(imagePath, result, startTime);
                }
                _ = InferCompleteAsync(batchResults[idx], processCallback, receiveAction);
            });

        }
        protected override ObbBatchResult PostprocessChannel(InferModel inferModel)
        {
            try
            {
                using (inferModel.Output0)
                {
                    var res = _postprocess.PostProcessSync(inferModel.Output0, inferModel.PreDetectResult);
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
                data.Initialize(results, null, preResult.ImagePath, startTime, preResult.PreResult);
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
        protected override List<ObbResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                using var results = RunInferenceBatch(preResult);
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcessSync(results, preResult.PreResult);

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

        protected override Task RunBatchInfer(ObbBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime, IBatchProcessCallback<ObbBatchResult> processCallback, Action<ObbBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                // 执行推理
                var results = RunInferenceBatch(item);
                isReturn = true;
                // 后处理
                var res = BatchPostProcess(batchResults, idx, results, item.ImagePath, item.PreResult, startTime, processCallback, receiveAction);

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


        public IYoloProcessAsync<PreDetectResultBatch, ObbResult> GetYoloProcessAsync()
        {
            return this;
        }

        public void DrawObbs(Mat image, List<ObbResult> results)
        {
            foreach (var pred in results)
            {
                // 1. 实例化 OpenCV 旋转矩形
                var color = _onnxModel.ColorPalette[pred.ClassId];
                RotatedRect rotatedRect = new RotatedRect(pred.Center, new Size2f(pred.Width, pred.Height), pred.Angle);

                // 2. 极其方便：直接获取旋转矩形的 4 个 Point2f 顶点
                Point2f[] verticesF = rotatedRect.Points();
                Point[] vertices = new Point[4];
                for (int i = 0; i < 4; i++)
                {
                    vertices[i] = new Point((int)Math.Round(verticesF[i].X), (int)Math.Round(verticesF[i].Y));
                }
                int thickness = Math.Clamp((int)Math.Min(pred.Width, pred.Height) / 50, 1, 2);
                // 3. 绘制多边形闭合线圈
                Cv2.Polylines(image, [vertices], isClosed: true, color: color, thickness: thickness, lineType: LineTypes.AntiAlias);


                // 4. 绘制文本标签（选在第一个顶点附近）
                YoloUtils.DrawLabel(image, pred.Confidence, pred.ClassName, vertices[0], (int)pred.Width, (int)pred.Height, color);
            }
        }
    }
}
