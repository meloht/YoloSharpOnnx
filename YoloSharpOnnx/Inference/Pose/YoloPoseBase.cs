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
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    public abstract class YoloPoseBase : OnnxInferenceCore<PoseResult, PreDetectResultBatch, PoseBatchResult>, IYoloProcessAsync<PreDetectResultBatch, PoseResult>
    {

        protected readonly IPosePostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;
        private bool disposedValue;

        protected abstract void DisposedSub();
        protected abstract OrtValue RunInferenceBatch(PreDetectResultBatch preResult);

        protected YoloPoseBase(InferenceSession session, SessionOptions options, IPosePostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
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
                //  _postprocess.Dispose();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloPoseBase()
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


        private static PoseBatchResult BuildBatchResult(string imagePath, List<PoseResult> results, long timestamp)
        {
            return new PoseBatchResult(imagePath, results, timestamp);
        }

        protected override PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizeMat, imageBatchData.FixedBuffer);
            var data = _preResultPool.Value.Rent();
            data.Initialize(preRes, imagePath, imageBatchData);
            return data;
        }

        public List<PoseResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        protected override PoseBatchResult PostprocessModel(PreDetectResultBatch preResult, long startTime)
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


        protected List<PoseResult> PostProcessTime(OrtValue output0, PreDetectResult preDetect, SpeedResult speed)
        {
            _stopwatch.Restart();
            // 后处理
            var res = _postprocess.PostProcessSync(output0, preDetect);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return res;
        }
        protected Task BatchPostProcess(PoseBatchResult[] batchResults, int idx, OrtValue output0, string imagePath, PreDetectResult preDetect, long startTime,
            IBatchProcessCallback<PoseBatchResult> processCallback, Action<PoseBatchResult> receiveAction)
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
        protected override PoseBatchResult PostprocessChannel(InferModel inferModel)
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
        protected override List<PoseResult> RunBatchInfer(PreDetectResultBatch preResult)
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

        protected override Task RunBatchInfer(PoseBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime, IBatchProcessCallback<PoseBatchResult> processCallback, Action<PoseBatchResult> receiveAction)
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


        public IYoloProcessAsync<PreDetectResultBatch, PoseResult> GetYoloProcessAsync()
        {
            return this;
        }



        public void DrawPoses(Mat image, List<PoseResult> results)
        {
            foreach (var det in results)
            {
                Cv2.Rectangle(image, det.Box, Scalar.Red, 2);

                foreach (var kp in det.KeyPoints)
                {
                    if (kp.Confidence < 0.5f)
                        continue;

                    Cv2.Circle(image, new Point(kp.X, kp.Y), 3, Scalar.Lime, -1);

                }

                //foreach (var pair in Skeleton)
                //{
                //    PosePoint p1 = det.KeyPoints[pair.Item1];
                //    PosePoint p2 = det.KeyPoints[pair.Item2];

                //    if (p1.Confidence < 0.5f ||
                //        p2.Confidence < 0.5f)
                //        continue;

                //    Cv2.Line(
                //        image,
                //        new Point(p1.X, p1.Y),
                //        new Point(p2.X, p2.Y),
                //        Scalar.Yellow,
                //        2);
                //}
            }
        }

    }
}
