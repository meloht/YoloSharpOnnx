using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx.Inference.Detect
{
    public abstract class YoloDetectBase : OnnxInferenceCore<DetectionResult, PreDetectResultBatch, DetectionBatchResult>, IBatchProcess<DetectionResult, PreDetectResultBatch, DetectionBatchResult>
    {
        protected readonly IDetPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;

        public YoloDetectBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
            InitBatchProcess(this);
        }


        public List<DetectionResult> RunCore(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            // 执行推理
            var output0 = RunInference();

            // 后处理
            var result = _postprocess.PostProcess(output0, preRes, _config);

            AfterInference(output0);
            return result;
        }

        public YoloResult<DetectionResult> RunWithTimeCore(Mat inputImage)
        {

            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 执行推理
            var output0 = RunInference();

            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 后处理
            var res = _postprocess.PostProcess(output0, preRes, _config);
            AfterInference(output0);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }


        public PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizedImg, imageBatchData.FixedBuffer, _config.ResizeAlgorithm);
            return new PreDetectResultBatch(preRes, imagePath, imageBatchData);
        }

        public DetectionBatchResult BuildBatchResult(PreDetectResultBatch batchPreResult, List<DetectionResult> results, long timestamp)
        {
            return new DetectionBatchResult(batchPreResult.ImagePath, results, timestamp);
        }

        public List<DetectionResult> RunBatch(PreDetectResultBatch preResult)
        {
            return RunBatchInfer(preResult);
        }

        public IBatchProcess<DetectionResult, PreDetectResultBatch, DetectionBatchResult> GetYoloDetectAsync()
        {
            return this;
        }

        public void DrawDetections(Mat inputImage, List<DetectionResult> list)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassId, item.ClassName);
            }
        }
        public void DrawDetections(Mat img, Rect box, float score, int classId, string className)
        {
            var color = _onnxModel.ColorPalette[classId];

            double fontScale = 1.0;
            // 绘制边界框
            Cv2.Rectangle(img, box, color, 2);

            int height = img.Height;
            int width = img.Width;

            // 绘制标签
            string label = $"{className}: {score:F2}";
            int fontThick = 2;
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, fontThick, out int baseline);

            int x = box.X;
            int y = box.Y - 10; ;
            if (y < textSize.Height)
                y = box.Y + 10;

            if (x + textSize.Width > width)
            {
                x = x - (x + textSize.Width - width) - 4;
            }

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(x - 1, y - 8 - textSize.Height),
                new OpenCvSharp.Point(x + textSize.Width, y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, new Point(x + 1, y), HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }


    }
}
