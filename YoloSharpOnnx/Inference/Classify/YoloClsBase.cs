using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public class YoloClsBase : OnnxInferenceCore
    {
        protected readonly IClsPostprocess _postprocess;
        protected readonly IClsPreprocess _preprocess;
        public YoloClsBase(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config, IClsPostprocess postprocess, IClsPreprocess preprocess) 
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;
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
    }
}
