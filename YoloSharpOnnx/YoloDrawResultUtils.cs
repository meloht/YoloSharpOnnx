using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public static class YoloDrawResultUtils
    {
        public static void DrawClassification(Mat img, List<ClsResult> results)
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

            DrawTransparentRectFast(img, rect, Scalar.Black, 0.5f);

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

        public static void DrawDetections(Mat inputImage, List<DetectionResult> list, Scalar[] colorPalette)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassName, colorPalette[item.ClassId]);
            }
        }

        public static void DrawObbs(Mat image, List<ObbResult> results, Scalar[] colorPalette)
        {
            foreach (var pred in results)
            {
                // 1. 实例化 OpenCV 旋转矩形
                var color = colorPalette[pred.ClassId];
                RotatedRect rotatedRect = new RotatedRect(pred.Center, new Size2f(pred.Width, pred.Height), pred.Angle);

                // 2. 极其方便：直接获取旋转矩形的 4 个 Point2f 顶点
                Point2f[] verticesF = rotatedRect.Points();
                OpenCvSharp.Point[] vertices = new OpenCvSharp.Point[4];
                for (int i = 0; i < 4; i++)
                {
                    vertices[i] = new OpenCvSharp.Point((int)Math.Round(verticesF[i].X), (int)Math.Round(verticesF[i].Y));
                }
                int thickness = Math.Clamp((int)Math.Min(pred.Width, pred.Height) / 50, 1, 2);
                // 3. 绘制多边形闭合线圈
                Cv2.Polylines(image, [vertices], isClosed: true, color: color, thickness: thickness, lineType: LineTypes.AntiAlias);

                // 4. 绘制文本标签（选在第一个顶点附近）
                DrawLabel(image, pred.Confidence, pred.ClassName, vertices[0], (int)pred.Width, (int)pred.Height, color);
            }
        }

        public static void DrawPoses(Mat image, List<PoseResult> results, Scalar[] colorPalette, YoloConfig config)
        {
            foreach (var det in results)
            {
                DrawDetections(image, det.Box, det.Confidence, det.ClassName, colorPalette[det.ClassId]);

                foreach (var kp in det.KeyPoints)
                {
                    if (kp.Confidence < config.KeypointConfidence)
                        continue;
                    int x = (int)Math.Round(kp.X);
                    int y = (int)Math.Round(kp.Y);
                    if (kp.X <= 0 || kp.Y <= 0 || kp.X >= image.Width || kp.Y >= image.Height)
                    {
                        continue;
                    }
                    Cv2.Circle(image, new Point(x, y), config.KeypointRadius, config.Skeleton.GetKeypointColor(kp.Index), -1, lineType: LineTypes.AntiAlias);

                }

                for (int i = 0; i < config.Skeleton.ConnectionCount; i++)
                {
                    var p1 = config.Skeleton.GetKeypoint1(i, det.KeyPoints);
                    var p2 = config.Skeleton.GetKeypoint2(i, det.KeyPoints);

                    if (p1.Confidence < config.KeypointConfidence || p2.Confidence < config.KeypointConfidence)
                        continue;

                    int x1 = (int)Math.Round(p1.X);
                    int y1 = (int)Math.Round(p1.Y);
                    int x2 = (int)Math.Round(p2.X);
                    int y2 = (int)Math.Round(p2.Y);

                    if (x1 <= 0 || y1 <= 0 || x1 >= image.Width || y1 >= image.Height)
                    {
                        continue;
                    }
                    if (x2 <= 0 || y2 <= 0 || x2 >= image.Width || y2 >= image.Height)
                    {
                        continue;
                    }

                    Cv2.Line(image, new OpenCvSharp.Point(x1, y1), new Point(x2, y2), config.Skeleton.GetLineColor(i), config.KeypointLineThickness, lineType: LineTypes.AntiAlias);
                }

            }
        }

        public static void DrawDetections(Mat img, Rect box, float score, string className, Scalar color)
        {
            int thickness = Math.Clamp(Math.Min(box.Height, box.Width) / 50, 1, 2);
            // 绘制边界框
            Cv2.Rectangle(img, box, color, thickness);

            DrawLabel(img, score, className, box.Location, box.Width, box.Height, color);
        }

        public static void DrawLabel(Mat img, float score, string className, OpenCvSharp.Point box, int boxW, int boxH, Scalar color)
        {
            double fontScale = Math.Clamp(Math.Min(boxH, boxW) / 50.0, 0.3, 1.0);

            int height = img.Height;
            int width = img.Width;

            // 绘制标签
            string label = $"{className}: {score:F2}";
            int fontThick = Math.Max(1, (int)(fontScale * 2));
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, fontThick, out int baseline);

            int padding = 2;
            int margin = 1;
            int x = box.X - margin;
            int y = Math.Max(0, box.Y - textSize.Height - padding * 2 - baseline);

            x = Math.Max(0, Math.Min(x, img.Width - textSize.Width - padding * 2));

            int h = textSize.Height + baseline + padding * 2;
            int w = textSize.Width + padding * 2;
            Rect labelRect = new Rect(x, y, w, h);

            if (labelRect.Bottom > img.Height)
            {
                labelRect.Y = Math.Max(0, img.Height - labelRect.Height);
            }

            DrawTransparentRectFast(img, labelRect, color, 0.6f);

            // 标签文本
            Cv2.PutText(img, label, new OpenCvSharp.Point(labelRect.X + padding, labelRect.Y + textSize.Height + padding), HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }

        public static void DrawTransparentRect(Mat img, Rect rect, Scalar color, double alpha)
        {
            rect = rect.Intersect(new Rect(0, 0, img.Width, img.Height));
            if (rect.Width <= 0 || rect.Height <= 0) return;

            using var roi = new Mat(img, rect);
            using var overlay = new Mat(roi.Size(), roi.Type(), color);

            Cv2.AddWeighted(overlay, alpha, roi, 1 - alpha, 0, roi);
        }

        public static unsafe void DrawTransparentRectFast(Mat img, Rect rect, Scalar color, float alpha)
        {
            rect = rect.Intersect(new Rect(0, 0, img.Width, img.Height));

            if (rect.Width <= 0 || rect.Height <= 0)
                return;

            int channels = img.Channels();

            byte b = (byte)color.Val0;
            byte g = (byte)color.Val1;
            byte r = (byte)color.Val2;

            float inv = 1f - alpha;

            byte* ptr = (byte*)img.DataPointer;

            for (int y = rect.Y; y < rect.Bottom; y++)
            {
                byte* row = ptr + y * img.Step();

                for (int x = rect.X; x < rect.Right; x++)
                {
                    byte* p = row + x * channels;

                    p[0] = (byte)(p[0] * inv + b * alpha);
                    p[1] = (byte)(p[1] * inv + g * alpha);
                    p[2] = (byte)(p[2] * inv + r * alpha);
                }
            }
        }
        public static void DrawLabel(Mat img, float score, string className, OpenCvSharp.Point box, Scalar color)
        {
            double fontScale = 1.0;
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
            //Cv2.Rectangle(img,
            //    new OpenCvSharp.Point(x - 1, y - 8 - textSize.Height),
            //    new OpenCvSharp.Point(x + textSize.Width, y + baseline),
            //    color, -1);

            DrawTransparentRectFast(img, new Rect(x - 1, y - 8 - textSize.Height, textSize.Width + 2, textSize.Height + baseline + 8), color, 0.5f);

            // 标签文本
            Cv2.PutText(img, label, new OpenCvSharp.Point(x + 1, y), HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }

        public static void DrawSegments(Mat inputImage, List<SegResult> list, Scalar[] colorPalette)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassName, colorPalette[item.ClassId]);
                DrawTransparentMask(inputImage, item.PackMask, item.Box, colorPalette[item.ClassId]);
            }
        }

        /// <summary>
        /// 直接绘制 PackedMask
        /// </summary>
        /// <param name="image"></param>
        /// <param name="packedMask"></param>
        /// <param name="box"></param>
        /// <param name="color"></param>
        /// <param name="alpha"></param>
        public static unsafe void DrawTransparentMask(Mat image, byte[] packedMask, Rect box, Scalar color, float alpha = 0.4f)
        {
            Rect imageRect = new Rect(0, 0, image.Width, image.Height);

            Rect roiRect = box & imageRect;

            if (roiRect.Width <= 0 ||
                roiRect.Height <= 0)
                return;

            using Mat roi = new Mat(image, roiRect);

            int width = roiRect.Width;
            int height = roiRect.Height;

            long step = roi.Step();
            byte* ptr = (byte*)roi.DataPointer;
            int channels = roi.Channels();

            fixed (byte* maskPtr = packedMask)
            {
                byte* mask = maskPtr;

                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = ptr + y * step;

                    int offset = y * width;

                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = offset + x;
                        byte packed = mask[pixelIndex >> 3];

                        if ((packed & (1 << (pixelIndex & 7))) == 0)
                            continue;

                        byte* pixel = rowPtr + x * channels;

                        pixel[0] = (byte)(pixel[0] * (1f - alpha) + color.Val0 * alpha);
                        pixel[1] = (byte)(pixel[1] * (1f - alpha) + color.Val1 * alpha);
                        pixel[2] = (byte)(pixel[2] * (1f - alpha) + color.Val2 * alpha);

                    }
                }
            }
        }
    }
}
