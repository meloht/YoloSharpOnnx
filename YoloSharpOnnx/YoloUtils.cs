using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;
using static System.Net.Mime.MediaTypeNames;

namespace YoloSharpOnnx
{
    public class YoloUtils
    {
        public static string GetDetectResult(List<DetectionResult> list)
        {
            return list.Summary();
        }
        public static string GetClsResult(List<ClsResult> list)
        {
            return list.Summary();
        }


        public static List<string> GetFilesFromDirectory(string path, string[] exts)
        {
            List<string> list = new List<string>();
            HashSet<string> set = new HashSet<string>(exts);
            GetFiles(list, path, set);
            return list;

        }

        public static BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = false,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };

            return channelOptions;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float ToDegree(float angle)
        {
            return angle * 180.0f / MathF.PI;
        }
        internal static void ClearList(PostResultArray resultArray)
        {
            resultArray.Boxes.Clear();
            resultArray.Scores.Clear();
            resultArray.ClassIds.Clear();
            resultArray.Ids?.Clear();
        }

        public static List<string> GetFilesFromListPaths(List<string> images, string[] exts)
        {

            List<string> list = new List<string>();
            HashSet<string> extSet = new HashSet<string>(exts);
            foreach (var item in images)
            {
                string ext = Path.GetExtension(item);
                string fileExt = ext.ToLower();
                if (extSet.Contains(fileExt))
                {
                    list.Add(item);
                }
            }
            return list;

        }

        public static void GetFiles(List<string> list, string path, HashSet<string> extSet)
        {
            DirectoryInfo directory = new DirectoryInfo(path);
            var files = directory.GetFiles();

            foreach (var item in files)
            {
                string fileExt = item.Extension.ToLower();
                if (extSet.Contains(fileExt))
                {
                    list.Add(item.FullName);
                }
            }
            var subDirectories = Directory.GetDirectories(path);

            foreach (string subDir in subDirectories)
            {
                GetFiles(list, subDir, extSet);
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

            DrawTransparentRect(img, labelRect, color, 0.6);

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

            DrawTransparentRect(img, new Rect(x - 1, y - 8 - textSize.Height, textSize.Width + 2, textSize.Height + baseline + 8), color, 0.5);

            // 标签文本
            Cv2.PutText(img, label, new OpenCvSharp.Point(x + 1, y), HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }


        public static unsafe void MatToBytes(Mat mat, byte[] buffer)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels();

            byte* ptr = (byte*)mat.DataPointer;
            fixed (byte* data = buffer)
            {
                int hw = width * height;

                long step = mat.Step();
                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = ptr + y * step;
                    int rowOffset = y * width;

                    for (int x = 0; x < width; x++)
                    {
                        byte* pixel = rowPtr + x * channels;
                        for (int c = 0; c < channels; c++)
                        {
                            int offset = hw * c + rowOffset + x;
                            data[offset] = pixel[c];
                        }

                    }
                }
            }

        }

        public static byte[] PackMask(byte[] src)
        {
            int packedLength = (src.Length + 7) >> 3;

            byte[] packed = new byte[packedLength];

            for (int i = 0; i < src.Length; i++)
            {
                if (src[i] != 0)
                {
                    packed[i >> 3] |= (byte)(1 << (i & 7));
                }
            }

            return packed;
        }

        public static byte[] UnpackMask(byte[] packed, int pixelCount)
        {
            byte[] dst = new byte[pixelCount];

            for (int i = 0; i < pixelCount; i++)
            {
                dst[i] =
                    (packed[i >> 3] & (1 << (i & 7))) != 0
                    ? (byte)255
                    : (byte)0;
            }

            return dst;
        }

        public static void UnpackMask(byte[] packed, byte[] dst, int pixelCount)
        {
            for (int i = 0; i < pixelCount; i++)
            {
                dst[i] =
                    (packed[i >> 3] & (1 << (i & 7))) != 0
                    ? (byte)255
                    : (byte)0;
            }
        }

    }
}
