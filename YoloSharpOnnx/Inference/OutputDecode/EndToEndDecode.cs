using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.OutputDecode
{
    internal static class EndToEndDecode
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Rect Decode(ReadOnlySpan<float> output0, int offset, PreDetectResult preResult)
        {
            // 读取6个基础属性 YOLOv26 默认输出通常是 [x1, y1, x2, y2]
            float x1 = (output0[offset + 0] - preResult.PadX) / preResult.Scale;
            float y1 = (output0[offset + 1] - preResult.PadY) / preResult.Scale;
            float x2 = (output0[offset + 2] - preResult.PadX) / preResult.Scale;
            float y2 = (output0[offset + 3] - preResult.PadY) / preResult.Scale;


            int x = (int)Math.Max(x1, 0);
            int y = (int)Math.Max(y1, 0);

            int w = Math.Min((int)(x2 - x1), preResult.ImageWidth - x);
            int h = Math.Min((int)(y2 - y1), preResult.ImageHeight - y);

            Rect box = new Rect(x, y, w, h);

            return box;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Rect RTDETRDecode(ReadOnlySpan<float> output0, int offset, int inputWidth, int inputHeight, PreDetectResult preResult)
        {
            float cx = output0[offset + 0] * inputWidth;
            float cy = output0[offset + 1] * inputHeight;
            float w = output0[offset + 2] * inputWidth;
            float h = output0[offset + 3] * inputHeight;

            int x = (int)((cx - w * 0.5f - preResult.PadX) / preResult.Scale);
            int y = (int)((cy - h * 0.5f - preResult.PadY) / preResult.Scale);
            int width = (int)(w / preResult.Scale);
            int height = (int)(h / preResult.Scale);

            Rect box = new Rect(x, y, width, height);

            return box;
        }
    }
}
