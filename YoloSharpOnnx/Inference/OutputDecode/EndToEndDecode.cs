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
    internal class EndToEndDecode
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Rect Decode(ReadOnlySpan<float> output0, int offset, PreDetectResult preResult)
        {
            // 读取6个基础属性 YOLOv26 默认输出通常是 [x1, y1, x2, y2]
            float x1 = (output0[offset + 0] - preResult.PadX) / preResult.Scale;
            float y1 = (output0[offset + 1] - preResult.PadY) / preResult.Scale;
            float x2 = (output0[offset + 2] - preResult.PadX) / preResult.Scale;
            float y2 = (output0[offset + 3] - preResult.PadY) / preResult.Scale;


            Rect box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));

            return box;
        }

    }
}
