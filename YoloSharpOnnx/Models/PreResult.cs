using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public struct PreResult
    {
        public float[] OutData { get; set; }

        public int PadY { get; set; }
        public int PadX { get; set; }
        public float Scale { get; set; }
        public PreResult(float[] outData, int padY, int padX, float scale)
        {
            OutData = outData;
            PadY = padY;
            PadX = padX;
            Scale = scale;
        }
    }
}
