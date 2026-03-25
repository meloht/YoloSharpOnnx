using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public struct PreResult
    {
        public int ImageHeight { get; set; }
        public int ImageWidth { get; set; }
        public int PadY { get; set; }
        public int PadX { get; set; }
        public float Scale { get; set; }
        public PreResult(int imageHeight, int imageWidth, int padY, int padX, float scale)
        {
            ImageHeight = imageHeight;
            ImageWidth = imageWidth;
            PadY = padY;
            PadX = padX;
            Scale = scale;
        }

    }
}
