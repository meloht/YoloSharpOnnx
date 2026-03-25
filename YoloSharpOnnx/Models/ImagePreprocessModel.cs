using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public record ImagePreprocessModel(int ImageHeight, int ImageWidth, float[] Data, int PadY, int PadX, float Scale);
}
