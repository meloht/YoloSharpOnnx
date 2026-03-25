using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public class OnnxModel
    {
        public string InputName { get; set; }

        public string OutputName { get; set; }

        public int InputWidth { get; set; }
        public int InputHeight { get; set; }

        public long[] InputShape { get; set; }
        public long[] OutputShape { get; set; }
        public long InputShapeSize { get; set; }
        public long OutputShapeSize { get; set; }
        public LabelModel[] Labels { get; set; }

        public bool IsEndToEnd { get; set; }
        public int BoxNum { get; set; }

        public Scalar[] ColorPalette { get; set; }
    }
}
