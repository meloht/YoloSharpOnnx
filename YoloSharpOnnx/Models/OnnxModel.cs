using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public class OnnxModel
    {
        public string InputName { get; set; }

        public string OutputName0 { get; set; }

        public string OutputName1 { get; set; }

        public int InputWidth { get; set; }
        public int InputHeight { get; set; }

        public long[] InputShape { get; set; }
        public long[] OutputShape0 { get; set; }
        public long[] OutputShape1 { get; set; }
        public long InputShapeSize { get; set; }
        public long OutputShapeSize0 { get; set; }
        public long OutputShapeSize1 { get; set; }
        public LabelModel[] Labels { get; set; }

        public bool IsEndToEnd { get; set; }
        public int BoxNum { get; set; }

        public Scalar[] ColorPalette { get; set; }

        public long InputSizeInBytes { get; set; }
        public long OutputSizeInBytes0 { get; set; }
        public long OutputSizeInBytes1 { get; set; }

        public DeviceType DeviceType { get; set; }

        public ModelType ModelType { get; set; }
    }
}
