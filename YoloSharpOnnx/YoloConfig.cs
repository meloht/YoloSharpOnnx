using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloConfig
    {
        public static readonly YoloConfig Default = new();
        public float Confidence { get; set; }

        public float IoU { get; set; }

        public InterpolationFlags ResizeAlgorithm { get; set; }

        public string[] ImageExtsBatch { get; set; } = [".jpg", ".png"];

        public YoloConfig(float confidence, float iou, InterpolationFlags resizeAlgorithm)
        {
            this.Confidence = confidence;
            this.IoU = iou;
            this.ResizeAlgorithm = resizeAlgorithm;
        }
        /// <summary>
        /// default ResizeAlgorithm=InterpolationFlags.Linear
        /// </summary>
        /// <param name="confidence"></param>
        /// <param name="iou"></param>
        public YoloConfig(float confidence, float iou) : this(confidence, iou, InterpolationFlags.Linear)
        {

        }

        /// <summary>
        /// default IoU=0.4 ,ResizeAlgorithm=InterpolationFlags.Linear
        /// </summary>
        /// <param name="confidence"></param>
        public YoloConfig(float confidence) : this(confidence, 0.4f, InterpolationFlags.Linear)
        {

        }
        /// <summary>
        /// default confidence=0.3, IoU=0.4 ,ResizeAlgorithm=InterpolationFlags.Linear
        /// </summary>
        public YoloConfig() : this(0.3f, 0.4f, InterpolationFlags.Linear)
        {

        }
    }
}
