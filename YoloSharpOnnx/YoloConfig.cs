using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloConfig
    {
        private int _batchPoolSize = 20;
        private float _confidence = 0.3f;
        private float _iou = 0.4f;
        public float Confidence
        {
            get { return _confidence; }
            set 
            {
                if (value < 0 && value > 1)
                {
                    throw new ArgumentException("The Confidence must be between 0 and 1");
                }
                _confidence = value; 
            }
        }


        public float IoU
        {
            get { return _iou; }
            set
            {
                if (value < 0 && value > 1)
                {
                    throw new ArgumentException("The IoU must be between 0 and 1");
                }
                _iou = value;
            }
        }

        public InterpolationFlags ResizeAlgorithm { get; set; }

        public string[] ImageExtsBatch { get; set; } = [".jpg", ".png"];


        public int BatchPoolSize
        {
            get { return _batchPoolSize; }
            set
            {
                if (value < 1 && value > 100)
                {
                    throw new ArgumentException("The BatchPoolSize must be between 1 and 100");
                }
                _batchPoolSize = value;
            }
        }

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
