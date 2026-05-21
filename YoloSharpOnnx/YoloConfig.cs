using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloConfig
    {
        private const float _defaultConfidence = 0.3f;
        private const float _defaultIoU = 0.4f;
        private int _batchPoolSize = 30;
        private float _confidence = _defaultConfidence;
        private float _iou = _defaultIoU;
        private int _asyncChannelTimeout = 5000;
        private int _clsTopK = 5;
        private ISkeleton _skeleton = new HumanSkeleton();

        public int KeypointRadius { get; set; } = 5;

        public int KeypointLineThickness { get; set; } = 2;

        public float KeypointConfidence { get; set; } = 0.25f;

        public ISkeleton Skeleton
        {
            get { return _skeleton; }
            set { _skeleton = value; }
        }
        public int ClassifyTopK
        {
            get { return _clsTopK; }
            set
            {
                if (value < 0 && value > 5)
                {
                    throw new ArgumentException("The ClassifyTopK must be between 1 and 5");
                }
                _clsTopK = value;
            }
        }
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
        /// <summary>
        /// default 5000 ms
        /// </summary>
        public int AsyncChannelTimeout
        {
            get { return _asyncChannelTimeout; }
            set
            {
                if (value < 1000)
                {
                    throw new ArgumentException("The AsyncChannelTimeout must be greater than 1000 ms");
                }
                _asyncChannelTimeout = value;
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
        public YoloConfig(float confidence) : this(confidence, _defaultIoU, InterpolationFlags.Linear)
        {

        }
        /// <summary>
        /// default confidence=0.3, IoU=0.4 ,ResizeAlgorithm=InterpolationFlags.Linear
        /// </summary>
        public YoloConfig() : this(_defaultConfidence, _defaultIoU, InterpolationFlags.Linear)
        {

        }
    }
}
