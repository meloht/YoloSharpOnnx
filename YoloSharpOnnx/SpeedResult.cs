using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public struct SpeedResult
    {
        public TimeSpan Preprocess { get; }

        public TimeSpan Inference { get; }

        public TimeSpan Postprocess { get; }

        public TimeSpan TotalTime { get; }

        public SpeedResult(TimeSpan preprocess, TimeSpan inference, TimeSpan postprocess)
        {
            Preprocess = preprocess;
            Inference = inference;
            Postprocess = postprocess;
            TotalTime = preprocess + inference + postprocess;
        }

        public override string ToString()
        {
            return $"Total:{TotalTime.TotalSeconds}, Preprocess: {Preprocess.TotalSeconds}, Inference: {Inference.TotalSeconds}, Postprocess: {Postprocess.TotalSeconds}";
        }
    }
}
