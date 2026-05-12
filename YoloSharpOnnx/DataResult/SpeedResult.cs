using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public class SpeedResult
    {
        public long Preprocess { get; set; }

        public long Inference { get; set; }

        public long Postprocess { get; set; }

        public long TotalTime { get; set; }

      

        public void SumTotal()
        {
            TotalTime = Preprocess + Inference + Postprocess;
        }

        public override string ToString()
        {
            return $"Total:{TotalTime}ms, Preprocess: {Preprocess}ms, Inference: {Inference}ms, Postprocess: {Postprocess}ms";
        }
    }
}
