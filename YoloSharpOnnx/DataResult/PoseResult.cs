using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public class PoseResult : IYoloResult, IYoloSummary<PoseResult>
    {
        public Rect Box { get; set; }
        public string ClassName { get; set; }
        public int ClassId { get; set; }
        public float Confidence { get; set; }

        public PosePoint[] KeyPoints { get; set; }

        static string IYoloSummary<PoseResult>.Describe(List<PoseResult> predictResults)
        {
            return predictResults.Summary();
        }

        public override string ToString()
        {
            return $"{ClassName} {Confidence}";
        }
    }
}
