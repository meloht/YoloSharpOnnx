using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public class SegResult : IYoloResult, IYoloSummary<SegResult>
    {
        public Rect Box { get; set; }
        public string ClassName { get; set; }
        public int ClassId { get; set; }
        public float Confidence { get; set; }

        public byte[] MaskBytes { get; set; }


        static string IYoloSummary<SegResult>.Describe(List<SegResult> predictResults)
        {
            return predictResults.Summary();
        }

        public override string ToString()
        {
            return $"{ClassName} {Confidence}";
        }
    }
}
