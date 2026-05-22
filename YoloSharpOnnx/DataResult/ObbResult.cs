using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public class ObbResult: IYoloResult, IYoloSummary<ObbResult>
    {
        public Point2f Center { get; set; }

        public float Width { get; set; }

        public float Height { get; set; }

        public string ClassName { get; set; }
        public int ClassId { get; set; }
        public float Confidence { get; set; }

        public float Angle { get; set; }

        static string IYoloSummary<ObbResult>.Describe(List<ObbResult> predictResults)
        {
            return predictResults.Summary();
        }

        public override string ToString()
        {
            return $"{ClassName} {Confidence}";
        }
    }
}
