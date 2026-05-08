using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public struct ClsResult
    {
        public string ClassName { get; set; }
        public int ClassId { get; set; }
        public float Confidence { get; set; }

        public ClsResult(string className, int classId, float confidence)
        {
            ClassName = className;
            ClassId = classId;
            Confidence = confidence;
        }
    }
}
