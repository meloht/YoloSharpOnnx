using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public interface IYoloResult
    {
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string ClassName { get; set; }
    }
}
