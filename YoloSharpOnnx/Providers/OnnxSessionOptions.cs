using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Providers
{
    public class OnnxSessionOptions
    {
        public bool EnableMemoryPattern { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
    }
}
