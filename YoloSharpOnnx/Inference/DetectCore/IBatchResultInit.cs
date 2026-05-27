using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal interface IBatchResultInit<TResult>
    {
        void Initialize(string imagePath, List<TResult> results, long timestamp);
    }
}
