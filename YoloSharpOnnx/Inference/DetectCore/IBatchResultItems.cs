using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal interface IBatchResultItems<TResult>
    {
        List<TResult> Results { get; set; }
    }
}
