using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Classify.Models;

namespace YoloSharpOnnx.Inference
{
    internal interface IBatchPreChannelResult<T> : IDisposable
    {
        void Initialize(Guid guid, T preResult);
    }
}
