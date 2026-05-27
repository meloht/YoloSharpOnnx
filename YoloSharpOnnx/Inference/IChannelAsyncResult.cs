using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    internal interface IChannelAsyncResult<T>
    {
        void Initialize(Guid guid, List<T> result, long timestamp);
    }
}
