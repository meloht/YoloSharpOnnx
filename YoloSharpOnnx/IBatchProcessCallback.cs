using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public interface IBatchProcessCallback<TBatchResult>
    {
        void ReceiveProcessResult(TBatchResult result);
    }
}
