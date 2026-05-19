using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public interface IRunBatch<TResult, TBatchPreResult>
    {
        List<TResult> RunBatch(TBatchPreResult preResult);
        void ReturnBatchPreResult(TBatchPreResult preResult);

        
    }
}
