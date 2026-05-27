using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Classify.Models;

namespace YoloSharpOnnx.Inference
{
    internal interface IBatchPreChannelResult<TBatchPreResult, TAsyncResult> : IDisposable
    {
        void Initialize(Guid guid, TBatchPreResult preResult, IBatchProcessCallback<TAsyncResult> Callback, Action<TAsyncResult> ReceiveAction);

        IBatchProcessCallback<TAsyncResult> Callback { get; set; }
        Action<TAsyncResult> ReceiveAction { get; set; }
    }
}
